"""Storage layer for structured and unstructured data"""
import sqlite3
import chromadb
from chromadb.config import Settings
import json
import shutil
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from config import DB_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer
import uuid
import torch  # Required for embedding operations


class StorageManager:
    """Manages both SQLite (structured) and ChromaDB (vector) storage"""
    
    def __init__(self):
        self.conn = None
        self.vector_db = None
        self.embedding_model = None
        self._initialize_databases()
        self._load_embedding_model()
    
    def _initialize_databases(self):
        """Initialize SQLite and ChromaDB"""
        # SQLite for structured data
        # Allow cross-thread access for Streamlit compatibility
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        
        # ChromaDB for vector storage
        Path(CHROMA_DB_PATH).mkdir(exist_ok=True)
        
        # Initialize ChromaDB with error handling for uninitialized databases
        try:
            self.vector_db = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
        except (ValueError, Exception) as e:
            # If database is uninitialized or corrupted, try to reset it
            error_str = str(e).lower()
            if 'tenant' in error_str or 'no such table' in error_str or 'not found' in error_str:
                print(f"⚠️  ChromaDB database not initialized, resetting...")
                # Remove the directory and recreate it
                if CHROMA_DB_PATH.exists():
                    try:
                        shutil.rmtree(str(CHROMA_DB_PATH))
                    except:
                        pass
                Path(CHROMA_DB_PATH).mkdir(exist_ok=True)
                # Try again
                self.vector_db = chromadb.PersistentClient(
                    path=str(CHROMA_DB_PATH),
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                raise
        
        # Get or create collection
        try:
            self.comments_collection = self.vector_db.get_collection("feedback_comments")
        except:
            self.comments_collection = self.vector_db.create_collection(
                name="feedback_comments",
                metadata={"hnsw:space": "cosine"}
            )
        
        print("✅ Databases initialized")
    
    def _create_tables(self):
        """Create SQLite tables"""
        cursor = self.conn.cursor()
        
        # Main responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                event_name TEXT,
                event_category TEXT,
                form_date TEXT,
                occurrence TEXT,
                timestamp TEXT,
                structured_data TEXT,
                comments TEXT,
                hotel_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Ratings summary table (for quick analytics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ratings_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT,
                event_category TEXT,
                rating_question TEXT,
                average_rating REAL,
                total_count INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Events metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_name TEXT PRIMARY KEY,
                event_category TEXT,
                form_date TEXT,
                occurrence TEXT,
                total_responses INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Migrate existing table to add hotel_name column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE responses ADD COLUMN hotel_name TEXT")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        self.conn.commit()
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        print(f"📦 Loading embedding model: {EMBEDDING_MODEL}...")
        import os
        import warnings
        import torch
        warnings.filterwarnings('ignore')
        
        # Set environment to force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Workaround for PyTorch 2.9+ meta tensor issue
        # Monkey-patch torch.nn.Module.to() to handle meta tensors gracefully
        # IMPORTANT: Keep this patch active permanently, don't restore it
        # because encoding operations may also trigger device moves
        if not hasattr(torch.nn.Module, '_nape_patched'):
            original_to = torch.nn.Module.to
            
            def patched_to(self, *args, **kwargs):
                try:
                    return original_to(self, *args, **kwargs)
                except (NotImplementedError, RuntimeError) as e:
                    error_str = str(e).lower()
                    if "meta tensor" in error_str or "meta" in error_str:
                        # If meta tensor error, skip the move and return self
                        # Meta tensors are placeholders and shouldn't be moved
                        return self
                    raise
            
            # Apply the patch permanently
            torch.nn.Module.to = patched_to
            torch.nn.Module._nape_patched = True  # Mark as patched
        
        try:
            # Load model with explicit device to avoid meta tensors
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                device='cpu'
            )
        except Exception as e:
            # If still fails, try with model_kwargs
            try:
                self.embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device='cpu',
                    model_kwargs={'torch_dtype': torch.float32, 'low_cpu_mem_usage': False}
                )
            except Exception as e2:
                # Last resort: basic load
                print(f"⚠️  Warning during model load: {str(e)[:100]}")
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        
        # CRITICAL: Materialize all meta tensors immediately after loading
        # This must happen before any encoding operations
        with torch.no_grad():
            def materialize_tensors(module):
                """Recursively materialize all meta tensors in the module"""
                for name, child in module.named_children():
                    materialize_tensors(child)
                
                # Materialize parameters
                for param in module.parameters(recurse=False):
                    if hasattr(param, 'is_meta') and param.is_meta:
                        # Create a properly initialized tensor (not zeros, but actual weights)
                        # We need to load the actual weights, not create zeros
                        try:
                            # Get the actual data from the state dict if available
                            if hasattr(module, 'state_dict'):
                                state_dict = module.state_dict()
                                param_name = None
                                for n, p in module.named_parameters():
                                    if p is param:
                                        param_name = n
                                        break
                                if param_name and param_name in state_dict:
                                    param.data = state_dict[param_name].to('cpu')
                                else:
                                    # Fallback: use normal initialization
                                    param.data = torch.nn.init.normal_(torch.empty_like(param, device='cpu'))
                            else:
                                # Fallback: use normal initialization
                                param.data = torch.nn.init.normal_(torch.empty_like(param, device='cpu'))
                        except Exception as e:
                            # Last resort: zeros
                            param.data = torch.zeros_like(param, device='cpu')
                
                # Materialize buffers
                for buffer in module.buffers(recurse=False):
                    if hasattr(buffer, 'is_meta') and buffer.is_meta:
                        try:
                            buffer.data = torch.zeros_like(buffer, device='cpu')
                        except:
                            pass
            
            try:
                materialize_tensors(self.embedding_model)
            except Exception as e:
                print(f"⚠️  Warning: Could not fully materialize tensors: {e}")
        
        # Ensure model is on CPU and in eval mode
        try:
            # Force all parameters to CPU explicitly
            for param in self.embedding_model.parameters():
                if hasattr(param, 'data') and hasattr(param.data, 'to'):
                    try:
                        param.data = param.data.to('cpu')
                    except:
                        pass
            for buffer in self.embedding_model.buffers():
                if hasattr(buffer, 'data') and hasattr(buffer.data, 'to'):
                    try:
                        buffer.data = buffer.data.to('cpu')
                    except:
                        pass
            
            self.embedding_model = self.embedding_model.to('cpu')
            self.embedding_model.eval()
        except Exception:
            # If to() fails due to meta tensors, that's okay - the patch will handle it
            pass
        
        # Also patch Tensor.item() to handle meta tensors
        if not hasattr(torch.Tensor, '_nape_item_patched'):
            original_item = torch.Tensor.item
            
            def patched_item(self):
                try:
                    # Check if this is a meta tensor before calling original
                    if hasattr(self, 'is_meta') and self.is_meta:
                        # Try to materialize it first
                        if hasattr(self, 'device'):
                            # Create a zero tensor with same shape/dtype on CPU
                            materialized = torch.zeros_like(self, device='cpu')
                            return materialized.item()
                        return 0.0  # Fallback
                    return original_item(self)
                except (NotImplementedError, RuntimeError) as e:
                    error_str = str(e).lower()
                    if "meta tensor" in error_str or "meta" in error_str or "item()" in error_str:
                        # Try to materialize and return 0 as fallback
                        try:
                            if hasattr(self, 'device'):
                                materialized = torch.zeros_like(self, device='cpu')
                                return materialized.item()
                        except:
                            pass
                        return 0.0  # Last resort fallback
                    raise
            
            torch.Tensor.item = patched_item
            torch.Tensor._nape_item_patched = True
        
        # Explicitly ensure model is on CPU and in eval mode
        try:
            import torch
            # Force model to CPU explicitly - handle meta tensors
            def move_to_cpu(module):
                """Recursively move module to CPU, skipping meta tensors"""
                for name, child in module.named_children():
                    move_to_cpu(child)
                for param in module.parameters(recurse=False):
                    # Skip meta tensors - they're placeholders without actual data
                    if hasattr(param, 'device') and param.device.type != 'meta':
                        try:
                            # Only move if not already on CPU
                            if param.device.type != 'cpu':
                                param.data = param.data.to('cpu')
                        except (NotImplementedError, RuntimeError) as e:
                            # If it's a meta tensor error, just skip it
                            if "meta" not in str(e).lower():
                                pass  # Other errors we can ignore
                for buffer in module.buffers(recurse=False):
                    if hasattr(buffer, 'device') and buffer.device.type != 'meta':
                        try:
                            if buffer.device.type != 'cpu':
                                buffer.data = buffer.data.to('cpu')
                        except (NotImplementedError, RuntimeError):
                            pass
            
            move_to_cpu(self.embedding_model)
            
            # Additional safety: ensure the model's device property is CPU
            try:
                # This will use our patched .to() method which handles meta tensors
                self.embedding_model = self.embedding_model.to('cpu')
            except:
                pass  # If it fails, the model is probably already on CPU
            
            # Set to eval mode to avoid training-related issues
            if hasattr(self.embedding_model, 'eval'):
                self.embedding_model.eval()
        except Exception as e:
            print(f"⚠️  Warning: Could not explicitly set model to CPU: {e}")
        
        print("✅ Embedding model loaded")
    
    def store_responses(self, records: List[Dict], all_events: List[Dict] = None):
        """Store response records in both databases
        
        Args:
            records: List of response records to store
            all_events: Optional list of all events from master sheet (to track events with no responses)
        """
        cursor = self.conn.cursor()
        
        # Clear existing data before storing new data to avoid duplicates
        print("🧹 Clearing existing data...")
        cursor.execute("DELETE FROM responses")
        cursor.execute("DELETE FROM ratings_summary")
        cursor.execute("DELETE FROM events")  # Clear events table too
        # Clear ChromaDB collection
        try:
            self.vector_db.delete_collection("feedback_comments")
            self.comments_collection = self.vector_db.create_collection(
                name="feedback_comments",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            # Collection might not exist, create it
            self.comments_collection = self.vector_db.create_collection(
                name="feedback_comments",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Store all events in events table (even those with no responses)
        if all_events:
            print(f"📋 Storing {len(all_events)} events in events table...")
            for event in all_events:
                event_name = event.get('event_name', '')
                if event_name:
                    # Count responses for this event
                    response_count = sum(1 for r in records if r.get('event_name') == event_name)
                    cursor.execute("""
                        INSERT OR REPLACE INTO events 
                        (event_name, event_category, form_date, occurrence, total_responses, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        event_name,
                        event.get('event_category', 'Professional'),
                        event.get('form_date', ''),
                        event.get('occurrence', ''),
                        response_count,
                        datetime.now().isoformat()
                    ))
        
        self.conn.commit()
        
        # Prepare data for ChromaDB
        comment_texts = []
        comment_metadatas = []
        comment_ids = []
        
        for record in records:
            # Store in SQLite
            cursor.execute("""
                INSERT OR REPLACE INTO responses 
                (response_id, event_name, event_category, form_date, occurrence, 
                 timestamp, structured_data, comments, hotel_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record['response_id'],
                record['event_name'],
                record['event_category'],
                record['form_date'],
                record['occurrence'],
                record['timestamp'],
                json.dumps(record['structured_data']),
                record['comments'],
                record.get('hotel_name')  # Can be None
            ))
            
            # Prepare for ChromaDB (only if there are comments)
            if record.get('comments'):
                comment_texts.append(record['comments'])
                metadata = {
                    'response_id': record['response_id'],
                    'event_name': record['event_name'],
                    'event_category': record['event_category'],
                    'form_date': record.get('form_date', ''),
                    'timestamp': record['timestamp']
                }
                # Add hotel name if available (for accommodation events)
                if record.get('hotel_name'):
                    metadata['hotel_name'] = record['hotel_name']
                
                comment_metadatas.append(metadata)
                comment_ids.append(record['response_id'])
        
        self.conn.commit()
        
        # Store comments in ChromaDB with embeddings
        if comment_texts:
            try:
                # Generate embeddings (model should already be on CPU)
                embeddings = self.embedding_model.encode(
                    comment_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).tolist()
            except Exception as e:
                # If meta tensor error, try to ensure model is on CPU first
                if "meta" in str(e).lower() or "device" in str(e).lower():
                    try:
                        import torch
                        # Force all model parameters to CPU
                        for param in self.embedding_model.parameters():
                            if param.device.type == 'meta':
                                continue  # Skip meta tensors
                        # Retry encoding
                        embeddings = self.embedding_model.encode(comment_texts).tolist()
                    except Exception as e2:
                        print(f"❌ Error generating embeddings: {e2}")
                        raise
                else:
                    raise
            self.comments_collection.add(
                ids=comment_ids,
                embeddings=embeddings,
                documents=comment_texts,
                metadatas=comment_metadatas
            )
        
        print(f"✅ Stored {len(records)} responses")
    
    def update_ratings_summary(self, processed_data: Dict):
        """Update ratings summary table for quick analytics"""
        cursor = self.conn.cursor()
        
        # Clear existing ratings summary to avoid duplicates
        cursor.execute("DELETE FROM ratings_summary")
        
        # Ratings structure: {column_name: {values, average, count}}
        ratings_dict = processed_data.get('ratings', {})
        
        # Get all unique events and their categories
        cursor.execute("""
            SELECT DISTINCT event_name, event_category FROM responses
        """)
        events = {row['event_name']: row['event_category'] for row in cursor.fetchall()}
        
        # For each rating question, calculate per-event averages
        for rating_question, rating_data in ratings_dict.items():
            if not isinstance(rating_data, dict) or 'average' not in rating_data:
                continue
            
            # Try to get per-event ratings from the database
            for event_name, event_category in events.items():
                cursor.execute("""
                    SELECT structured_data FROM responses 
                    WHERE event_name = ?
                """, (event_name,))
                rows = cursor.fetchall()
                
                # Extract ratings for this event and question
                event_ratings = []
                for row in rows:
                    try:
                        import json
                        structured = json.loads(row['structured_data'])
                        if rating_question in structured:
                            val = structured[rating_question]
                            if isinstance(val, (int, float)):
                                event_ratings.append(float(val))
                            elif isinstance(val, str):
                                import re
                                num_match = re.search(r'(\d+)', val)
                                if num_match:
                                    rating_val = float(num_match.group(1))
                                    # Validate it's in reasonable range (1-5)
                                    if 1 <= rating_val <= 5:
                                        event_ratings.append(rating_val)
                    except:
                        continue
                
                # Only insert if we have ratings for this event/question combination
                if event_ratings:
                    avg_rating = sum(event_ratings) / len(event_ratings)
                    # Count unique responses that answered this question
                    # Since we're iterating through all responses, count unique response_ids
                    unique_response_ids = set()
                    cursor.execute("""
                        SELECT response_id, structured_data FROM responses 
                        WHERE event_name = ?
                    """, (event_name,))
                    all_rows = cursor.fetchall()
                    for row in all_rows:
                        try:
                            structured = json.loads(row['structured_data'])
                            if rating_question in structured:
                                val = structured[rating_question]
                                # Check if value exists and is valid
                                if val is not None and str(val).strip():
                                    unique_response_ids.add(row['response_id'])
                        except:
                            continue
                    
                    # Use the count of unique responses, fallback to len(event_ratings) if needed
                    unique_count = len(unique_response_ids) if unique_response_ids else len(event_ratings)
                    
                    cursor.execute("""
                        INSERT INTO ratings_summary
                        (event_name, event_category, rating_question, average_rating, total_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        event_name,
                        event_category,
                        rating_question,
                        avg_rating,
                        unique_count  # Count of unique responses that answered this question
                    ))
        
        self.conn.commit()
    
    def search_comments(self, query: str, top_k: int = 5, 
                       event_filter: Optional[str] = None,
                       category_filter: Optional[str] = None,
                       hotel_filter: Optional[str] = None) -> List[Dict]:
        """Semantic search in comments"""
        # Generate query embedding
        try:
            with torch.no_grad():
                query_embedding = self.embedding_model.encode(
                    [query],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device='cpu'
                ).tolist()[0]
        except Exception as e:
            # If meta tensor error, return empty results gracefully
            # This allows quantitative queries to still work
            error_str = str(e).lower()
            if "meta" in error_str or "device" in error_str or "item()" in error_str:
                print(f"⚠️  Meta tensor error during encoding. This may be due to PyTorch 2.9+ compatibility issues.")
                print(f"   Please ensure PyTorch <2.4.0 is installed (check requirements.txt)")
                print(f"   Returning empty results for this qualitative query.")
                # Return empty list instead of crashing
                return []
            else:
                # For other errors, still raise
                print(f"❌ Error generating query embedding: {e}")
                raise
        
        # Build where clause for metadata filtering
        where = {}
        if event_filter:
            where['event_name'] = event_filter
        if category_filter:
            where['event_category'] = category_filter
        if hotel_filter:
            where['hotel_name'] = hotel_filter
        
        # Search in ChromaDB
        results = self.comments_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where if where else None
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'response_id': results['ids'][0][i],
                    'comment': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def get_structured_stats(self, event_name: Optional[str] = None,
                            category: Optional[str] = None) -> Dict:
        """Get structured statistics from SQLite"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM ratings_summary WHERE 1=1"
        params = []
        
        if event_name:
            query += " AND event_name = ?"
            params.append(event_name)
        
        if category:
            query += " AND event_category = ?"
            params.append(category)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        stats = {
            'total_events': len(set(row['event_name'] for row in rows)),
            'ratings': []
        }
        
        for row in rows:
            stats['ratings'].append({
                'event_name': row['event_name'],
                'event_category': row['event_category'],
                'rating_question': row['rating_question'],
                'average_rating': row['average_rating'],
                'total_count': row['total_count']
            })
        
        return stats
    
    def get_event_list(self) -> List[Dict]:
        """Get list of all events with their response counts"""
        cursor = self.conn.cursor()
        
        # First, get actual response counts from responses table
        cursor.execute("""
            SELECT event_name, event_category, 
                   COUNT(*) as response_count
            FROM responses
            GROUP BY event_name, event_category
        """)
        events_with_responses = {row['event_name']: dict(row) for row in cursor.fetchall()}
        
        # Get all events from events table (includes events with no responses)
        cursor.execute("""
            SELECT event_name, event_category, total_responses
            FROM events
            ORDER BY event_name
        """)
        all_events_from_table = cursor.fetchall()
        
        # Merge: use actual count from responses if available, otherwise use stored count
        result = []
        for row in all_events_from_table:
            event_name = row['event_name']
            if event_name in events_with_responses:
                # Use actual count from responses table
                result.append(events_with_responses[event_name])
            else:
                # Use stored count (might be 0 for events with no responses)
                # Convert sqlite3.Row to dict for easier access
                row_dict = dict(row)
                result.append({
                    'event_name': event_name,
                    'event_category': row_dict.get('event_category', 'Professional'),
                    'response_count': row_dict.get('total_responses', 0)
                })
        
        # If events table is empty, fall back to events with responses only
        if not result:
            return list(events_with_responses.values())
        
        return result
    
    def get_hotel_feedback(self, hotel_name: str) -> Dict:
        """Get all feedback for a specific hotel"""
        cursor = self.conn.cursor()
        
        # Get structured data (ratings, etc.) - use case-insensitive matching
        cursor.execute("""
            SELECT structured_data, comments, timestamp
            FROM responses
            WHERE UPPER(hotel_name) = UPPER(?) AND event_name = 'ACCOMMODATION'
        """, (hotel_name,))
        
        rows = cursor.fetchall()
        feedback_items = []
        ratings = []
        all_ratings_by_question = {}  # Track ratings by question type
        seen_comments = set()  # Track unique comments to avoid duplicates
        
        for row in rows:
            structured = json.loads(row['structured_data'])
            if row['comments']:
                # Only add unique comments (avoid duplicates from multiple pipeline runs)
                comment_text = row['comments'].strip()
                if comment_text and comment_text not in seen_comments:
                    seen_comments.add(comment_text)
                    feedback_items.append({
                        'comment': comment_text,
                        'timestamp': row['timestamp']
                    })
            
            # Extract ratings by question - be more inclusive
            for key, val in structured.items():
                # Skip hotel name column and description columns
                if 'hotel' in key.lower() and ('lodged' in key.lower() or 'where' in key.lower()):
                    continue
                if 'describe yourself' in key.lower():
                    continue
                
                # Check if it's a rating question
                is_rating = False
                if any(kw in key.lower() for kw in ['rate', 'rating', 'satisfied', 'satisfaction']):
                    is_rating = True
                
                if is_rating:
                    rating_val = None
                    if isinstance(val, (int, float)):
                        rating_val = float(val)
                    elif isinstance(val, str):
                        import re
                        # Try to extract number (handle various formats)
                        num_match = re.search(r'(\d+)', str(val))
                        if num_match:
                            rating_val = float(num_match.group(1))
                            # Validate it's in reasonable range (1-5)
                            if rating_val < 1 or rating_val > 5:
                                rating_val = None
                    
                    if rating_val is not None:
                        ratings.append(rating_val)
                        # Track by question
                        if key not in all_ratings_by_question:
                            all_ratings_by_question[key] = []
                        all_ratings_by_question[key].append(rating_val)
        
        # Calculate average per question
        avg_by_question = {}
        for question, question_ratings in all_ratings_by_question.items():
            if question_ratings:
                avg_by_question[question] = {
                    'average': sum(question_ratings) / len(question_ratings),
                    'count': len(question_ratings)
                }
        
        return {
            'hotel_name': hotel_name,
            'total_responses': len(rows),
            'comments': feedback_items,
            'average_rating': sum(ratings) / len(ratings) if ratings else None,
            'rating_count': len(ratings),
            'ratings_by_question': avg_by_question  # Add breakdown by question
        }
    
    def get_all_hotels(self) -> List[str]:
        """Get list of all hotels that received responses in accommodation feedback"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT hotel_name
            FROM responses
            WHERE hotel_name IS NOT NULL AND event_name = 'ACCOMMODATION'
            ORDER BY hotel_name
        """)
        
        hotels_with_responses = [row['hotel_name'] for row in cursor.fetchall()]
        return hotels_with_responses
    
    def get_all_available_hotels(self) -> List[str]:
        """Get list of all 17 hotels that were available/used for accommodation"""
        from config import ALL_ACCOMMODATION_HOTELS
        return ALL_ACCOMMODATION_HOTELS
    
    def get_all_available_hotels(self) -> List[str]:
        """Get the complete list of all 17 hotels that were available/used"""
        # These are the 17 hotels that were available in the accommodation form
        all_hotels = [
            'Beni Apartments',
            'BWC Hotel',
            'Chateux De Atlantique',
            'Eko Hotel & Suites',
            'Exclusive Mansion',
            'H210',
            'Hotellnn',
            'Hotels No. 35.',
            'Morning Side',
            'Newcastle Hotel',
            'Posh Hotel',
            'Peninsula Hotel',
            'Presken Hotel',
            'S&S Hotel & Suits',
            'Timeoak Hotel',
            'VirginRose Hotel',
            'Villa Toscana'
        ]
        return all_hotels


if __name__ == "__main__":
    # Test storage
    storage = StorageManager()
    print("✅ Storage manager initialized successfully")


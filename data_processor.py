"""Process and structure feedback data for storage"""
import pandas as pd
import re
from typing import Dict, List, Optional
from datetime import datetime


class DataProcessor:
    """Processes raw form data into structured and unstructured components"""
    
    @staticmethod
    def extract_ratings(df: pd.DataFrame) -> Dict:
        """Extract all star ratings from dataframe columns"""
        ratings = {}
        
        # Find columns that look like ratings (contain "rate", "rating", or scale indicators)
        rating_patterns = [
            r'rate',
            r'rating',
            r'\(1.*5\)',
            r'scale',
            r'how.*rate'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            if any(re.search(pattern, col_lower) for pattern in rating_patterns):
                # Extract numeric ratings (handle various formats)
                rating_values = []
                for val in df[col].dropna():
                    if pd.isna(val):
                        continue
                    # Try to extract number
                    if isinstance(val, (int, float)):
                        rating_values.append(float(val))
                    elif isinstance(val, str):
                        # Extract first number found
                        num_match = re.search(r'(\d+)', val)
                        if num_match:
                            rating_values.append(float(num_match.group(1)))
                
                if rating_values:
                    ratings[col] = {
                        'values': rating_values,
                        'average': sum(rating_values) / len(rating_values),
                        'count': len(rating_values)
                    }
        
        return ratings
    
    @staticmethod
    def extract_multiple_choice(df: pd.DataFrame) -> Dict:
        """Extract multiple choice responses"""
        mc_data = {}
        
        for col in df.columns:
            # Skip timestamp, event_name, etc.
            if col.lower() in ['timestamp', 'event_name', 'form_url', 'extracted_at', 
                             'event_category', 'form_date', 'occurrence']:
                continue
            
            # Check if column looks like multiple choice (has limited unique values)
            unique_vals = df[col].dropna().unique()
            
            # If it's not a rating column and has reasonable number of unique values
            if (len(unique_vals) <= 20 and 
                not any(keyword in col.lower() for keyword in ['rate', 'rating', 'comment', 'suggestion'])):
                
                value_counts = df[col].value_counts().to_dict()
                mc_data[col] = value_counts
        
        return mc_data
    
    @staticmethod
    def extract_comments(df: pd.DataFrame) -> List[Dict]:
        """Extract open-ended comments/suggestions"""
        comments = []
        
        # Find comment columns
        comment_keywords = ['comment', 'suggestion', 'feedback', 'other', 'additional']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in comment_keywords):
                for idx, row in df.iterrows():
                    comment_text = row[col]
                    if pd.notna(comment_text) and str(comment_text).strip():
                        comments.append({
                            'response_id': f"{row.get('event_name', 'unknown')}_{idx}",
                            'event_name': row.get('event_name', 'Unknown'),
                            'event_category': row.get('event_category', 'Other'),
                            'comment_text': str(comment_text).strip(),
                            'timestamp': row.get('Timestamp', row.get('extracted_at', datetime.now().isoformat())),
                            'source_column': col
                        })
        
        return comments
    
    @staticmethod
    def process_dataframe(df: pd.DataFrame) -> Dict:
        """Process entire dataframe and return structured data"""
        processed = {
            'ratings': DataProcessor.extract_ratings(df),
            'multiple_choice': DataProcessor.extract_multiple_choice(df),
            'comments': DataProcessor.extract_comments(df),
            'metadata': {
                'total_responses': len(df),
                'events': df['event_name'].unique().tolist() if 'event_name' in df.columns else [],
                'event_categories': df['event_category'].unique().tolist() if 'event_category' in df.columns else []
            }
        }
        
        return processed
    
    @staticmethod
    def create_response_records(df: pd.DataFrame) -> List[Dict]:
        """Create individual response records for database storage"""
        records = []
        
        for idx, row in df.iterrows():
            # Extract structured data
            structured = {}
            comments_text = []
            
            for col in df.columns:
                if col.lower() in ['timestamp', 'event_name', 'form_url', 'extracted_at', 
                                 'event_category', 'form_date', 'occurrence']:
                    continue
                
                val = row[col]
                if pd.isna(val):
                    continue
                
                # Check if it's a comment field
                if any(keyword in col.lower() for keyword in ['comment', 'suggestion', 'feedback']):
                    if str(val).strip():
                        comments_text.append(str(val).strip())
                else:
                    # Store as structured data
                    structured[col] = str(val) if not isinstance(val, (int, float)) else val
            
            # Extract hotel name for accommodation events
            hotel_name = None
            if row.get('event_name', '').upper() == 'ACCOMMODATION':
                # Look for hotel-related questions - be more flexible
                hotel_keywords = ['hotel', 'lodged', 'lodging', 'accommodation']
                
                # Known hotel names from the form (for validation)
                known_hotels = [
                    'Beni Apartments', 'BWC Hotel', 'Chateux De Atlantique',
                    'Eko Hotel & Suites', 'Exclusive Mansion', 'H210',
                    'Hotellnn', 'Hotels No. 35.', 'Morning Side',
                    'Newcastle Hotel', 'Posh Hotel', 'Peninsula Hotel',
                    'Presken Hotel', 'S&S Hotel & Suits', 'Timeoak Hotel',
                    'VirginRose Hotel', 'Villa Toscana'
                ]
                
                # Try to find hotel column
                hotel_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    # Check if column is about hotels/lodging
                    if any(keyword in col_lower for keyword in hotel_keywords):
                        # Skip if it's a rating question
                        if 'rate' not in col_lower and 'rating' not in col_lower:
                            hotel_col = col
                            break
                
                # Extract hotel name from the identified column
                if hotel_col and hotel_col in structured:
                    val = structured[hotel_col]
                    if pd.notna(val) and str(val).strip():
                        raw_hotel_name = str(val).strip()
                        
                        # Normalize hotel name - try exact match first, then partial
                        hotel_name = raw_hotel_name
                        best_match = None
                        best_match_score = 0
                        
                        for known_hotel in known_hotels:
                            raw_lower = raw_hotel_name.lower()
                            known_lower = known_hotel.lower()
                            
                            # Exact match (highest priority)
                            if raw_lower == known_lower:
                                hotel_name = known_hotel
                                break
                            # Partial match - calculate similarity
                            elif raw_lower in known_lower or known_lower in raw_lower:
                                # Prefer longer/more complete name
                                match_score = min(len(raw_lower), len(known_lower)) / max(len(raw_lower), len(known_lower))
                                if match_score > best_match_score:
                                    best_match_score = match_score
                                    best_match = known_hotel
                        
                        # Use best partial match if no exact match found
                        if hotel_name == raw_hotel_name and best_match:
                            hotel_name = best_match
            
            # Create deterministic response_id based on event, timestamp from form, and row index
            # Use form timestamp if available, otherwise use a hash of the row data
            form_timestamp = row.get('Timestamp', row.get('extracted_at', ''))
            if pd.notna(form_timestamp) and form_timestamp:
                timestamp_hash = hash(str(form_timestamp) + str(idx))
            else:
                timestamp_hash = hash(str(row.to_dict()))
            
            record = {
                'response_id': f"{row.get('event_name', 'unknown')}_{idx}_{abs(timestamp_hash)}",
                'event_name': row.get('event_name', 'Unknown'),
                'event_category': row.get('event_category', 'Other'),
                'form_date': row.get('form_date', ''),
                'occurrence': row.get('occurrence', ''),
                'timestamp': row.get('Timestamp', row.get('extracted_at', datetime.now().isoformat())),
                'structured_data': structured,
                'comments': ' | '.join(comments_text) if comments_text else None,
                'hotel_name': hotel_name  # Add hotel name for accommodation events
            }
            
            records.append(record)
        
        return records


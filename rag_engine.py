"""RAG (Retrieval-Augmented Generation) query engine"""
from typing import List, Dict, Optional
import ollama
import hashlib
import json
from config import USE_OLLAMA, OLLAMA_MODEL, USE_GROQ, GROQ_API_KEY, GROQ_MODEL, TOP_K_RESULTS, MAX_CONTEXT_LENGTH, ENABLE_CACHING
from storage import StorageManager


class RAGEngine:
    """Handles natural language queries and generates insights"""
    
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.llm_client = self._initialize_llm()
        self.query_cache = {} if ENABLE_CACHING else None  # Simple in-memory cache
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        if USE_OLLAMA:
            try:
                # Test connection
                ollama.list()
                print(f"✅ Connected to Ollama (model: {OLLAMA_MODEL})")
                return 'ollama'
            except Exception as e:
                print(f"⚠️  Ollama not available: {e}")
                print("   Install Ollama from https://ollama.ai and run: ollama pull llama3.1")
                return None
        
        elif USE_GROQ and GROQ_API_KEY:
            try:
                import groq
                client = groq.Groq(api_key=GROQ_API_KEY)
                print(f"✅ Connected to Groq API (model: {GROQ_MODEL})")
                return client
            except Exception as e:
                print(f"⚠️  Groq API error: {e}")
                return None
        
        return None
    
    def _detect_query_type(self, query: str) -> Dict:
        """Detect what type of query this is"""
        query_lower = query.lower()
        
        # Quantitative queries (ratings, averages, counts)
        quantitative_keywords = [
            'average', 'avg', 'mean', 'rating', 'score', 'how many',
            'count', 'total', 'percentage', 'percent', 'statistics', 'stats'
        ]
        
        # Qualitative queries (comments, feedback, opinions)
        qualitative_keywords = [
            'what did', 'what do', 'what are', 'comments', 'feedback',
            'suggestions', 'opinions', 'thoughts', 'say about', 'think about',
            'complain', 'praise', 'like', 'dislike', 'issues', 'problems'
        ]
        
        # Event-specific queries
        event_keywords = [
            'wige', 'accommodation', 'transportation', 'feeding', 'technical',
            'poster', 'hackathon', 'competition', 'awards', 'social'
        ]
        
        is_quantitative = any(kw in query_lower for kw in quantitative_keywords)
        is_qualitative = any(kw in query_lower for kw in qualitative_keywords)
        
        # Extract event mentions - also check for hotel mentions
        mentioned_events = []
        events = self.storage.get_event_list()
        for event in events:
            event_name_lower = event['event_name'].lower()
            if event_name_lower in query_lower:
                mentioned_events.append(event['event_name'])
        
        # If hotel is mentioned, likely about accommodation
        if 'hotel' in query_lower and 'ACCOMMODATION' not in mentioned_events:
            mentioned_events.append('ACCOMMODATION')
        
        return {
            'is_quantitative': is_quantitative,
            'is_qualitative': is_qualitative,
            'is_hybrid': is_quantitative and is_qualitative,
            'mentioned_events': mentioned_events,
            'query_type': 'quantitative' if is_quantitative and not is_qualitative 
                         else 'qualitative' if is_qualitative and not is_quantitative
                         else 'hybrid'
        }
    
    def _retrieve_relevant_feedback(self, query: str, query_info: Dict) -> List[Dict]:
        """Retrieve relevant feedback based on query"""
        results = []
        
        # Semantic search for comments
        if query_info['is_qualitative'] or query_info['is_hybrid']:
            event_filter = query_info['mentioned_events'][0] if query_info['mentioned_events'] else None
            comments = self.storage.search_comments(
                query, 
                top_k=TOP_K_RESULTS,
                event_filter=event_filter
            )
            results.extend(comments)
        
        return results
    
    def _get_structured_data(self, query_info: Dict, user_query: str) -> Dict:
        """Get structured statistics if needed"""
        event_filter = query_info['mentioned_events'][0] if query_info['mentioned_events'] else None
        
        # Try to infer category from query
        category_filter = None
        query_lower = user_query.lower()
        if 'wige' in query_lower:
            category_filter = 'WiGE'
        elif any(kw in query_lower for kw in ['accommodation', 'feeding', 'transportation']):
            category_filter = 'Logistics'
        elif 'technical' in query_lower:
            category_filter = 'Technical'
        
        stats = self.storage.get_structured_stats(
            event_name=event_filter,
            category=category_filter
        )
        
        # If query mentions hotels, add hotel-specific data
        if 'hotel' in query_lower:
            # Auto-detect accommodation if hotel is mentioned
            if not event_filter:
                event_filter = 'ACCOMMODATION'
            
            if event_filter == 'ACCOMMODATION' or 'accommodation' in query_lower or 'hotel' in query_lower:
                # Get hotels with responses
                hotels_with_responses = self.storage.get_all_hotels()
                # Get all 17 hotels that were available/used
                all_hotels = self.storage.get_all_available_hotels()
                
                stats['hotels'] = hotels_with_responses
                stats['all_hotels'] = all_hotels
                stats['total_hotels'] = len(all_hotels)  # Always 17
                stats['hotels_with_responses'] = len(hotels_with_responses)  # Hotels that got feedback
                
                # Check if query asks about a specific hotel (check all 17 hotels, not just those with responses)
                mentioned_hotel = None
                for hotel in all_hotels:
                    # Check if hotel name appears in query (case-insensitive, partial match)
                    hotel_words = [w for w in hotel.lower().split() if len(w) > 3]
                    if any(word in query_lower for word in hotel_words):
                        mentioned_hotel = hotel
                        break
                    # Also check full hotel name
                    if hotel.lower() in query_lower:
                        mentioned_hotel = hotel
                        break
                
                # Get hotel feedback - get feedback for hotels that have responses
                hotel_feedback = {}
                for hotel in hotels_with_responses:
                    feedback = self.storage.get_hotel_feedback(hotel)
                    hotel_feedback[hotel] = feedback
                
                stats['hotel_feedback'] = hotel_feedback if hotel_feedback else {}  # Empty dict if no feedback yet
                
                # Mark if specific hotel was mentioned
                if mentioned_hotel:
                    stats['mentioned_hotel'] = mentioned_hotel
        
        return stats
    
    def _generate_response_ollama(self, query: str, context: Dict) -> str:
        """Generate response using Ollama"""
        # Build context prompt
        prompt = self._build_prompt(query, context)
        
        try:
            # Generation settings
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.7,  # Balanced temperature for robust responses
                    'num_predict': 500,  # Sufficient tokens for comprehensive answers
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_response_groq(self, query: str, context: Dict) -> str:
        """Generate response using Groq API"""
        prompt = self._build_prompt(query, context)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing conference feedback data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _build_prompt(self, query: str, context: Dict) -> str:
        """Build prompt for LLM"""
        prompt_parts = [
            "You are analyzing feedback from the NAPE 43rd Annual International Conference & Exhibition.",
            "Based on the following data, answer the user's question accurately and concisely.",
            "IMPORTANT: Do NOT repeat the user's question in your answer. Just provide the answer directly.",
            "",
            f"USER QUESTION: {query}",
            ""
        ]
        
        # Add explicit instruction for hotel count questions
        query_lower = query.lower()
        if 'how many' in query_lower and 'hotel' in query_lower and ('used' in query_lower or 'total' in query_lower):
            prompt_parts.append("CRITICAL INSTRUCTION: When asked 'how many hotels were used in total' or similar questions about the total number of hotels, the answer is ALWAYS 17. All 17 hotels were used for accommodation during the conference, regardless of whether they received feedback responses.")
            prompt_parts.append("")
        
        # Add structured data if available
        if context.get('stats') and context['stats'].get('ratings'):
            prompt_parts.append("STRUCTURED DATA (Ratings & Statistics):")
            for rating in context['stats']['ratings'][:10]:  # Limit to avoid token overflow
                prompt_parts.append(
                    f"- {rating['event_name']}: {rating['rating_question']} = "
                    f"{rating['average_rating']:.2f}/5 (from {rating['total_count']} responses)"
                )
            prompt_parts.append("")
        
        # Add hotel-specific feedback if available or if query is about hotels
        if context.get('stats') and (context['stats'].get('total_hotels') or 'hotel' in query.lower()):
            prompt_parts.append("HOTEL-SPECIFIC DATA:")
            
            # Always include total hotels count
            total_hotels = context['stats'].get('total_hotels', 17)
            hotels_with_responses_count = context['stats'].get('hotels_with_responses', 0)
            
            prompt_parts.append(f"Total hotels used: {total_hotels} (all {total_hotels} hotels were used for accommodation)")
            if hotels_with_responses_count > 0:
                prompt_parts.append(f"Hotels that received feedback responses: {hotels_with_responses_count} out of {total_hotels}")
            
            # List all hotels if available
            if context['stats'].get('all_hotels'):
                prompt_parts.append(f"All {total_hotels} hotels used: {', '.join(context['stats']['all_hotels'])}")
            
            prompt_parts.append("Note: All 17 hotels were used for accommodation. The hotels listed below (if any) are those that received feedback responses from attendees.")
            
            # If specific hotel was mentioned, prioritize it
            mentioned_hotel = context['stats'].get('mentioned_hotel')
            hotel_items = list(context['stats']['hotel_feedback'].items())
            if mentioned_hotel and mentioned_hotel in context['stats']['hotel_feedback']:
                # Put mentioned hotel first
                hotel_items = [(mentioned_hotel, context['stats']['hotel_feedback'][mentioned_hotel])] + \
                             [(h, f) for h, f in hotel_items if h != mentioned_hotel]
            
            for hotel, feedback in hotel_items:
                prompt_parts.append(f"\n{hotel}:")
                prompt_parts.append(f"  - Total responses: {feedback['total_responses']}")
                if feedback['average_rating']:
                    prompt_parts.append(f"  - Overall average rating: {feedback['average_rating']:.2f}/5 (from {feedback['rating_count']} ratings)")
                # Add breakdown by question if available
                if feedback.get('ratings_by_question'):
                    prompt_parts.append(f"  - Detailed ratings:")
                    for question, q_data in feedback['ratings_by_question'].items():
                        q_short = question[:70] + "..." if len(question) > 70 else question
                        prompt_parts.append(f"    * {q_short}: {q_data['average']:.2f}/5 ({q_data['count']} responses)")
                if feedback['comments']:
                    prompt_parts.append(f"  - Written comments ({len(feedback['comments'])}):")
                    for comment in feedback['comments']:  # Show all comments
                        prompt_parts.append(f"    * {comment['comment']}")
                elif feedback['total_responses'] > 0:
                    prompt_parts.append(f"  - No written comments provided")
            prompt_parts.append("")
        
        # Add qualitative feedback
        if context.get('feedback'):
            prompt_parts.append("QUALITATIVE FEEDBACK (Comments from attendees):")
            for i, fb in enumerate(context['feedback'][:10], 1):  # Limit to top 10
                event = fb['metadata'].get('event_name', 'Unknown')
                hotel = fb['metadata'].get('hotel_name', '')
                comment = fb['comment']
                # Include hotel name if available (for accommodation events)
                if hotel:
                    prompt_parts.append(f"{i}. [{event} - Hotel: {hotel}] {comment}")
                else:
                    prompt_parts.append(f"{i}. [{event}] {comment}")
            prompt_parts.append("")
        
        prompt_parts.append(
            "INSTRUCTIONS: "
            "- Provide a clear, concise answer based ONLY on the data provided above. "
            "- If the data doesn't contain enough information, say so. "
            "- For quantitative questions, cite specific numbers and calculations. "
            "- For qualitative questions, summarize key themes and patterns. "
            "- When asked 'How many hotels were used in total?' or similar questions about total hotel count, the answer is 17 (all hotels that were available/used). "
            "- When asked about hotels, check ALL hotels in the HOTEL-SPECIFIC DATA section. "
            "- For questions about complaints/problems, review all hotels' comments, not just one. "
            "- Be objective and professional. "
            "- If hotel-specific data is provided, use it to answer hotel-related questions accurately."
        )
        
        return "\n".join(prompt_parts)
    
    def query(self, user_query: str) -> Dict:
        """Main query method"""
        if not self.llm_client:
            return {
                'answer': "LLM not available. Please set up Ollama or Groq API.",
                'sources': [],
                'stats': None
            }
        
        # Check cache first
        if self.query_cache is not None:
            query_hash = hashlib.md5(user_query.lower().strip().encode()).hexdigest()
            if query_hash in self.query_cache:
                return self.query_cache[query_hash]
        
        # Analyze query
        query_info = self._detect_query_type(user_query)
        
        # Retrieve relevant data
        feedback = self._retrieve_relevant_feedback(user_query, query_info)
        stats = self._get_structured_data(query_info, user_query) if query_info['is_quantitative'] or query_info['is_hybrid'] else None
        
        # Build context
        context = {
            'feedback': feedback,
            'stats': stats,
            'query_info': query_info
        }
        
        # Generate response
        if self.llm_client == 'ollama':
            answer = self._generate_response_ollama(user_query, context)
        elif hasattr(self.llm_client, 'chat'):
            answer = self._generate_response_groq(user_query, context)
        else:
            answer = "LLM client not properly configured."
        
        result = {
            'answer': answer,
            'sources': feedback[:TOP_K_RESULTS],  # Top N sources (from config)
            'stats': stats,
            'query_type': query_info['query_type']
        }
        
        # Cache result
        if self.query_cache is not None:
            query_hash = hashlib.md5(user_query.lower().strip().encode()).hexdigest()
            self.query_cache[query_hash] = result
        
        return result


if __name__ == "__main__":
    from storage import StorageManager
    storage = StorageManager()
    engine = RAGEngine(storage)
    
    # Test query
    result = engine.query("What did attendees say about the WiGE event?")
    print(f"\nQuery Result:\n{result['answer']}\n")


"""Data ingestion module for Google Forms responses"""
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from typing import List, Dict, Optional
import re
import time
import json
import tempfile
from datetime import datetime
from config import CREDENTIALS_PATH, MASTER_SHEET_NAME, EVENT_CATEGORIES


def clean_json_string(json_str: str) -> str:
    """Clean JSON string to handle common formatting issues from TOML/Streamlit secrets"""
    # Remove triple quotes if present
    json_str = json_str.strip()
    if json_str.startswith('"""') or json_str.startswith("'''"):
        json_str = json_str[3:-3].strip()
    
    return json_str


def parse_credentials_json(json_str: str) -> dict:
    """Parse credentials JSON with better error handling for control characters"""
    cleaned = clean_json_string(json_str)
    
    # Try normal parsing first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # If it fails due to control characters, try to fix them
        error_msg = str(e).lower()
        if 'control character' in error_msg or 'invalid' in error_msg:
            # The issue is likely in the private_key field with unescaped newlines
            # TOML might have converted \n to actual newlines
            # We need to escape them back to \n for JSON
            
            # Strategy: Find the private_key field and fix newlines
            import re
            
            # Find the private_key field - it spans multiple lines
            # Pattern: "private_key": "-----BEGIN...\n...\n...-----END..."
            # We need to match from "private_key" to the closing quote, handling newlines
            
            # First, try to find where private_key starts and ends
            private_key_pattern = r'"private_key"\s*:\s*"'
            match_start = re.search(private_key_pattern, cleaned)
            
            if match_start:
                start_pos = match_start.end()
                # Find the closing quote - it should be after "-----END PRIVATE KEY-----"
                # Look for the pattern that ends the private key value
                end_pattern = r'-----END PRIVATE KEY-----\s*"'
                end_match = re.search(end_pattern, cleaned[start_pos:])
                
                if end_match:
                    end_pos = start_pos + end_match.end()
                    # Extract the private key value
                    key_value = cleaned[start_pos:end_pos-1]  # -1 to exclude the closing quote
                    
                    # Escape newlines and other control characters
                    escaped_value = key_value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    
                    # Reconstruct the JSON
                    fixed_json = (
                        cleaned[:start_pos] + 
                        escaped_value + 
                        '"' + 
                        cleaned[end_pos:]
                    )
                    
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        pass
            
            # Alternative: Try to escape all control characters in the entire string
            # This is more aggressive but might work
            fixed_json = cleaned
            # Replace actual newlines that are inside string values with \n
            # This is tricky - we need to be careful not to break the JSON structure
            
            # Simple approach: replace newlines that appear to be in the private_key
            if 'BEGIN PRIVATE KEY' in fixed_json and 'END PRIVATE KEY' in fixed_json:
                # Find the section between BEGIN and END
                begin_idx = fixed_json.find('BEGIN PRIVATE KEY')
                end_idx = fixed_json.find('END PRIVATE KEY') + len('END PRIVATE KEY')
                
                # Get the section
                private_section = fixed_json[begin_idx:end_idx]
                # Escape newlines in this section
                private_section_fixed = private_section.replace('\n', '\\n').replace('\r', '\\r')
                
                # Replace in the full string
                fixed_json = fixed_json[:begin_idx] + private_section_fixed + fixed_json[end_idx:]
                
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
            
            # Last resort: provide helpful error message
            raise ValueError(
                f"Could not parse credentials JSON due to control characters. "
                f"Original error: {e}\n\n"
                f"**Solution:** In Streamlit Cloud secrets, make sure your private_key uses \\n for newlines:\n"
                f'  "private_key": "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"\n\n'
                f"Or use triple quotes in TOML and keep the newlines as-is (the app will handle them)."
            )
        
        # Re-raise if it's a different error
        raise


class DataIngestion:
    """Handles data collection from Google Sheets"""
    
    def __init__(self):
        self.client = None
        self.master_sheet = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Sheets API"""
        try:
            scope = [
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly'
            ]
            
            # Check if credentials file exists
            if not CREDENTIALS_PATH.exists():
                # Try to get from Streamlit secrets if file doesn't exist
                try:
                    import streamlit as st
                    if hasattr(st, 'secrets'):
                        google_creds = st.secrets.get("GOOGLE_CREDENTIALS", None)
                        if google_creds:
                            # Handle both string and dict formats
                            if isinstance(google_creds, str):
                                try:
                                    creds_dict = parse_credentials_json(google_creds)
                                except Exception as e:
                                    raise ValueError(f"Could not parse GOOGLE_CREDENTIALS: {e}")
                            elif isinstance(google_creds, dict):
                                creds_dict = google_creds
                            else:
                                raise ValueError(f"Unexpected type for GOOGLE_CREDENTIALS: {type(google_creds)}")
                            
                            # Create temp file
                            temp_creds = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                            json.dump(creds_dict, temp_creds, indent=2)
                            temp_creds.close()
                            creds_path = temp_creds.name
                            print(f"✅ Using credentials from Streamlit secrets")
                        else:
                            raise FileNotFoundError(
                                f"credentials.json not found at {CREDENTIALS_PATH} and "
                                "GOOGLE_CREDENTIALS not found in Streamlit secrets. "
                                "Please add GOOGLE_CREDENTIALS to Streamlit Cloud secrets."
                            )
                    else:
                        raise FileNotFoundError(
                            f"credentials.json not found at {CREDENTIALS_PATH}. "
                            "Please follow SETUP_GUIDE.md to create it or add GOOGLE_CREDENTIALS to Streamlit secrets."
                        )
                except (ImportError, RuntimeError, AttributeError):
                    # Not in Streamlit context
                    raise FileNotFoundError(
                        f"credentials.json not found at {CREDENTIALS_PATH}. "
                        "Please follow SETUP_GUIDE.md to create it."
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                    raise FileNotFoundError(
                        f"GOOGLE_CREDENTIALS found in Streamlit secrets but could not be parsed: {e}. "
                        "Please check the format of your credentials JSON."
                    )
            else:
                creds_path = str(CREDENTIALS_PATH)
            
            creds = Credentials.from_service_account_file(creds_path, scopes=scope)
            self.client = gspread.authorize(creds)
            print("✅ Authenticated with Google Sheets API")
        except FileNotFoundError:
            raise  # Re-raise FileNotFoundError with our custom message
        except Exception as e:
            raise Exception(f"Authentication failed: {str(e)}")
    
    def _retry_with_backoff(self, func, max_retries=5, initial_delay=2, max_delay=60):
        """Retry a function with exponential backoff for rate limit errors"""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error (429)
                if '429' in error_str or 'quota' in error_str or 'rate limit' in error_str:
                    if attempt < max_retries - 1:
                        wait_time = min(delay, max_delay)
                        print(f"   ⏳ Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"   ❌ Rate limit error after {max_retries} attempts: {str(e)[:100]}")
                        raise
                else:
                    # Not a rate limit error, re-raise immediately
                    raise
        return None
    
    def get_master_sheet(self):
        """Get the master tracking sheet"""
        try:
            # Try to open by name first
            self.master_sheet = self.client.open(MASTER_SHEET_NAME)
            return self.master_sheet
        except gspread.SpreadsheetNotFound:
            # If not found, try to search
            print(f"Sheet '{MASTER_SHEET_NAME}' not found. Searching...")
            sheets = self.client.list_spreadsheet_files()
            for sheet in sheets:
                if MASTER_SHEET_NAME.lower() in sheet['name'].lower():
                    self.master_sheet = self.client.open_by_key(sheet['id'])
                    return self.master_sheet
            raise Exception(f"Could not find sheet: {MASTER_SHEET_NAME}")
    
    def get_events_list(self) -> List[Dict]:
        """Extract events list from master sheet"""
        if not self.master_sheet:
            self.get_master_sheet()
        
        worksheet = self.master_sheet.sheet1
        data = worksheet.get_all_records()
        
        events = []
        for row in data:
            # Handle variations in column names (spaces, typos, etc.)
            form_link = (row.get('FORM LINK') or row.get('FORM  LINK') or 
                        row.get('FORM_LINK') or '').strip()
            event_name = row.get('EVENT', '').strip()
            
            if form_link and event_name:
                events.append({
                    'event_name': event_name,
                    'form_link': form_link,
                    'response_sheet_link': (row.get('RESPONSE SHEET LINK') or 
                                          row.get('RESPONSE_SHEET_LINK') or '').strip(),
                    'form_date': row.get('FORM DATE', ''),
                    'occurrence': (row.get('OCCURRENCE') or row.get('OCCURENCE') or '').strip(),
                    'form_manager': row.get('FORM MANAGER', ''),
                    'report_manager': row.get('REPORT MANAGER', ''),
                    'status': row.get('NOTE ON FORM STATUS', '')
                })
        
        return events
    
    def extract_form_id_from_url(self, form_url: str) -> Optional[str]:
        """Extract Google Form ID from URL"""
        # Handle different URL formats
        patterns = [
            r'forms\.gle/([a-zA-Z0-9_-]+)',
            r'/d/([a-zA-Z0-9_-]+)',
            r'formId=([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, form_url)
            if match:
                return match.group(1)
        return None
    
    def get_form_responses(self, form_url: str, event_name: str, response_sheet_link: str = None) -> Optional[pd.DataFrame]:
        """Get responses from a Google Form via its linked Google Sheet"""
        try:
            # Method 1: Use direct response sheet link if provided (BEST METHOD)
            if response_sheet_link:
                try:
                    def _get_sheet_data():
                        sheet = self.client.open_by_url(response_sheet_link)
                        worksheet = sheet.sheet1
                        return worksheet.get_all_records()
                    
                    data = self._retry_with_backoff(_get_sheet_data)
                    
                    if data:
                        df = pd.DataFrame(data)
                        df['event_name'] = event_name
                        df['form_url'] = form_url
                        df['extracted_at'] = datetime.now().isoformat()
                        return df
                except Exception as e:
                    print(f"   ⚠️  Could not access response sheet link: {str(e)}")
            
            # Method 2: Try to find sheet by event name with smart matching
            try:
                # Get all sheets with retry
                def _list_sheets():
                    return self.client.list_spreadsheet_files()
                
                all_sheets = self._retry_with_backoff(_list_sheets)
                if not all_sheets:
                    raise Exception("Could not list spreadsheet files")
                
                # Normalize event name for matching
                event_lower = event_name.lower()
                # Handle special cases/abbreviations and variations
                event_variations = [event_lower]
                
                # Abbreviations (check for both full name and abbreviation)
                if 'field trip' in event_lower and 'virtual' in event_lower:
                    event_variations.extend(['vft', 'virtual field trip', 'field trip'])
                elif 'vft' in event_lower:
                    event_variations.extend(['vft', 'virtual field trip'])
                
                if 'basin evaluation' in event_lower and 'competition' in event_lower:
                    # Prioritize BEC abbreviation
                    event_variations = ['bec'] + event_variations + ['basin evaluation competition']
                elif 'bec' in event_lower:
                    event_variations.extend(['bec', 'basin evaluation'])
                
                # Typos and variations
                if 'registeration' in event_lower:
                    event_variations.append('registration')
                if 'accommodation' in event_lower:
                    # Prioritize common variations
                    event_variations = ['accommodation', 'accomodation'] + event_variations
                if 'feeding' in event_lower:
                    # Prioritize exact match
                    event_variations = ['feeding'] + event_variations
                if 'transportation' in event_lower:
                    # Prioritize exact match
                    event_variations = ['transportation', 'transport'] + event_variations
                if 'exhibition' in event_lower and 'exhibitions' not in event_lower:
                    event_variations.append('exhibitions')  # Sheet uses plural
                if 'poster session' in event_lower or 'poster presentation' in event_lower:
                    # Prioritize "poster presentation" as that's what the form uses
                    # The form name is "NAPE 43rd AICE POSTER PRESENTATION FEEDBACK FORM"
                    event_variations = ['poster presentation', 'poster session', 'poster'] + event_variations
                if 'management session' in event_lower and 'executive' not in event_lower:
                    # For regular "MANAGEMENT SESSION", prioritize plural form at the very front
                    # This ensures it matches "MANAGEMENT SESSIONS" (123 responses) over "Executive Management Session" (6 responses)
                    event_variations = ['management sessions'] + [v for v in event_variations if 'executive' not in v]
                elif 'executive management' in event_lower:
                    # For "EXECUTIVE MANAGEMENT SESSION", prioritize that
                    event_variations = ['executive management'] + event_variations
                if 'awards' in event_lower and 'recognition' in event_lower:
                    event_variations.append('awards & recognition')
                if 'all' in event_lower and 'convention' in event_lower and 'luncheon' in event_lower:
                    # Handle "ALL - CONVENTION LUNCHEON" (with space and hyphen) vs "ALL-CONVENTION LUNCHEON" (no space)
                    event_variations = ['all-convention luncheon', 'all convention luncheon', 'all - convention luncheon'] + event_variations
                if 'african nite' in event_lower or 'african night' in event_lower:
                    event_variations.extend(['african nite', 'african night'])
                if 'committee feedback' in event_lower:
                    event_variations.extend(['committee feedback', 'committee'])
                if 'alumni reunion' in event_lower:
                    # Prioritize exact match for ALUMNI REUNION
                    event_variations = ['alumni reunion'] + event_variations
                
                # Score and rank matches
                matches = []
                for sheet_info in all_sheets:
                    sheet_name = sheet_info['name'].lower()
                    score = 0
                    
                    # Skip the master sheet
                    if 'monitoring' in sheet_name and 'evaluation' in sheet_name:
                        continue
                    
                    # Exact match (highest priority)
                    for idx, variation in enumerate(event_variations):
                        if variation in sheet_name:
                            # Calculate match quality - earlier in list = higher priority
                            priority_bonus = (len(event_variations) - idx) * 5
                            
                            # Penalize matches with "executive" if event name doesn't have it
                            penalty = 0
                            if 'executive' in sheet_name and 'executive' not in event_lower:
                                penalty = 50  # Heavy penalty to avoid wrong match
                            
                            if variation == event_lower:
                                score = 100 + priority_bonus - penalty  # Perfect match
                            elif len(variation) > 8:  # Longer variations are more specific
                                score = 90 + priority_bonus - penalty  # Very good match
                            elif len(variation) > 5:
                                score = 80 + priority_bonus - penalty  # Good partial match
                            else:
                                score = 60 + priority_bonus - penalty  # Partial match (abbreviations)
                            break
                    
                    # If no match yet, try key words (but be more selective)
                    if score == 0:
                        # Only use meaningful words (not common ones like "forum", "night", etc.)
                        common_words = {'forum', 'night', 'session', 'event', 'feedback', 'form', 'responses', 'aice', 'nape', '43rd'}
                        event_words = [w for w in event_lower.split() if len(w) > 3 and w not in common_words]
                        
                        if event_words:
                            matched_words = sum(1 for word in event_words if word in sheet_name)
                            if matched_words > 0:
                                # For single-word events (like ACCOMMODATION, FEEDING, TRANSPORTATION),
                                # be more lenient - accept if the word is long enough (>8 chars) or if it's an exact match
                                # For multi-word events, require at least 2 words or 1 very specific word
                                if len(event_words) == 1:
                                    # Single-word event: accept if word is found and is substantial
                                    if len(event_words[0]) > 8 or event_words[0] in sheet_name:
                                        score = 40  # Moderate match for single-word events
                                elif matched_words >= 2:
                                    # For multi-word events, if both/all words match, give higher score
                                    if matched_words == len(event_words):
                                        score = 50  # All words matched - good match
                                    else:
                                        score = 30  # Some words matched - weak match
                                elif matched_words == 1 and len(event_words[0]) > 8:
                                    score = 30  # Single long word matched
                    
                    # Final fallback: Check if the entire event name (as a word) appears in sheet name
                    # This catches cases like "ACCOMMODATION FEEDBACK", "FEEDING FORM", etc.
                    if score == 0:
                        # Check if event name appears as a whole word in sheet name
                        event_word_pattern = r'\b' + re.escape(event_lower) + r'\b'
                        if re.search(event_word_pattern, sheet_name):
                            score = 35  # Fallback match
                    
                    if score > 0:
                        matches.append({
                            'sheet_info': sheet_info,
                            'score': score,
                            'name': sheet_info['name']
                        })
                
                # Sort by score (highest first) and try the best match
                if matches:
                    matches.sort(key=lambda x: x['score'], reverse=True)
                    best_match = matches[0]
                    
                    # Debug: Show top matches for troubleshooting
                    # Always show matches for ALUMNI REUNION to help debug
                    if len(matches) > 0 and (best_match['score'] < 50 or 'alumni' in event_lower):
                        # Low score match or ALUMNI REUNION - show what we found
                        top_3 = matches[:3]
                        print(f"   🔍 Found potential matches (top 3):")
                        for m in top_3:
                            print(f"      - '{m['name']}' (score: {m['score']})")
                    
                    # Try all matches in order until one works
                    last_error = None
                    for match_idx, match in enumerate(matches[:5]):  # Try top 5 matches
                        try:
                            def _get_sheet_data():
                                matching_sheet = self.client.open_by_key(match['sheet_info']['id'])
                                worksheet = matching_sheet.sheet1
                                
                                # Handle duplicate headers (some sheets like BEC have this issue)
                                try:
                                    return worksheet.get_all_records()
                                except Exception as e:
                                    if 'not unique' in str(e).lower() or 'header' in str(e).lower():
                                        # Get headers and make them unique
                                        headers = worksheet.row_values(1)
                                        unique_headers = []
                                        seen = {}
                                        for h in headers:
                                            if h in seen:
                                                seen[h] += 1
                                                unique_headers.append(f"{h}_{seen[h]}")
                                            else:
                                                seen[h] = 0
                                                unique_headers.append(h)
                                        # Get data manually
                                        rows = worksheet.get_all_values()[1:]  # Skip header
                                        return [dict(zip(unique_headers, row)) for row in rows if any(row)]
                                    else:
                                        raise
                            
                            data = self._retry_with_backoff(_get_sheet_data)
                            
                            if data:
                                df = pd.DataFrame(data)
                                df['event_name'] = event_name
                                df['form_url'] = form_url
                                df['extracted_at'] = datetime.now().isoformat()
                                return df
                            else:
                                # Empty sheet - try next match
                                if match_idx == 0:
                                    print(f"   ⚠️  Matched sheet '{match['name']}' but it's empty, trying next match...")
                                continue
                        except Exception as e:
                            last_error = str(e)
                            # If this was the first match, show debug info
                            if match_idx == 0:
                                error_msg = str(e).lower()
                                if 'permission' in error_msg or 'access' in error_msg:
                                    print(f"   ⚠️  Cannot access matched sheet '{match['name']}': Permission denied")
                                elif 'not found' in error_msg:
                                    print(f"   ⚠️  Matched sheet '{match['name']}' not found, trying next match...")
                                # For other errors, try next match silently
                            continue
                    
                    # If we get here, all matches failed
                    if last_error:
                        print(f"   ⚠️  All {len(matches)} potential matches failed. Last error: {last_error[:100]}")
                    
                    # Special debug for specific events - show all matching sheets
                    debug_keywords = []
                    if 'alumni' in event_lower or 'reunion' in event_lower:
                        debug_keywords = ['alumni', 'reunion']
                    elif 'poster' in event_lower:
                        debug_keywords = ['poster', 'presentation']
                    elif 'all' in event_lower and 'convention' in event_lower and 'luncheon' in event_lower:
                        debug_keywords = ['all', 'convention', 'luncheon']
                    
                    if debug_keywords:
                        print(f"   🔍 Debug: Searching for sheets containing {', '.join(debug_keywords)}...")
                        try:
                            all_sheets_debug = self._retry_with_backoff(_list_sheets)
                            matching_sheets = [s for s in all_sheets_debug if any(kw in s['name'].lower() for kw in debug_keywords)]
                            if matching_sheets:
                                print(f"   📋 Found {len(matching_sheets)} sheet(s) with {', '.join(debug_keywords)}:")
                                for sheet in matching_sheets[:5]:
                                    print(f"      - '{sheet['name']}'")
                            else:
                                print(f"   ⚠️  No sheets found containing {', '.join(debug_keywords)}")
                        except Exception as debug_e:
                            print(f"   ⚠️  Could not list sheets for debug: {str(debug_e)[:100]}")
            except Exception as e:
                pass
            
            print(f"⚠️  Could not find response sheet for {event_name}")
            print(f"   💡 Tip: Add 'RESPONSE SHEET LINK' column to master sheet with direct sheet URLs")
            return None
                
        except Exception as e:
            print(f"❌ Error processing {event_name}: {str(e)}")
            return None
    
    def categorize_event(self, event_name: str) -> str:
        """Categorize event based on name"""
        event_lower = event_name.lower()
        
        for category, keywords in EVENT_CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in event_lower:
                    return category
        
        # Default to Professional if no match found (most events are professional)
        return "Professional"
    
    def collect_all_responses(self) -> pd.DataFrame:
        """Collect responses from all forms"""
        events = self.get_events_list()
        all_responses = []
        events_with_responses = {}
        events_without_responses = []
        failed_events = []  # Track events that failed due to rate limits
        
        print(f"\n📊 Found {len(events)} events to process\n")
        
        for i, event in enumerate(events, 1):
            event_name = event['event_name']
            form_url = event['form_link']
            response_sheet_link = event.get('response_sheet_link', '')
            
            print(f"[{i}/{len(events)}] Processing: {event_name}")
            
            try:
                df = self.get_form_responses(form_url, event_name, response_sheet_link)
                
                if df is not None and not df.empty:
                    df['event_category'] = self.categorize_event(event_name)
                    df['form_date'] = event['form_date']
                    df['occurrence'] = event['occurrence']
                    all_responses.append(df)
                    events_with_responses[event_name] = len(df)
                    print(f"  ✅ Collected {len(df)} responses")
                else:
                    events_without_responses.append(event_name)
                    print(f"  ⚠️  No responses found for {event_name} (event will still be tracked)")
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if '429' in error_str or 'quota' in error_str or 'rate limit' in error_str:
                    print(f"  ⚠️  Rate limit hit for {event_name}, will retry later")
                    failed_events.append(event)  # Store the full event for retry
                else:
                    # Other errors - don't retry
                    events_without_responses.append(event_name)
                    print(f"  ⚠️  Error processing {event_name}: {str(e)[:100]}")
            
            # Add a small delay between events to avoid hitting rate limits
            # Google Sheets API allows ~60 requests per minute, so ~1 second delay is safe
            if i < len(events):  # Don't delay after the last event
                time.sleep(1)
        
        # Retry failed events after a longer wait
        if failed_events:
            print(f"\n🔄 Retrying {len(failed_events)} events that hit rate limits...")
            print("   Waiting 30 seconds for rate limit to reset...")
            time.sleep(30)  # Wait for rate limit to reset
            
            for event in failed_events:
                event_name = event['event_name']
                form_url = event['form_link']
                response_sheet_link = event.get('response_sheet_link', '')
                
                print(f"🔄 Retrying: {event_name}")
                try:
                    df = self.get_form_responses(form_url, event_name, response_sheet_link)
                    
                    if df is not None and not df.empty:
                        df['event_category'] = self.categorize_event(event_name)
                        df['form_date'] = event['form_date']
                        df['occurrence'] = event['occurrence']
                        all_responses.append(df)
                        events_with_responses[event_name] = len(df)
                        print(f"  ✅ Collected {len(df)} responses on retry")
                    else:
                        events_without_responses.append(event_name)
                        print(f"  ⚠️  Still no responses found for {event_name}")
                except Exception as e:
                    events_without_responses.append(event_name)
                    print(f"  ❌ Retry failed for {event_name}: {str(e)[:100]}")
                
                # Delay between retries
                time.sleep(2)
        
        if not all_responses:
            raise Exception("No responses collected from any form!")
        
        combined_df = pd.concat(all_responses, ignore_index=True)
        print(f"\n✅ Total responses collected: {len(combined_df)}")
        print(f"✅ Collected from {len(events_with_responses)} events with responses")
        print(f"⚠️  {len(events_without_responses)} events had no responses")
        
        # Print summary
        if events_without_responses:
            print(f"\n📋 Events without responses: {', '.join(events_without_responses)}")
        
        return combined_df


if __name__ == "__main__":
    # Test the ingestion
    ingester = DataIngestion()
    df = ingester.collect_all_responses()
    print(f"\n📋 Sample data:\n{df.head()}")
    print(f"\n📊 Columns: {df.columns.tolist()}")


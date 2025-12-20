"""Main data pipeline: Ingest, process, and store all feedback data"""
from data_ingestion import DataIngestion
from data_processor import DataProcessor
from storage import StorageManager
import sys


def process_and_store_data(storage=None):
    """
    Run the complete data pipeline and store data.
    Can be called from Streamlit app or standalone.
    
    Args:
        storage: Optional StorageManager instance. If None, creates a new one.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Step 1: Ingest data from Google Sheets
        ingester = DataIngestion()
        
        # Get all events first (including those without responses)
        all_events = ingester.get_events_list()
        # Add event_category to each event
        for event in all_events:
            event['event_category'] = ingester.categorize_event(event['event_name'])
        
        # Collect responses
        df = ingester.collect_all_responses()
        
        if df.empty:
            return False, "No data collected from Google Sheets"
        
        # Step 2: Process data
        processor = DataProcessor()
        records = processor.create_response_records(df)
        processed = processor.process_dataframe(df)
        
        # Step 3: Store data
        if storage is None:
            storage = StorageManager()
        storage.store_responses(records, all_events=all_events)
        storage.update_ratings_summary(processed)
        
        return True, f"Successfully loaded {len(records)} responses from {len(df['event_name'].unique())} events"
        
    except FileNotFoundError as e:
        return False, f"Credentials not found: {str(e)}"
    except Exception as e:
        return False, f"Error loading data: {str(e)}"


def main():
    """Run the complete data pipeline"""
    print("=" * 60)
    print("NAPE 43rd AICE Feedback Data Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Ingest data from Google Sheets
        print("\n[1/4] 📥 Ingesting data from Google Sheets...")
        ingester = DataIngestion()
        
        # Get all events first (including those without responses)
        all_events = ingester.get_events_list()
        # Add event_category to each event
        for event in all_events:
            event['event_category'] = ingester.categorize_event(event['event_name'])
        
        # Collect responses
        df = ingester.collect_all_responses()
        
        if df.empty:
            print("❌ No data collected!")
            return
        
        print(f"✅ Collected {len(df)} responses from {len(df['event_name'].unique())} events")
        
        # Step 2: Process data
        print("\n[2/4] 🔄 Processing data...")
        processor = DataProcessor()
        records = processor.create_response_records(df)
        processed = processor.process_dataframe(df)
        
        print(f"✅ Processed {len(records)} response records")
        print(f"   - Found {len(processed['comments'])} comments")
        print(f"   - Found {len(processed['ratings'])} rating questions")
        
        # Step 3: Store data
        print("\n[3/4] 💾 Storing data...")
        storage = StorageManager()
        storage.store_responses(records, all_events=all_events)
        storage.update_ratings_summary(processed)
        
        print("✅ Data stored successfully")
        
        # Step 4: Summary
        print("\n[4/4] 📊 Summary")
        events = storage.get_event_list()
        print(f"   - Total events: {len(events)}")
        print(f"   - Total responses: {sum(e['response_count'] for e in events)}")
        
        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        print("\nYou can now run the Streamlit app with: streamlit run app.py")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease ensure:")
        print("1. credentials.json exists in the project directory")
        print("2. You've followed the SETUP_GUIDE.md instructions")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


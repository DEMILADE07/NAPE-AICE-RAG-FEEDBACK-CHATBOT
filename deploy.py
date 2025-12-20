#!/usr/bin/env python3
"""
Deployment verification and setup script for NAPE RAG Chatbot
"""
import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing ({filepath})")
        return False

def check_import(module_name, description):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: Installed")
        return True
    except ImportError:
        print(f"❌ {description}: Not installed (pip install {module_name})")
        return False

def main():
    print("=" * 60)
    print("NAPE RAG Chatbot - Deployment Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check required files
    print("📁 Checking Required Files...")
    all_checks_passed &= check_file_exists("app.py", "Streamlit app")
    all_checks_passed &= check_file_exists("pipeline.py", "Data pipeline")
    all_checks_passed &= check_file_exists("storage.py", "Storage module")
    all_checks_passed &= check_file_exists("rag_engine.py", "RAG engine")
    all_checks_passed &= check_file_exists("config.py", "Configuration")
    all_checks_passed &= check_file_exists("requirements.txt", "Requirements file")
    
    # Check credentials (warn but don't fail if missing - might be in secrets)
    credentials_exist = check_file_exists("credentials.json", "Google credentials")
    if not credentials_exist:
        print("   ⚠️  Warning: credentials.json not found")
        print("      This is OK if using Streamlit Cloud secrets or environment variables")
    
    print()
    
    # Check Python version
    print("🐍 Checking Python Version...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 9:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}: OK")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}: Requires Python 3.9+")
        all_checks_passed = False
    
    print()
    
    # Check critical dependencies
    print("📦 Checking Critical Dependencies...")
    all_checks_passed &= check_import("streamlit", "Streamlit")
    all_checks_passed &= check_import("pandas", "Pandas")
    all_checks_passed &= check_import("chromadb", "ChromaDB")
    all_checks_passed &= check_import("sentence_transformers", "Sentence Transformers")
    all_checks_passed &= check_import("torch", "PyTorch")
    all_checks_passed &= check_import("gspread", "Google Sheets API")
    all_checks_passed &= check_import("ollama", "Ollama client")
    
    print()
    
    # Check database files (optional - might not exist yet)
    print("💾 Checking Database Files...")
    db_exists = Path("nape_feedback.db").exists()
    chroma_exists = Path("chroma_db").exists()
    
    if db_exists and chroma_exists:
        print("✅ Databases: Found (data already loaded)")
    else:
        print("⚠️  Databases: Not found (run 'python pipeline.py' to load data)")
    
    print()
    
    # Summary
    print("=" * 60)
    if all_checks_passed:
        print("✅ All critical checks passed!")
        print()
        print("Next steps:")
        print("1. If databases don't exist, run: python pipeline.py")
        print("2. Start the app: streamlit run app.py")
        print("3. See DEPLOYMENT.md for production deployment")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Add credentials.json (see SETUP_GUIDE.md)")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()


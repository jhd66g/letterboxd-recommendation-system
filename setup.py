#!/usr/bin/env python3
"""
Setup script for the Letterboxd Recommendation System

This script helps set up the environment and run initial data cleaning.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("📁 Checking data files...")
    
    data_dir = Path("data")
    required_files = [
        "cleaned_reviews_v5.csv",
        "cleaned_movie_metadata_v5.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            # Check if file is not empty
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    print(f"✅ {file}: {len(df)} rows")
                else:
                    print(f"⚠️  {file}: Empty file")
                    missing_files.append(file)
            except Exception as e:
                print(f"❌ {file}: Error reading file - {e}")
                missing_files.append(file)
        else:
            print(f"❌ {file}: Not found")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def run_data_cleaning():
    """Run data cleaning if needed"""
    print("🧹 Running data cleaning...")
    
    # Check if raw data exists
    raw_metadata = Path("data/movie_metadata_20250710_182254.csv")
    raw_reviews = Path("data/cleaned_reviews_with_jaaackd.csv")
    
    if not raw_metadata.exists():
        print(f"❌ Raw metadata file not found: {raw_metadata}")
        return False
    
    if not raw_reviews.exists():
        print(f"❌ Raw reviews file not found: {raw_reviews}")
        return False
    
    try:
        # Clean metadata
        print("Cleaning metadata...")
        os.chdir("data_cleaning")
        
        cmd = [
            sys.executable, 
            "clean_movie_metadata.py",
            "../data/movie_metadata_20250710_182254.csv",
            "-o", "../data/cleaned_movie_metadata_v5.csv",
            "-r", "../data/cleaned_reviews_with_jaaackd.csv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Metadata cleaning failed: {result.stderr}")
            return False
        
        # Clean reviews
        print("Cleaning reviews...")
        cmd = [
            sys.executable,
            "clean_reviews_data.py",
            "../data/cleaned_reviews_with_jaaackd.csv",
            "../data/cleaned_movie_metadata_v5.csv",
            "-o", "../data/cleaned_reviews_v5.csv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Reviews cleaning failed: {result.stderr}")
            return False
        
        os.chdir("..")
        print("✅ Data cleaning completed")
        return True
        
    except Exception as e:
        os.chdir("..")
        print(f"❌ Data cleaning failed: {e}")
        return False

def test_system():
    """Test the recommendation system"""
    print("🧪 Testing recommendation system...")
    
    try:
        # Import the engine
        from recommendation_engine import LetterboxdRecommendationEngine, RecommendationConfig
        
        # Create engine
        config = RecommendationConfig(num_recommendations=5)
        engine = LetterboxdRecommendationEngine(config)
        
        # Load data
        if not engine.load_data():
            print("❌ Failed to load data")
            return False
        
        # Test with a known user
        test_user = 'schaffrillas'
        if engine.check_user_exists(test_user):
            print(f"✅ Test user '{test_user}' found in dataset")
            
            # Generate quick recommendations
            recommendations = engine.generate_recommendations(test_user)
            if recommendations is not None:
                print(f"✅ Successfully generated {len(recommendations)} recommendations")
                return True
            else:
                print("❌ Failed to generate recommendations")
                return False
        else:
            print(f"⚠️  Test user '{test_user}' not found, but system is working")
            return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Letterboxd Recommendation System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check data files
    data_ok, missing_files = check_data_files()
    
    if not data_ok:
        print(f"\n⚠️  Missing data files: {missing_files}")
        print("Attempting to run data cleaning...")
        
        if not run_data_cleaning():
            print("❌ Setup failed - could not clean data")
            print("\nNext steps:")
            print("1. Ensure raw data files are in the data/ directory")
            print("2. Run the data cleaning scripts manually")
            return
        
        # Check again after cleaning
        data_ok, missing_files = check_data_files()
        if not data_ok:
            print(f"❌ Still missing files after cleaning: {missing_files}")
            return
    
    # Test the system
    if test_system():
        print("\n✅ Setup completed successfully!")
        print("=" * 50)
        print("\n🎉 Your Letterboxd Recommendation System is ready!")
        print("\nQuick start commands:")
        print("1. python demo.py                          # Run demo")
        print("2. python recommendation_engine.py schaffrillas  # Get recommendations")
        print("3. python api.py                           # Start API server")
        print("\nFor more information, see README.md")
    else:
        print("\n❌ Setup completed but system test failed")
        print("Please check the error messages above and try again")

if __name__ == "__main__":
    main()

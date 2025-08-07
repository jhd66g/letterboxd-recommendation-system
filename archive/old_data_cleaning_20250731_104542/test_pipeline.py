#!/usr/bin/env python3
"""
Test Script for Streamlined Pipeline

Tests each component of the pipeline individually to ensure everything works.
"""

import os
import sys
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_user_scraping():
    """Test the user scraping functionality."""
    logger.info("🧪 Testing user scraping...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scrapers'))
        from get_top_users import scrape_top_users
        
        # Test with small number of users
        result = scrape_top_users(num_users=10)
        logger.info(f"✅ User scraping test passed")
        return True
    except Exception as e:
        logger.error(f"❌ User scraping test failed: {e}")
        return False

def test_unique_movies_extraction():
    """Test the unique movies extraction with sample data."""
    logger.info("🧪 Testing unique movies extraction...")
    
    try:
        sys.path.append(os.path.dirname(__file__))
        from unique_movies_extractor import extract_unique_movies
        
        # Create sample review data
        sample_reviews = pd.DataFrame([
            {'username': 'user1', 'title': 'The Matrix', 'year': 1999, 'rating': 4.5},
            {'username': 'user1', 'title': 'Inception', 'year': 2010, 'rating': 4.0},
            {'username': 'user2', 'title': 'The Matrix', 'year': 1999, 'rating': 5.0},
            {'username': 'user2', 'title': 'Pulp Fiction', 'year': 1994, 'rating': 4.5},
            {'username': 'user3', 'title': 'The Matrix', 'year': 1999, 'rating': 4.0},
        ])
        
        unique_movies = extract_unique_movies(sample_reviews, min_user_reviews=1, min_movie_reviews=1)
        
        if len(unique_movies) > 0:
            logger.info(f"✅ Unique movies extraction test passed - found {len(unique_movies)} movies")
            return True
        else:
            logger.error("❌ No unique movies found in test")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unique movies extraction test failed: {e}")
        return False

def test_data_cleaner():
    """Test the data cleaner with sample data."""
    logger.info("🧪 Testing data cleaner...")
    
    try:
        sys.path.append(os.path.dirname(__file__))
        from data_cleaner import clean_final_dataset
        
        # Create sample data
        sample_reviews = pd.DataFrame([
            {'username': 'user1', 'title': 'The Matrix', 'year': 1999, 'rating': 4.5},
            {'username': 'user2', 'title': 'The Matrix', 'year': 1999, 'rating': 5.0},
        ])
        
        sample_metadata = pd.DataFrame([
            {
                'title': 'The Matrix',
                'year': 1999,
                'tmdb_id': 603,
                'directors': ['Wachowski Sisters'],
                'cast': ['Keanu Reeves', 'Laurence Fishburne'],
                'genres': ['Action', 'Sci-Fi']
            }
        ])
        
        cleaned_reviews, cleaned_metadata = clean_final_dataset(
            sample_reviews, 
            sample_metadata,
            min_director_appearances=1,
            min_cast_appearances=1
        )
        
        if len(cleaned_reviews) > 0 and len(cleaned_metadata) > 0:
            logger.info(f"✅ Data cleaner test passed")
            return True
        else:
            logger.error("❌ Data cleaner returned empty results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data cleaner test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting pipeline component tests...")
    
    tests = [
        ("User Scraping", test_user_scraping),
        ("Unique Movies Extraction", test_unique_movies_extraction),
        ("Data Cleaner", test_data_cleaner),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test {test_name} failed!")
    
    logger.info(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests passed! Pipeline components are ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

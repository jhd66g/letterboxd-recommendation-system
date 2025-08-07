#!/usr/bin/env python3
"""
Quick Test Pipeline

Runs the complete pipeline with smaller numbers for quick testing.
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add scrapers and data_cleaning to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scrapers'))
sys.path.append(os.path.dirname(__file__))

from get_top_users import scrape_top_users
from review_scraper_explicit import scrape_reviews_for_users
from unique_movies_extractor import extract_unique_movies
from tmdb_metadata_fetcher import fetch_metadata_for_movies
from data_cleaner import clean_final_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete streamlined pipeline with small numbers for testing"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info("🚀 Starting QUICK TEST of streamlined Letterboxd data pipeline")
    
    # Step 1: Fetch top users (small number for testing)
    logger.info("📥 Step 1: Fetching top 20 Letterboxd users (quick test)")
    users_file = os.path.join(data_dir, f'test_users_{timestamp}.csv')
    
    try:
        users_df = scrape_top_users(num_users=20)
        if isinstance(users_df, str):  # If it returns a filename, load it
            users_df = pd.read_csv(users_df)
        users_df.to_csv(users_file, index=False)
        logger.info(f"✅ Fetched {len(users_df)} users, saved to {users_file}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch users: {e}")
        return False
    
    # Step 2: Scrape review data (explicit ratings only, limited reviews)
    logger.info("📝 Step 2: Scraping review data (explicit ratings only, max 50 per user)")
    reviews_file = os.path.join(data_dir, f'test_reviews_{timestamp}.csv')
    
    try:
        reviews_df = scrape_reviews_for_users(users_df, explicit_only=True, max_reviews_per_user=50)
        reviews_df.to_csv(reviews_file, index=False)
        logger.info(f"✅ Scraped {len(reviews_df)} reviews, saved to {reviews_file}")
    except Exception as e:
        logger.error(f"❌ Failed to scrape reviews: {e}")
        return False
    
    # Step 3: Extract unique movies
    logger.info("🎬 Step 3: Extracting unique movies from review data")
    unique_movies_file = os.path.join(data_dir, f'test_unique_movies_{timestamp}.csv')
    
    try:
        unique_movies_df = extract_unique_movies(
            reviews_df, 
            min_user_reviews=5,  # Lower for testing
            min_movie_reviews=3   # Lower for testing
        )
        unique_movies_df.to_csv(unique_movies_file, index=False)
        logger.info(f"✅ Extracted {len(unique_movies_df)} unique movies, saved to {unique_movies_file}")
    except Exception as e:
        logger.error(f"❌ Failed to extract unique movies: {e}")
        return False
    
    # Step 4: Fetch TMDB metadata (for a limited number)
    logger.info("🎭 Step 4: Fetching TMDB metadata")
    metadata_file = os.path.join(data_dir, f'test_metadata_{timestamp}.csv')
    
    try:
        # Limit to first 50 movies for quick test
        limited_movies = unique_movies_df.head(50)
        metadata_df = fetch_metadata_for_movies(limited_movies)
        metadata_df.to_csv(metadata_file, index=False)
        logger.info(f"✅ Fetched metadata for {len(metadata_df)} movies, saved to {metadata_file}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch metadata: {e}")
        return False
    
    # Step 5: Clean and prepare final dataset
    logger.info("🧹 Step 5: Cleaning and preparing final dataset")
    final_reviews_file = os.path.join(data_dir, f'test_cleaned_reviews_{timestamp}.csv')
    final_metadata_file = os.path.join(data_dir, f'test_cleaned_metadata_{timestamp}.csv')
    
    try:
        clean_reviews_df, clean_metadata_df = clean_final_dataset(
            reviews_df, 
            metadata_df,
            min_director_appearances=2,  # Lower for testing
            min_cast_appearances=3       # Lower for testing
        )
        
        clean_reviews_df.to_csv(final_reviews_file, index=False)
        clean_metadata_df.to_csv(final_metadata_file, index=False)
        
        logger.info(f"✅ Cleaned dataset saved:")
        logger.info(f"   Reviews: {len(clean_reviews_df)} rows -> {final_reviews_file}")
        logger.info(f"   Metadata: {len(clean_metadata_df)} rows -> {final_metadata_file}")
    except Exception as e:
        logger.error(f"❌ Failed to clean dataset: {e}")
        return False
    
    logger.info("🎉 Quick test pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

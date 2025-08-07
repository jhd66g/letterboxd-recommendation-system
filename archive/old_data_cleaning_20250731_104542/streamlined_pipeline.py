#!/usr/bin/env python3
"""
Streamlined Letterboxd Data Pipeline

This script implements the complete data pipeline according to the plan:
1. Fetch top ~500 letterboxd users
2. Scrape letterboxd for their review data (only explicit ratings)
3. Pull unique movies from review data (unique title + year combos)
4. Pull metadata from TMDB
5. Clean and prepare final dataset

Usage:
    python streamlined_pipeline.py
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

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
    """Run the complete streamlined pipeline"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info("🚀 Starting streamlined Letterboxd data pipeline")
    
    # Step 1: Fetch top users
    logger.info("📥 Step 1: Fetching top ~500 Letterboxd users")
    users_file = os.path.join(data_dir, f'letterboxd_users_{timestamp}.csv')
    
    try:
        users_df = scrape_top_users(num_users=500)
        if isinstance(users_df, str):  # If it returns a filename, load it
            users_df = pd.read_csv(users_df)
        users_df.to_csv(users_file, index=False)
        logger.info(f"✅ Fetched {len(users_df)} users, saved to {users_file}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch users: {e}")
        return False
    
    # Step 2: Scrape review data (explicit ratings only)
    logger.info("📝 Step 2: Scraping review data (explicit ratings only)")
    reviews_file = os.path.join(data_dir, f'reviews_{timestamp}.csv')
    
    try:
        reviews_df = scrape_reviews_for_users(users_df, explicit_only=True)
        reviews_df.to_csv(reviews_file, index=False)
        logger.info(f"✅ Scraped {len(reviews_df)} reviews, saved to {reviews_file}")
    except Exception as e:
        logger.error(f"❌ Failed to scrape reviews: {e}")
        return False
    
    # Step 3: Extract unique movies
    logger.info("🎬 Step 3: Extracting unique movies from review data")
    unique_movies_file = os.path.join(data_dir, f'unique_movies_{timestamp}.csv')
    
    try:
        unique_movies_df = extract_unique_movies(
            reviews_df, 
            min_user_reviews=10, 
            min_movie_reviews=5
        )
        unique_movies_df.to_csv(unique_movies_file, index=False)
        logger.info(f"✅ Extracted {len(unique_movies_df)} unique movies, saved to {unique_movies_file}")
    except Exception as e:
        logger.error(f"❌ Failed to extract unique movies: {e}")
        return False
    
    # Step 4: Fetch TMDB metadata
    logger.info("🎭 Step 4: Fetching TMDB metadata")
    metadata_file = os.path.join(data_dir, f'movie_metadata_{timestamp}.csv')
    
    try:
        metadata_df = fetch_metadata_for_movies(unique_movies_df)
        metadata_df.to_csv(metadata_file, index=False)
        logger.info(f"✅ Fetched metadata for {len(metadata_df)} movies, saved to {metadata_file}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch metadata: {e}")
        return False
    
    # Step 5: Clean and prepare final dataset
    logger.info("🧹 Step 5: Cleaning and preparing final dataset")
    final_reviews_file = os.path.join(data_dir, f'cleaned_reviews_{timestamp}.csv')
    final_metadata_file = os.path.join(data_dir, f'cleaned_movie_metadata_{timestamp}.csv')
    
    try:
        clean_reviews_df, clean_metadata_df = clean_final_dataset(
            reviews_df, 
            metadata_df,
            min_director_appearances=3,
            min_cast_appearances=5
        )
        
        clean_reviews_df.to_csv(final_reviews_file, index=False)
        clean_metadata_df.to_csv(final_metadata_file, index=False)
        
        logger.info(f"✅ Cleaned dataset saved:")
        logger.info(f"   Reviews: {len(clean_reviews_df)} rows -> {final_reviews_file}")
        logger.info(f"   Metadata: {len(clean_metadata_df)} rows -> {final_metadata_file}")
    except Exception as e:
        logger.error(f"❌ Failed to clean dataset: {e}")
        return False
    
    # Run demo to test the new data
    logger.info("🧪 Running demo on new data")
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from streamlined_demo import main as run_demo
        # Run demo with the cleaned data files
        sys.argv = ['streamlined_demo.py']  # Reset argv for demo
        run_demo()
        logger.info("✅ Demo completed successfully")
    except Exception as e:
        logger.error(f"⚠️  Demo failed: {e}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

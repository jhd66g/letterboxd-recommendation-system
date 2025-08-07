#!/usr/bin/env python3
"""
Full Production Pipeline

Runs the complete pipeline with production-scale parameters for building
a comprehensive Letterboxd recommendation dataset.
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
    """Run the complete streamlined pipeline with production parameters"""
    
    logger.info("🚀 Starting FULL PRODUCTION Letterboxd data pipeline")
    
    # Get timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Production parameters
    NUM_USERS = 500  # Get top 500 users
    MAX_REVIEWS_PER_USER = 500  # Up to 500 reviews per user
    MAX_TMDB_MOVIES = 1000  # Fetch metadata for top 1000 movies
    
    try:
        # Step 1: Get top users
        logger.info(f"📥 Step 1: Fetching top {NUM_USERS} Letterboxd users")
        users_file = scrape_top_users(
            output_file=f"../data/production_users_{timestamp}.csv",
            num_users=NUM_USERS
        )
        logger.info(f"✅ Fetched {NUM_USERS} users, saved to {users_file}")
        
        # Step 2: Scrape reviews (explicit ratings only)
        logger.info(f"📝 Step 2: Scraping review data (explicit ratings only, max {MAX_REVIEWS_PER_USER} per user)")
        
        # Load users DataFrame
        users_df = pd.read_csv(users_file)
        
        # Scrape reviews
        reviews_df = scrape_reviews_for_users(
            users_df=users_df,
            max_reviews_per_user=MAX_REVIEWS_PER_USER,
            explicit_only=True
        )
        
        # Save reviews
        reviews_file = f"../data/production_reviews_{timestamp}.csv"
        reviews_df.to_csv(reviews_file, index=False)
        
        logger.info(f"✅ Scraped {len(reviews_df)} reviews, saved to {reviews_file}")
        
        # Step 3: Extract unique movies
        logger.info("🎬 Step 3: Extracting unique movies from review data")
        unique_movies_file = extract_unique_movies(
            reviews_file=reviews_file,
            output_file=f"../data/production_unique_movies_{timestamp}.csv"
        )
        logger.info(f"✅ Extracted unique movies, saved to {unique_movies_file}")
        
        # Step 4: Fetch TMDB metadata
        logger.info("🎭 Step 4: Fetching TMDB metadata")
        metadata_file = fetch_metadata_for_movies(
            unique_movies_file=unique_movies_file,
            output_file=f"../data/production_metadata_{timestamp}.csv",
            max_movies=MAX_TMDB_MOVIES
        )
        logger.info(f"✅ Fetched TMDB metadata, saved to {metadata_file}")
        
        # Step 5: Clean and prepare final dataset
        logger.info("🧹 Step 5: Cleaning and preparing final dataset")
        final_reviews, final_metadata = clean_final_dataset(
            reviews_file=reviews_file,
            metadata_file=metadata_file,
            output_reviews=f"../data/production_cleaned_reviews_{timestamp}.csv",
            output_metadata=f"../data/production_cleaned_metadata_{timestamp}.csv"
        )
        logger.info("✅ Final dataset cleaning completed")
        
        logger.info("🎉 Full production pipeline completed successfully!")
        logger.info(f"📊 Final dataset summary:")
        logger.info(f"   Reviews: {len(final_reviews)} rows")
        logger.info(f"   Movies: {len(final_metadata)} rows")
        logger.info(f"   Users: {final_reviews['username'].nunique()}")
        logger.info(f"   Timestamp: {timestamp}")
        
        return final_reviews, final_metadata
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed at step: {e}")
        raise

if __name__ == "__main__":
    main()

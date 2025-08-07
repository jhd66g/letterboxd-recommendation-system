#!/usr/bin/env python3
"""
Resume Production Pipeline

Resumes the pipeline from Step 2 (review scraping) using existing users file.
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

from ultra_fast_scraper import scrape_reviews_ultra_fast
from ultra_fast_unique_movies_extractor import extract_unique_movies_ultra_fast
from ultra_fast_tmdb_fetcher import fetch_metadata_for_movies
from ultra_fast_data_cleaner import clean_final_dataset_ultra_fast

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Resume the pipeline from step 2 with existing users"""
    
    logger.info("🚀 Resuming FULL PRODUCTION Letterboxd data pipeline from Step 2")
    
    # Get timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Production parameters  
    MAX_TMDB_MOVIES = 1000  # Fetch metadata for top 1000 movies
    
    # Use existing users file
    users_file = "../data/production_users_20250730_211212.csv"
    
    try:
        logger.info(f"📂 Using existing users file: {users_file}")
        
        # Step 2: Scrape reviews (explicit ratings only)
        logger.info(f"📝 Step 2: Ultra-fast parallel scraping of ALL review data (explicit ratings only)")
        
        # Load users DataFrame
        users_df = pd.read_csv(users_file)
        logger.info(f"   Loaded {len(users_df)} users")
        
        # Ultra-fast parallel scraping of ALL reviews (20 parallel workers)
        reviews_df = scrape_reviews_ultra_fast(
            users_df=users_df,
            explicit_only=True,
            max_workers=20
        )
        
        # Save reviews
        reviews_file = f"../data/production_reviews_{timestamp}.csv"
        reviews_df.to_csv(reviews_file, index=False)
        
        logger.info(f"✅ Scraped {len(reviews_df)} reviews, saved to {reviews_file}")
        
        # Step 3: Extract unique movies
        logger.info("🎬 Step 3: Ultra-fast extraction of unique movies from review data")
        
        # Extract unique movies using ultra-fast method
        unique_movies_df = extract_unique_movies_ultra_fast(reviews_df)
        
        # Save unique movies
        unique_movies_file = f"../data/production_unique_movies_{timestamp}.csv"
        unique_movies_df.to_csv(unique_movies_file, index=False)
        
        logger.info(f"✅ Extracted {len(unique_movies_df)} unique movies, saved to {unique_movies_file}")
        
        # Step 4: Fetch TMDB metadata
        logger.info("🎭 Step 4: Ultra-fast TMDB metadata fetching")
        
        # Limit to top movies for TMDB processing if needed
        if len(unique_movies_df) > MAX_TMDB_MOVIES:
            logger.info(f"   Limiting to top {MAX_TMDB_MOVIES} most popular movies")
            unique_movies_df = unique_movies_df.head(MAX_TMDB_MOVIES)
        
        # Fetch metadata using ultra-fast fetcher with 30 parallel workers
        movies_with_metadata_df = fetch_metadata_for_movies(unique_movies_df, max_workers=30)
        
        # Save metadata
        metadata_file = f"../data/production_metadata_{timestamp}.csv"
        movies_with_metadata_df.to_csv(metadata_file, index=False)
        
        logger.info(f"✅ Fetched TMDB metadata, saved to {metadata_file}")
        
        # Step 5: Clean and prepare final dataset
        logger.info("🧹 Step 5: Ultra-fast cleaning and preparing final dataset")
        final_reviews, final_metadata = clean_final_dataset_ultra_fast(
            reviews_df=reviews_df,
            metadata_df=movies_with_metadata_df
        )
        
        # Save final cleaned datasets
        final_reviews_file = f"../data/production_cleaned_reviews_{timestamp}.csv"
        final_metadata_file = f"../data/production_cleaned_metadata_{timestamp}.csv"
        
        final_reviews.to_csv(final_reviews_file, index=False)
        final_metadata.to_csv(final_metadata_file, index=False)
        
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

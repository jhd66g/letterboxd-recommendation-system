#!/usr/bin/env python3
"""
Chunked TMDB Metadata Fetcher

Processes large datasets in manageable chunks to avoid stalling.
Based on the ultra-fast fetcher but with chunk-based processing.
"""

import pandas as pd
import sys
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import math

# Load environment variables
load_dotenv()

sys.path.append('.')
from ultra_fast_tmdb_fetcher import fetch_metadata_for_movies

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_metadata_in_chunks(unique_movies_df, chunk_size=1000, max_workers=15):
    """
    Fetch TMDB metadata in manageable chunks to avoid stalling.
    
    Args:
        unique_movies_df: DataFrame with unique movies
        chunk_size: Number of movies per chunk (default: 1000)
        max_workers: Number of parallel workers per chunk (default: 15)
    
    Returns:
        Combined DataFrame with all metadata
    """
    total_movies = len(unique_movies_df)
    num_chunks = math.ceil(total_movies / chunk_size)
    
    logger.info(f'🎬 Chunked TMDB metadata fetching')
    logger.info(f'   Total movies: {total_movies:,}')
    logger.info(f'   Chunk size: {chunk_size:,}')
    logger.info(f'   Number of chunks: {num_chunks}')
    logger.info(f'   Workers per chunk: {max_workers}')
    
    all_metadata = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_movies)
        
        chunk_df = unique_movies_df.iloc[start_idx:end_idx].copy()
        
        logger.info(f'📦 Processing chunk {chunk_idx + 1}/{num_chunks}')
        logger.info(f'   Movies {start_idx + 1:,} to {end_idx:,} ({len(chunk_df):,} movies)')
        
        try:
            # Fetch metadata for this chunk
            chunk_metadata = fetch_metadata_for_movies(chunk_df, max_workers=max_workers)
            all_metadata.append(chunk_metadata)
            
            logger.info(f'   ✅ Completed chunk {chunk_idx + 1}: {len(chunk_metadata):,} movies processed')
            
            # Save progress after each chunk
            if all_metadata:
                combined_df = pd.concat(all_metadata, ignore_index=True)
                progress_file = f'data/tmdb_progress_chunk_{chunk_idx + 1}.csv'
                combined_df.to_csv(progress_file, index=False)
                logger.info(f'   💾 Progress saved: {len(combined_df):,} total movies')
            
        except Exception as e:
            logger.error(f'   ❌ Error processing chunk {chunk_idx + 1}: {e}')
            continue
    
    if all_metadata:
        final_df = pd.concat(all_metadata, ignore_index=True)
        logger.info(f'🚀 CHUNKED TMDB FETCH COMPLETE')
        logger.info(f'   Total movies processed: {len(final_df):,}')
        logger.info(f'   Success rate: {len(final_df) / total_movies * 100:.1f}%')
        return final_df
    else:
        logger.error('❌ No metadata fetched successfully')
        return pd.DataFrame()

def main():
    """Main execution function for chunked TMDB fetching."""
    
    # Load the unique movies data
    logger.info('📖 Loading unique movies data...')
    unique_movies_df = pd.read_csv('../data/production_unique_movies_20250730_215832.csv')
    logger.info(f'   Loaded {len(unique_movies_df):,} unique movies')
    
    # Fetch metadata in chunks
    movie_metadata_df = fetch_metadata_in_chunks(
        unique_movies_df, 
        chunk_size=1000,  # Process 1000 movies at a time
        max_workers=15    # Conservative worker count
    )
    
    if len(movie_metadata_df) > 0:
        # Save final results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_file = f'../data/production_movie_metadata_{timestamp}.csv'
        movie_metadata_df.to_csv(metadata_file, index=False)
        logger.info(f'💾 Final metadata saved to {metadata_file}')
        
        print(f'SUCCESS: Chunked processing completed with {len(movie_metadata_df):,} movies')
    else:
        print('FAILED: No movies processed successfully')

if __name__ == "__main__":
    main()

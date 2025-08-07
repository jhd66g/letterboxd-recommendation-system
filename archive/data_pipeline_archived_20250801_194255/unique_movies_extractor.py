#!/usr/bin/env python3
"""
Ultra-Fast Unique Movies Extractor

Optimized for maximum speed with parallel processing and efficient data structures.
Handles massive datasets with minimal memory usage and fast processing.

Functions:
    extract_unique_movies_ultra_fast(reviews_df, min_user_reviews=10, min_movie_reviews=5) -> pd.DataFrame
"""

import pandas as pd
import re
import logging
from collections import Counter, defaultdict
import numpy as np
from multiprocessing import Pool, cpu_count
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_title_vectorized(titles):
    """Vectorized title cleaning for better performance."""
    # Convert to numpy array for vectorized operations
    titles = pd.Series(titles, dtype=str)
    
    # Remove articles and normalize
    titles = titles.str.replace(r'\b(The|A|An)\s+', '', regex=True, case=False)
    
    # Remove special characters but keep letters, numbers, spaces
    titles = titles.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    
    # Normalize whitespace and convert to lowercase
    titles = titles.str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
    
    return titles

def process_chunk(chunk_data):
    """Process a chunk of reviews for parallel processing."""
    chunk, min_user_reviews, min_movie_reviews = chunk_data
    
    # Filter users with sufficient reviews
    user_counts = chunk['username'].value_counts()
    valid_users = user_counts[user_counts >= min_user_reviews].index
    chunk_filtered = chunk[chunk['username'].isin(valid_users)]
    
    # Create movie identifiers
    chunk_filtered = chunk_filtered.copy()
    chunk_filtered['clean_title'] = clean_title_vectorized(chunk_filtered['title'])
    chunk_filtered['year_str'] = chunk_filtered['year'].fillna('Unknown').astype(str)
    chunk_filtered['movie_id'] = chunk_filtered['clean_title'] + '_' + chunk_filtered['year_str']
    
    return chunk_filtered

def extract_unique_movies_ultra_fast(reviews_df, min_user_reviews=10, min_movie_reviews=5, use_parallel=True):
    """
    Ultra-fast extraction of unique movies with parallel processing and optimizations.
    
    Args:
        reviews_df (pd.DataFrame): Reviews with columns ['username', 'title', 'year', 'rating']
        min_user_reviews (int): Minimum reviews per user to include
        min_movie_reviews (int): Minimum appearances for a movie to be included
        use_parallel (bool): Whether to use parallel processing for large datasets
    
    Returns:
        pd.DataFrame: Unique movies with columns ['title', 'year', 'clean_title', 'appearance_count']
    """
    
    start_time = time.time()
    logger.info(f"🚀 Starting ULTRA-FAST unique movies extraction...")
    logger.info(f"   Initial reviews: {len(reviews_df):,}")
    logger.info(f"   Using parallel processing: {use_parallel}")
    
    # Step 1: Remove rows with missing titles
    df = reviews_df.dropna(subset=['title']).copy()
    logger.info(f"   After removing missing titles: {len(df):,}")
    
    # Step 2: Parallel processing for large datasets
    if use_parallel and len(df) > 100000:
        logger.info("   Using parallel processing for large dataset...")
        
        # Split data into chunks for parallel processing
        num_processes = min(cpu_count(), 8)  # Limit to 8 processes
        chunk_size = len(df) // num_processes
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        logger.info(f"   Split into {len(chunks)} chunks for {num_processes} processes")
        
        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            chunk_data = [(chunk, min_user_reviews, min_movie_reviews) for chunk in chunks]
            processed_chunks = pool.map(process_chunk, chunk_data)
        
        # Combine results
        df_processed = pd.concat(processed_chunks, ignore_index=True)
        
    else:
        # Single-threaded processing for smaller datasets
        logger.info("   Using single-threaded processing...")
        
        # Filter users with sufficient reviews
        user_counts = df['username'].value_counts()
        valid_users = user_counts[user_counts >= min_user_reviews].index
        df_processed = df[df['username'].isin(valid_users)].copy()
        
        logger.info(f"   Users with >= {min_user_reviews} reviews: {len(valid_users):,}")
        logger.info(f"   Reviews after user filtering: {len(df_processed):,}")
        
        # Create movie identifiers with vectorized operations
        df_processed['clean_title'] = clean_title_vectorized(df_processed['title'])
        df_processed['year_str'] = df_processed['year'].fillna('Unknown').astype(str)
        df_processed['movie_id'] = df_processed['clean_title'] + '_' + df_processed['year_str']
    
    # Step 3: Fast movie counting using Counter
    logger.info("   Counting movie appearances...")
    movie_counts = Counter(df_processed['movie_id'])
    valid_movie_ids = {movie_id for movie_id, count in movie_counts.items() if count >= min_movie_reviews}
    
    logger.info(f"   Movies appearing >= {min_movie_reviews} times: {len(valid_movie_ids):,}")
    
    # Step 4: Create unique movies DataFrame efficiently
    logger.info("   Creating unique movies DataFrame...")
    df_valid = df_processed[df_processed['movie_id'].isin(valid_movie_ids)]
    
    # Use groupby with efficient aggregation
    unique_movies = df_valid.groupby('movie_id').agg({
        'title': 'first',
        'year': 'first',
        'clean_title': 'first',
        'username': 'count'
    }).reset_index()
    
    # Rename count column and clean up
    unique_movies = unique_movies.rename(columns={'username': 'appearance_count'})
    unique_movies = unique_movies.drop('movie_id', axis=1)
    
    # Handle year column efficiently
    unique_movies.loc[unique_movies['year'] == 'Unknown', 'year'] = None
    unique_movies['year'] = pd.to_numeric(unique_movies['year'], errors='coerce')
    
    # Sort efficiently
    unique_movies = unique_movies.sort_values(['appearance_count', 'title'], ascending=[False, True])
    unique_movies = unique_movies.reset_index(drop=True)
    
    # Calculate final stats
    elapsed_time = time.time() - start_time
    processing_rate = len(reviews_df) / elapsed_time
    
    logger.info(f"\n🚀 ULTRA-FAST UNIQUE MOVIES EXTRACTION COMPLETE")
    logger.info(f"   Total processing time: {elapsed_time:.1f} seconds")
    logger.info(f"   Processing rate: {processing_rate:,.0f} reviews per second")
    logger.info(f"   Total unique movies: {len(unique_movies):,}")
    logger.info(f"   Movies with years: {unique_movies['year'].notna().sum():,}")
    logger.info(f"   Movies without years: {unique_movies['year'].isna().sum():,}")
    logger.info(f"   Average appearances per movie: {unique_movies['appearance_count'].mean():.1f}")
    if len(unique_movies) > 0:
        logger.info(f"   Most popular movie: {unique_movies.iloc[0]['title']} ({unique_movies.iloc[0]['appearance_count']} appearances)")
    
    return unique_movies

def get_movie_statistics_fast(unique_movies_df):
    """Get detailed statistics about the unique movies dataset with optimizations."""
    
    stats = {
        'total_movies': len(unique_movies_df),
        'movies_with_years': unique_movies_df['year'].notna().sum(),
        'movies_without_years': unique_movies_df['year'].isna().sum(),
        'year_range': {
            'min': unique_movies_df['year'].min(),
            'max': unique_movies_df['year'].max()
        } if unique_movies_df['year'].notna().any() else {'min': None, 'max': None},
        'appearance_stats': {
            'min': unique_movies_df['appearance_count'].min(),
            'max': unique_movies_df['appearance_count'].max(),
            'mean': unique_movies_df['appearance_count'].mean(),
            'median': unique_movies_df['appearance_count'].median()
        },
        'top_10_movies': unique_movies_df.head(10)[['title', 'year', 'appearance_count']].to_dict('records')
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.csv', '_unique_movies.csv')
        
        print(f"🚀 Loading reviews from {input_file}...")
        reviews_df = pd.read_csv(input_file)
        
        print(f"🎬 Extracting unique movies ultra-fast...")
        unique_movies_df = extract_unique_movies_ultra_fast(reviews_df)
        
        print(f"💾 Saving results to {output_file}...")
        unique_movies_df.to_csv(output_file, index=False)
        
        print(f"✅ Complete! Extracted {len(unique_movies_df)} unique movies.")
        
        # Print statistics
        stats = get_movie_statistics_fast(unique_movies_df)
        print(f"\n📊 Statistics:")
        print(f"   Total movies: {stats['total_movies']:,}")
        print(f"   With years: {stats['movies_with_years']:,}")
        print(f"   Without years: {stats['movies_without_years']:,}")
        print(f"   Year range: {stats['year_range']['min']} - {stats['year_range']['max']}")
        print(f"   Avg appearances: {stats['appearance_stats']['mean']:.1f}")
    else:
        print("Usage: python ultra_fast_unique_movies_extractor.py <input_csv> [output_csv]")

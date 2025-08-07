#!/usr/bin/env python3
"""
Ultra-Fast Final Data Cleaner

Optimized for maximum speed with vectorized operations and parallel processing.
Cleans and prepares the final dataset with minimal memory usage.

Functions:
    clean_final_dataset_ultra_fast(reviews_df, metadata_df) -> tuple
"""

import pandas as pd
import logging
from collections import Counter, defaultdict
import ast
import numpy as np
import time
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_eval_list_vectorized(series):
    """Vectorized version of safe list evaluation."""
    def safe_eval_single(x):
        # Handle None first
        if x is None:
            return []
        
        # Handle pandas NA values carefully to avoid array truth value error
        try:
            if pd.isna(x):
                return []
        except (ValueError, TypeError):
            # If pd.isna fails (e.g., on arrays), continue with other checks
            pass
            
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            if x == '':
                return []
            try:
                return ast.literal_eval(x)
            except:
                if x.startswith('[') and x.endswith(']'):
                    content = x[1:-1]
                    if content.strip():
                        return [item.strip().strip("'\"") for item in content.split(',')]
                return []
        try:
            if hasattr(x, '__iter__') and not isinstance(x, str):
                return list(x)
        except:
            pass
        return []
    
    return series.apply(safe_eval_single)

def normalize_title(title):
    """Normalize title for flexible matching."""
    if pd.isna(title):
        return ""
    return str(title).lower().strip().replace(' ', '').replace('-', '').replace(':', '').replace('.', '').replace("'", '').replace('"', '')

def create_title_year_mapping_fast(metadata_df):
    """Ultra-fast creation of title-year to TMDB ID mapping using vectorized operations with flexible matching."""
    
    logger.info("🚀 Creating title-year to TMDB ID mapping (ultra-fast with flexible matching)...")
    
    # Filter out rows without TMDB IDs
    valid_metadata = metadata_df[metadata_df['tmdb_id'].notna()].copy()
    
    # Create normalized title mapping for flexible matching
    valid_metadata['title_norm'] = valid_metadata['title'].apply(normalize_title)
    
    # Create both exact and flexible mappings
    # 1. Exact title-year mapping (for precision)
    valid_metadata['year_clean'] = valid_metadata['year'].where(valid_metadata['year'].notna(), None)
    title_year_pairs = list(zip(valid_metadata['title'], valid_metadata['year_clean']))
    tmdb_ids = valid_metadata['tmdb_id'].tolist()
    exact_mapping = dict(zip(title_year_pairs, tmdb_ids))
    
    # 2. Flexible title-only mapping (for coverage)
    flexible_mapping = dict(zip(valid_metadata['title_norm'], valid_metadata['tmdb_id']))
    
    # Combine both mappings
    mapping = {
        'exact': exact_mapping,
        'flexible': flexible_mapping
    }
    
    logger.info(f"   Created exact mapping for {len(exact_mapping):,} title-year combinations")
    logger.info(f"   Created flexible mapping for {len(flexible_mapping):,} normalized titles")
    return mapping

def add_tmdb_ids_to_reviews_fast(reviews_df, title_year_mapping):
    """Ultra-fast addition of TMDB IDs to reviews using flexible matching."""
    
    logger.info("🚀 Adding TMDB IDs to reviews (ultra-fast with flexible matching)...")
    
    reviews_with_ids = reviews_df.copy()
    
    # Step 1: Try exact title-year matching first
    reviews_with_ids['year_clean'] = reviews_with_ids['year'].where(reviews_with_ids['year'].notna(), None)
    title_year_pairs = list(zip(reviews_with_ids['title'], reviews_with_ids['year_clean']))
    
    exact_mapping = title_year_mapping['exact']
    tmdb_ids = [exact_mapping.get(pair) for pair in title_year_pairs]
    
    # Step 2: For unmatched entries, try flexible title matching
    flexible_mapping = title_year_mapping['flexible']
    reviews_with_ids['title_norm'] = reviews_with_ids['title'].apply(normalize_title)
    
    unmatched_indices = [i for i, tmdb_id in enumerate(tmdb_ids) if tmdb_id is None]
    logger.info(f"   Trying flexible matching for {len(unmatched_indices):,} unmatched reviews...")
    
    for idx in unmatched_indices:
        title_norm = reviews_with_ids.iloc[idx]['title_norm']
        if title_norm in flexible_mapping:
            tmdb_ids[idx] = flexible_mapping[title_norm]
    
    reviews_with_ids['tmdb_id'] = tmdb_ids
    
    matched_count = sum(1 for tmdb_id in tmdb_ids if tmdb_id is not None)
    exact_matches = len(tmdb_ids) - len(unmatched_indices)
    flexible_matches = matched_count - exact_matches
    
    logger.info(f"   Exact matches: {exact_matches:,}")
    logger.info(f"   Flexible matches: {flexible_matches:,}")
    logger.info(f"   Total matched: {matched_count:,}/{len(reviews_with_ids):,} reviews ({matched_count/len(reviews_with_ids)*100:.1f}%)")
    
    return reviews_with_ids

def clean_final_dataset_ultra_fast(reviews_df, metadata_df):
    """
    Ultra-fast cleaning and preparation of the final dataset.
    
    Args:
        reviews_df (pd.DataFrame): Raw reviews data
        metadata_df (pd.DataFrame): TMDB metadata
        min_director_appearances (int): Minimum director appearances to keep in main column
        min_cast_appearances (int): Minimum cast appearances to keep in main column
    
    Returns:
        tuple: (cleaned_reviews_df, cleaned_metadata_df)
    """
    
    start_time = time.time()
    logger.info("🚀 Starting ULTRA-FAST final dataset cleaning...")
    
    # Step 1: Create title-year to TMDB ID mapping (ultra-fast)
    title_year_mapping = create_title_year_mapping_fast(metadata_df)
    
    # Step 2: Add TMDB IDs to reviews (ultra-fast)
    cleaned_reviews = add_tmdb_ids_to_reviews_fast(reviews_df, title_year_mapping)

    # Step 2.5: Fill missing years in metadata using release_date
    logger.info("🚀 Filling missing years in metadata using release_date...")
    # Extract year from release_date (first 4 characters) for metadata
    metadata_missing_year = metadata_df['year'].isna()
    metadata_has_release_date = metadata_df['release_date'].notna()
    fillable_metadata = metadata_missing_year & metadata_has_release_date
    
    metadata_df.loc[fillable_metadata, 'year'] = metadata_df.loc[fillable_metadata, 'release_date'].str[:4].astype(int)
    
    logger.info(f"   Filled {fillable_metadata.sum():,} missing years in metadata from release_date")
    
    # Step 2.6: Fill missing years in reviews using updated metadata (by tmdb_id)
    logger.info("🚀 Filling missing years in reviews using metadata...")
    # Build tmdb_id -> year mapping from updated metadata (handle duplicates by taking first)
    metadata_for_mapping = metadata_df.drop_duplicates(subset=['tmdb_id'], keep='first')
    tmdbid_to_year = metadata_for_mapping.set_index('tmdb_id')['year'].to_dict()
    # Only fill where year is missing (NaN or None)
    missing_year_mask = cleaned_reviews['year'].isna()
    # Map tmdb_id to year for missing years
    cleaned_reviews.loc[missing_year_mask, 'year'] = cleaned_reviews.loc[missing_year_mask, 'tmdb_id'].map(tmdbid_to_year)

    # Step 3: Use metadata as-is without director/cast processing
    cleaned_metadata = metadata_df.copy()

    # Step 4: Filter reviews to only include those with TMDB IDs (vectorized)
    logger.info("🚀 Filtering reviews with TMDB IDs...")
    reviews_with_tmdb = cleaned_reviews[cleaned_reviews['tmdb_id'].notna()].copy()

    # Step 5: Filter metadata to only include movies that appear in reviews (vectorized)
    logger.info("🚀 Filtering metadata for used movies...")
    used_tmdb_ids = set(reviews_with_tmdb['tmdb_id'].unique())
    metadata_used = cleaned_metadata[cleaned_metadata['tmdb_id'].isin(used_tmdb_ids)].copy()
    
    # Step 6: Convert tmdb_id and year columns to integers where possible
    logger.info("🚀 Converting tmdb_id and year columns to integers...")
    
    # Convert tmdb_id to nullable int for reviews
    reviews_with_tmdb['tmdb_id'] = reviews_with_tmdb['tmdb_id'].astype('Int64')
    
    # Convert year to nullable int for reviews
    reviews_with_tmdb['year'] = reviews_with_tmdb['year'].astype('Int64')
    
    # Convert tmdb_id to nullable int for metadata
    metadata_used['tmdb_id'] = metadata_used['tmdb_id'].astype('Int64')
    
    # Convert year to nullable int for metadata
    metadata_used['year'] = metadata_used['year'].astype('Int64')

    # Calculate final stats
    elapsed_time = time.time() - start_time

    logger.info(f"\n🚀 ULTRA-FAST FINAL DATASET CLEANING COMPLETE")
    logger.info(f"   Total processing time: {elapsed_time:.1f} seconds")
    logger.info(f"   Processing rate: {len(reviews_df)/elapsed_time:,.0f} reviews per second")
    logger.info(f"   Reviews with TMDB IDs: {len(reviews_with_tmdb):,}")
    logger.info(f"   Unique movies in final dataset: {len(metadata_used):,}")
    logger.info(f"   Unique users in final dataset: {reviews_with_tmdb['username'].nunique():,}")

    return reviews_with_tmdb, metadata_used

def get_dataset_statistics_fast(reviews_df, metadata_df):
    """Get comprehensive statistics about the final dataset with optimizations."""
    
    # Use vectorized operations for statistics
    user_stats = reviews_df.groupby('username').size()
    movie_stats = reviews_df.groupby('tmdb_id').size()
    
    def safe_len_for_stats(x):
        try:
            if pd.isna(x) or x is None:
                return 0
            if isinstance(x, list):
                return len(x)
            if isinstance(x, str):
                try:
                    parsed = ast.literal_eval(x)
                    return len(parsed) if isinstance(parsed, list) else 0
                except:
                    return 0
            return 0
        except:
            return 0
    
    stats = {
        'reviews': {
            'total_reviews': len(reviews_df),
            'unique_users': reviews_df['username'].nunique(),
            'unique_movies': reviews_df['tmdb_id'].nunique(),
            'reviews_per_user': {
                'mean': user_stats.mean(),
                'median': user_stats.median(),
                'min': user_stats.min(),
                'max': user_stats.max()
            },
            'reviews_per_movie': {
                'mean': movie_stats.mean(),
                'median': movie_stats.median(),
                'min': movie_stats.min(),
                'max': movie_stats.max()
            }
        },
        'metadata': {
            'total_movies': len(metadata_df),
            'movies_with_release_date': metadata_df['release_date'].notna().sum(),
            'average_genres_per_movie': metadata_df['genres'].apply(safe_len_for_stats).mean(),
        }
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 2:
        reviews_file = sys.argv[1]
        metadata_file = sys.argv[2]
        output_reviews = sys.argv[3] if len(sys.argv) > 3 else reviews_file.replace('.csv', '_cleaned.csv')
        output_metadata = sys.argv[4] if len(sys.argv) > 4 else metadata_file.replace('.csv', '_cleaned.csv')
        
        print(f"🚀 Loading data...")
        reviews_df = pd.read_csv(reviews_file)
        metadata_df = pd.read_csv(metadata_file)
        
        print(f"🧹 Cleaning dataset ultra-fast...")
        cleaned_reviews, cleaned_metadata = clean_final_dataset_ultra_fast(reviews_df, metadata_df)
        
        print(f"💾 Saving results...")
        cleaned_reviews.to_csv(output_reviews, index=False)
        cleaned_metadata.to_csv(output_metadata, index=False)
        
        print(f"✅ Complete!")
        
        # Print statistics
        stats = get_dataset_statistics_fast(cleaned_reviews, cleaned_metadata)
        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Reviews: {stats['reviews']['total_reviews']:,}")
        print(f"   Users: {stats['reviews']['unique_users']:,}")
        print(f"   Movies: {stats['reviews']['unique_movies']:,}")
        print(f"   Avg reviews per user: {stats['reviews']['reviews_per_user']['mean']:.1f}")
        print(f"   Avg reviews per movie: {stats['reviews']['reviews_per_movie']['mean']:.1f}")
    else:
        print("Usage: python ultra_fast_data_cleaner.py <reviews_csv> <metadata_csv> [output_reviews] [output_metadata]")

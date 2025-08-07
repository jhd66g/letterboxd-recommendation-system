#!/usr/bin/env python3
"""
Final Data Cleaner

Cleans and prepares the final dataset:
1. Add TMDB ID to all review entries for lookup
2. For directors appearing < 3 times, move to director_other column
3. For cast members appearing < 5 times, move to cast_other column

Functions:
    clean_final_dataset(reviews_df, metadata_df, min_director_appearances=3, min_cast_appearances=5)
"""

import pandas as pd
import logging
from collections import Counter
import ast

logger = logging.getLogger(__name__)

def safe_eval_list(x):
    """Safely evaluate string representation of list."""
    # Handle None first
    if x is None:
        return []
    
    # Handle lists/arrays directly
    if isinstance(x, list):
        return x
    
    # Handle pandas NaN values
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        # pd.isna can fail on some data types, continue processing
        pass
    
    # Handle string representations
    if isinstance(x, str):
        if x == '':
            return []
        try:
            # Try to evaluate as literal
            return ast.literal_eval(x)
        except:
            # If that fails, try simple parsing
            if x.startswith('[') and x.endswith(']'):
                # Remove brackets and split by comma
                content = x[1:-1]
                if content.strip():
                    return [item.strip().strip("'\"") for item in content.split(',')]
            return []
    
    # Handle other iterable types
    try:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return list(x)
    except:
        pass
    
    return []

def create_title_year_mapping(reviews_df, metadata_df):
    """Create mapping from (title, year) to TMDB ID."""
    
    logger.info("🔗 Creating title-year to TMDB ID mapping...")
    
    # Create mapping dictionary
    mapping = {}
    
    for _, row in metadata_df.iterrows():
        if pd.notna(row['tmdb_id']):
            title = row['title']
            year = row['year'] if pd.notna(row['year']) else None
            tmdb_id = row['tmdb_id']
            
            # Create key (title, year)
            key = (title, year)
            mapping[key] = tmdb_id
    
    logger.info(f"   Created mapping for {len(mapping)} title-year combinations")
    return mapping

def add_tmdb_ids_to_reviews(reviews_df, title_year_mapping):
    """Add TMDB IDs to review dataset using title-year mapping."""
    
    logger.info("🆔 Adding TMDB IDs to reviews...")
    
    reviews_with_ids = reviews_df.copy()
    reviews_with_ids['tmdb_id'] = None
    
    matched_count = 0
    
    for idx, row in reviews_with_ids.iterrows():
        title = row['title']
        year = row['year'] if pd.notna(row['year']) else None
        
        # Try exact match first
        key = (title, year)
        if key in title_year_mapping:
            reviews_with_ids.at[idx, 'tmdb_id'] = title_year_mapping[key]
            matched_count += 1
        else:
            # Try match without year if year was provided
            if year is not None:
                key_no_year = (title, None)
                if key_no_year in title_year_mapping:
                    reviews_with_ids.at[idx, 'tmdb_id'] = title_year_mapping[key_no_year]
                    matched_count += 1
    
    logger.info(f"   Matched {matched_count}/{len(reviews_with_ids)} reviews to TMDB IDs")
    logger.info(f"   Match rate: {matched_count/len(reviews_with_ids)*100:.1f}%")
    
    return reviews_with_ids

def process_directors_and_cast(metadata_df, min_director_appearances=3, min_cast_appearances=5):
    """
    Process directors and cast, moving infrequent ones to 'other' columns.
    
    Args:
        metadata_df (pd.DataFrame): Metadata with directors and cast columns
        min_director_appearances (int): Minimum appearances to keep director in main column
        min_cast_appearances (int): Minimum appearances to keep cast member in main column
    
    Returns:
        pd.DataFrame: Processed metadata with main and 'other' columns
    """
    
    logger.info(f"🎬 Processing directors and cast (min directors: {min_director_appearances}, min cast: {min_cast_appearances})...")
    
    processed_df = metadata_df.copy()
    
    # Initialize new columns
    processed_df['directors_main'] = None
    processed_df['directors_other'] = None
    processed_df['cast_main'] = None
    processed_df['cast_other'] = None
    
    # Count director appearances across all movies
    all_directors = []
    for _, row in processed_df.iterrows():
        directors = safe_eval_list(row['directors'])
        all_directors.extend(directors)
    
    director_counts = Counter(all_directors)
    frequent_directors = {director for director, count in director_counts.items() 
                         if count >= min_director_appearances}
    
    logger.info(f"   Directors: {len(director_counts)} total, {len(frequent_directors)} frequent (>= {min_director_appearances} appearances)")
    
    # Count cast appearances across all movies
    all_cast = []
    for _, row in processed_df.iterrows():
        cast = safe_eval_list(row['cast'])
        all_cast.extend(cast)
    
    cast_counts = Counter(all_cast)
    frequent_cast = {cast_member for cast_member, count in cast_counts.items() 
                    if count >= min_cast_appearances}
    
    logger.info(f"   Cast: {len(cast_counts)} total, {len(frequent_cast)} frequent (>= {min_cast_appearances} appearances)")
    
    # Process each movie
    for idx, row in processed_df.iterrows():
        # Process directors
        directors = safe_eval_list(row['directors'])
        main_directors = [d for d in directors if d in frequent_directors]
        other_directors = [d for d in directors if d not in frequent_directors]
        
        processed_df.at[idx, 'directors_main'] = main_directors
        processed_df.at[idx, 'directors_other'] = other_directors
        
        # Process cast
        cast = safe_eval_list(row['cast'])
        main_cast = [c for c in cast if c in frequent_cast]
        other_cast = [c for c in cast if c not in frequent_cast]
        
        processed_df.at[idx, 'cast_main'] = main_cast
        processed_df.at[idx, 'cast_other'] = other_cast
    
    # Calculate statistics with safer approach
    def safe_len(x):
        try:
            if pd.isna(x):
                return 0
            return len(x) if hasattr(x, '__len__') else 0
        except:
            return 0
    
    avg_main_directors = processed_df['directors_main'].apply(safe_len).mean()
    avg_other_directors = processed_df['directors_other'].apply(safe_len).mean()
    avg_main_cast = processed_df['cast_main'].apply(safe_len).mean()
    avg_other_cast = processed_df['cast_other'].apply(safe_len).mean()
    
    logger.info(f"   Average main directors per movie: {avg_main_directors:.1f}")
    logger.info(f"   Average other directors per movie: {avg_other_directors:.1f}")
    logger.info(f"   Average main cast per movie: {avg_main_cast:.1f}")
    logger.info(f"   Average other cast per movie: {avg_other_cast:.1f}")
    
    return processed_df

def clean_final_dataset(reviews_df, metadata_df, min_director_appearances=3, min_cast_appearances=5):
    """
    Clean and prepare the final dataset.
    
    Args:
        reviews_df (pd.DataFrame): Raw reviews data
        metadata_df (pd.DataFrame): TMDB metadata
        min_director_appearances (int): Minimum director appearances to keep in main column
        min_cast_appearances (int): Minimum cast appearances to keep in main column
    
    Returns:
        tuple: (cleaned_reviews_df, cleaned_metadata_df)
    """
    
    logger.info("🧹 Starting final dataset cleaning...")
    
    # Step 1: Create title-year to TMDB ID mapping
    title_year_mapping = create_title_year_mapping(reviews_df, metadata_df)
    
    # Step 2: Add TMDB IDs to reviews
    cleaned_reviews = add_tmdb_ids_to_reviews(reviews_df, title_year_mapping)
    
    # Step 3: Process directors and cast in metadata
    cleaned_metadata = process_directors_and_cast(
        metadata_df, 
        min_director_appearances, 
        min_cast_appearances
    )
    
    # Step 4: Filter reviews to only include those with TMDB IDs
    reviews_with_tmdb = cleaned_reviews[cleaned_reviews['tmdb_id'].notna()].copy()
    
    # Step 5: Filter metadata to only include movies that appear in reviews
    used_tmdb_ids = set(reviews_with_tmdb['tmdb_id'].unique())
    metadata_used = cleaned_metadata[cleaned_metadata['tmdb_id'].isin(used_tmdb_ids)].copy()
    
    logger.info(f"\\n✅ FINAL DATASET CLEANING COMPLETE")
    logger.info(f"   Reviews with TMDB IDs: {len(reviews_with_tmdb)}")
    logger.info(f"   Unique movies in final dataset: {len(metadata_used)}")
    logger.info(f"   Unique users in final dataset: {reviews_with_tmdb['username'].nunique()}")
    
    return reviews_with_tmdb, metadata_used

def get_dataset_statistics(reviews_df, metadata_df):
    """Get comprehensive statistics about the final dataset."""
    
    stats = {
        'reviews': {
            'total_reviews': len(reviews_df),
            'unique_users': reviews_df['username'].nunique(),
            'unique_movies': reviews_df['tmdb_id'].nunique(),
            'reviews_per_user': {
                'mean': reviews_df.groupby('username').size().mean(),
                'median': reviews_df.groupby('username').size().median(),
                'min': reviews_df.groupby('username').size().min(),
                'max': reviews_df.groupby('username').size().max()
            },
            'reviews_per_movie': {
                'mean': reviews_df.groupby('tmdb_id').size().mean(),
                'median': reviews_df.groupby('tmdb_id').size().median(),
                'min': reviews_df.groupby('tmdb_id').size().min(),
                'max': reviews_df.groupby('tmdb_id').size().max()
            }
        },
        'metadata': {
            'total_movies': len(metadata_df),
            'movies_with_release_date': metadata_df['release_date'].notna().sum(),
            'average_genres_per_movie': metadata_df['genres'].apply(lambda x: len(safe_eval_list(x))).mean(),
            'average_main_directors': metadata_df['directors_main'].apply(lambda x: len(safe_eval_list(x))).mean(),
            'average_main_cast': metadata_df['cast_main'].apply(lambda x: len(safe_eval_list(x))).mean(),
        }
    }
    
    return stats

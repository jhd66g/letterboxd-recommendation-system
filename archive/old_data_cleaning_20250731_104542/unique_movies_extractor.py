#!/usr/bin/env python3
"""
Unique Movies Extractor

Extracts unique movies from review data based on title + year combinations.
Includes data cleanup and filtering according to specifications:
- Remove reviews from reviewers with < min_user_reviews reviews
- Only include movies that appear >= min_movie_reviews times
- Handle missing years gracefully

Functions:
    extract_unique_movies(reviews_df, min_user_reviews=10, min_movie_reviews=5) -> pd.DataFrame
"""

import pandas as pd
import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def clean_title(title):
    """Clean movie title for better matching."""
    if pd.isna(title):
        return ""
    
    # Convert to string and strip whitespace
    title = str(title).strip()
    
    # Remove common variations
    title = re.sub(r'\\b(The|A|An)\\s+', '', title, flags=re.IGNORECASE)
    
    # Remove special characters but keep letters, numbers, spaces
    title = re.sub(r'[^a-zA-Z0-9\\s]', '', title)
    
    # Normalize whitespace
    title = re.sub(r'\\s+', ' ', title).strip()
    
    return title.lower()

def extract_unique_movies(reviews_df, min_user_reviews=10, min_movie_reviews=5):
    """
    Extract unique movies from review data with filtering.
    
    Args:
        reviews_df (pd.DataFrame): Reviews with columns ['username', 'title', 'year', 'rating']
        min_user_reviews (int): Minimum reviews per user to include
        min_movie_reviews (int): Minimum appearances for a movie to be included
    
    Returns:
        pd.DataFrame: Unique movies with columns ['title', 'year', 'clean_title', 'appearance_count']
    """
    
    logger.info("🎬 Starting unique movies extraction...")
    logger.info(f"   Initial reviews: {len(reviews_df)}")
    
    # Step 1: Clean the data
    df = reviews_df.copy()
    
    # Remove rows with missing titles
    df = df.dropna(subset=['title'])
    logger.info(f"   After removing missing titles: {len(df)}")
    
    # Step 2: Filter users with insufficient reviews
    user_review_counts = df['username'].value_counts()
    valid_users = user_review_counts[user_review_counts >= min_user_reviews].index
    df = df[df['username'].isin(valid_users)]
    
    logger.info(f"   Users with >= {min_user_reviews} reviews: {len(valid_users)}")
    logger.info(f"   Reviews after user filtering: {len(df)}")
    
    # Step 3: Clean titles and create movie identifiers
    df['clean_title'] = df['title'].apply(clean_title)
    
    # Create movie identifier (title + year combination)
    # Handle missing years by using 'Unknown' as placeholder
    df['year_str'] = df['year'].fillna('Unknown').astype(str)
    df['movie_id'] = df['clean_title'] + '_' + df['year_str']
    
    # Step 4: Count movie appearances
    movie_counts = df['movie_id'].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_reviews].index
    
    logger.info(f"   Movies appearing >= {min_movie_reviews} times: {len(valid_movies)}")
    
    # Step 5: Create unique movies dataframe
    df_filtered = df[df['movie_id'].isin(valid_movies)]
    
    # Group by movie_id to get unique movies with their info
    unique_movies = df_filtered.groupby('movie_id').agg({
        'title': 'first',  # Take first occurrence of title
        'year': 'first',   # Take first occurrence of year
        'clean_title': 'first',
        'username': 'count'  # Count appearances
    }).reset_index()
    
    # Rename count column
    unique_movies = unique_movies.rename(columns={'username': 'appearance_count'})
    
    # Clean up year column (convert 'Unknown' back to NaN)
    unique_movies.loc[unique_movies['year'] == 'Unknown', 'year'] = None
    unique_movies['year'] = pd.to_numeric(unique_movies['year'], errors='coerce')
    
    # Sort by appearance count (descending) and then by title
    unique_movies = unique_movies.sort_values(['appearance_count', 'title'], ascending=[False, True])
    
    # Drop the movie_id column as it's no longer needed
    unique_movies = unique_movies.drop('movie_id', axis=1)
    
    logger.info(f"\\n✅ UNIQUE MOVIES EXTRACTION COMPLETE")
    logger.info(f"   Total unique movies: {len(unique_movies)}")
    logger.info(f"   Movies with years: {unique_movies['year'].notna().sum()}")
    logger.info(f"   Movies without years: {unique_movies['year'].isna().sum()}")
    logger.info(f"   Average appearances per movie: {unique_movies['appearance_count'].mean():.1f}")
    logger.info(f"   Most popular movie: {unique_movies.iloc[0]['title']} ({unique_movies.iloc[0]['appearance_count']} appearances)")
    
    return unique_movies

def get_movie_statistics(unique_movies_df):
    """Get detailed statistics about the unique movies dataset."""
    
    stats = {
        'total_movies': len(unique_movies_df),
        'movies_with_years': unique_movies_df['year'].notna().sum(),
        'movies_without_years': unique_movies_df['year'].isna().sum(),
        'year_range': {
            'min': unique_movies_df['year'].min(),
            'max': unique_movies_df['year'].max()
        },
        'appearance_stats': {
            'min': unique_movies_df['appearance_count'].min(),
            'max': unique_movies_df['appearance_count'].max(),
            'mean': unique_movies_df['appearance_count'].mean(),
            'median': unique_movies_df['appearance_count'].median()
        },
        'top_10_movies': unique_movies_df.head(10)[['title', 'year', 'appearance_count']].to_dict('records')
    }
    
    return stats

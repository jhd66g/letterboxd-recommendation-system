#!/usr/bin/env python3

"""
Review Data Cleaner for LightFM

This script cleans and preprocesses review data for use in LightFM recommendation models.

Features:
- Adds TMDB IDs for all movies using metadata mapping
- Adds release year for movies missing year information
- Removes movies not found in metadata (not in TMDB)
- Creates user-item interaction matrix suitable for LightFM
- Generates user and item mappings for model training
- Handles missing ratings and normalizes rating scales

Usage:
    python clean_review_data.py reviews.csv metadata.csv -o cleaned_reviews.csv
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

def load_metadata_mapping(metadata_file):
    """
    Load metadata and create mappings for title/year -> tmdb_id and release_year.
    """
    print("📖 Loading metadata for mapping...")
    metadata_df = pd.read_csv(metadata_file)
    
    # Filter only movies found in TMDB (if column exists)
    if 'found_in_tmdb' in metadata_df.columns:
        metadata_df = metadata_df[metadata_df['found_in_tmdb'] == True].copy()
    else:
        # If using cleaned metadata, all movies are already valid TMDB movies
        print("Using cleaned metadata (all movies are valid)")
    
    print(f"Loaded {len(metadata_df)} movies from metadata")
    
    # Create mapping dictionaries
    # Use both input_title and title for mapping (some may differ)
    title_to_tmdb = {}
    title_to_year = {}
    
    for _, row in metadata_df.iterrows():
        tmdb_id = row['tmdb_id']
        
        # Handle different metadata formats
        if 'input_title' in metadata_df.columns:
            input_title = str(row['input_title']).strip()
        else:
            input_title = str(row['title']).strip()
            
        tmdb_title = str(row['title']).strip()
        
        # Get the best year available
        release_year = None
        if 'release_year' in metadata_df.columns and pd.notna(row['release_year']):
            release_year = int(row['release_year'])
        elif 'input_year' in metadata_df.columns and pd.notna(row['input_year']) and row['input_year'] != 'Unknown':
            try:
                release_year = int(float(row['input_year']))
            except (ValueError, TypeError):
                pass
        elif 'release_date' in metadata_df.columns and pd.notna(row['release_date']):
            try:
                release_year = int(str(row['release_date'])[:4])
            except (ValueError, TypeError):
                pass
        
        # Create mappings using input_title (original Letterboxd title)
        if input_title and input_title != 'nan':
            # Try with original year if available
            input_year = row.get('input_year', 'Unknown')
            if pd.isna(input_year):
                input_year = 'Unknown'
            key = (input_title, str(input_year))
            title_to_tmdb[key] = tmdb_id
            if release_year:
                title_to_year[key] = release_year
        
        # Also create mapping using TMDB title as fallback
        if tmdb_title and tmdb_title != 'nan' and tmdb_title != input_title:
            key = (tmdb_title, str(input_year))
            title_to_tmdb[key] = tmdb_id
            if release_year:
                title_to_year[key] = release_year
    
    print(f"Created mappings for {len(title_to_tmdb)} title/year combinations")
    return title_to_tmdb, title_to_year, metadata_df

def clean_review_data(reviews_file, metadata_file, output_file):
    """
    Main function to clean review data for LightFM.
    """
    print("🎬 Review Data Cleaner for LightFM")
    print("=" * 50)
    print(f"Reviews file: {reviews_file}")
    print(f"Metadata file: {metadata_file}")
    print(f"Output file: {output_file}")
    print("=" * 50)
    
    # Load metadata mappings
    title_to_tmdb, title_to_year, metadata_df = load_metadata_mapping(metadata_file)
    
    # Load review data
    print("\n📖 Loading review data...")
    reviews_df = pd.read_csv(reviews_file)
    initial_count = len(reviews_df)
    print(f"Loaded {initial_count:,} reviews")
    
    # Handle missing years in reviews
    print("\n🔧 Processing years...")
    reviews_df['year'] = reviews_df['year'].fillna('Unknown')
    reviews_df['year'] = reviews_df['year'].astype(str)
    
    # Add TMDB ID and release year columns
    print("\n🔗 Mapping to TMDB IDs...")
    
    def get_tmdb_id_and_year(row):
        """Get TMDB ID and release year for a review row."""
        title = str(row['title']).strip()
        year = str(row['year']).strip()
        
        # Try exact match first
        key = (title, year)
        if key in title_to_tmdb:
            tmdb_id = title_to_tmdb[key]
            release_year = title_to_year.get(key, None)
            return tmdb_id, release_year
        
        # Try with 'Unknown' year if year wasn't found
        if year != 'Unknown':
            key = (title, 'Unknown')
            if key in title_to_tmdb:
                tmdb_id = title_to_tmdb[key]
                release_year = title_to_year.get(key, None)
                return tmdb_id, release_year
        
        # Not found
        return None, None
    
    # Apply mapping
    mapping_results = reviews_df.apply(get_tmdb_id_and_year, axis=1, result_type='expand')
    reviews_df['tmdb_id'] = mapping_results[0]
    reviews_df['release_year'] = mapping_results[1]
    
    # Count successful mappings
    mapped_count = reviews_df['tmdb_id'].notna().sum()
    print(f"✅ Successfully mapped {mapped_count:,} reviews ({mapped_count/initial_count*100:.1f}%)")
    
    # Remove reviews without TMDB mapping
    before_filter = len(reviews_df)
    reviews_df = reviews_df[reviews_df['tmdb_id'].notna()].copy()
    removed_count = before_filter - len(reviews_df)
    print(f"✂️  Removed {removed_count:,} reviews without TMDB mapping")
    print(f"📊 Remaining reviews: {len(reviews_df):,}")
    
    if len(reviews_df) == 0:
        print("❌ No reviews remaining after filtering!")
        return
    
    # Fill missing years with release_year where available
    print("\n📅 Fixing missing years...")
    missing_year_mask = (reviews_df['year'] == 'Unknown') | (reviews_df['year'] == 'nan')
    missing_count_before = missing_year_mask.sum()
    
    if missing_count_before > 0:
        # Fill missing years with release_year
        reviews_df.loc[missing_year_mask, 'year'] = reviews_df.loc[missing_year_mask, 'release_year'].fillna('Unknown')
        
        missing_count_after = (reviews_df['year'] == 'Unknown').sum()
        fixed_count = missing_count_before - missing_count_after
        print(f"✅ Fixed {fixed_count:,} missing years using release data")
        print(f"📊 Remaining unknown years: {missing_count_after:,}")
    
    # Convert tmdb_id to integer
    reviews_df['tmdb_id'] = reviews_df['tmdb_id'].astype(int)
    
    # Handle ratings
    print("\n⭐ Processing ratings...")
    
    # Check rating distribution
    rating_dist = reviews_df['rating'].value_counts().sort_index()
    print("Rating distribution:")
    for rating, count in rating_dist.items():
        if pd.notna(rating):
            print(f"  {rating}: {count:,} reviews")
    
    # Count missing ratings
    missing_ratings = reviews_df['rating'].isna().sum()
    print(f"Missing ratings: {missing_ratings:,} ({missing_ratings/len(reviews_df)*100:.1f}%)")
    
    # For LightFM, we can either:
    # 1. Remove reviews without ratings
    # 2. Treat them as implicit feedback (rating = 1)
    # Let's keep them as implicit feedback
    reviews_df['has_rating'] = reviews_df['rating'].notna()
    reviews_df['rating_type'] = reviews_df['has_rating'].map({True: 'explicit', False: 'implicit'})
    
    # Fill missing ratings with 1 (implicit positive feedback)
    reviews_df['rating'] = reviews_df['rating'].fillna(1.0)
    
    # Create user and item mappings for LightFM
    print("\n👥 Creating user mappings...")
    unique_users = reviews_df['username'].unique()
    user_to_id = {user: idx for idx, user in enumerate(unique_users)}
    reviews_df['user_id'] = reviews_df['username'].map(user_to_id)
    
    print(f"📊 Unique users: {len(unique_users):,}")
    
    print("\n🎬 Creating movie mappings...")
    unique_movies = reviews_df['tmdb_id'].unique()
    movie_to_id = {movie: idx for idx, movie in enumerate(unique_movies)}
    reviews_df['movie_id'] = reviews_df['tmdb_id'].map(movie_to_id)
    
    print(f"📊 Unique movies: {len(unique_movies):,}")
    
    # Calculate some statistics
    print("\n📈 Dataset Statistics:")
    user_review_counts = reviews_df['user_id'].value_counts()
    movie_review_counts = reviews_df['movie_id'].value_counts()
    
    print(f"  Average reviews per user: {user_review_counts.mean():.1f}")
    print(f"  Median reviews per user: {user_review_counts.median():.1f}")
    print(f"  Max reviews per user: {user_review_counts.max():,}")
    
    print(f"  Average reviews per movie: {movie_review_counts.mean():.1f}")
    print(f"  Median reviews per movie: {movie_review_counts.median():.1f}")
    print(f"  Max reviews per movie: {movie_review_counts.max():,}")
    
    # Calculate sparsity
    sparsity = 1 - (len(reviews_df) / (len(unique_users) * len(unique_movies)))
    print(f"  Matrix sparsity: {sparsity*100:.2f}%")
    
    # Select final columns for output
    output_columns = [
        'user_id',
        'movie_id', 
        'username',
        'tmdb_id',
        'title',
        'year',
        'release_year',
        'rating',
        'rating_type',
        'has_rating'
    ]
    
    available_columns = [col for col in output_columns if col in reviews_df.columns]
    reviews_clean = reviews_df[available_columns].copy()
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned reviews...")
    reviews_clean.to_csv(output_file, index=False)
    
    # Save user and movie mappings for later use
    user_mapping_file = output_file.replace('.csv', '_user_mapping.csv')
    movie_mapping_file = output_file.replace('.csv', '_movie_mapping.csv')
    
    # Create mapping DataFrames
    user_mapping_df = pd.DataFrame([
        {'user_id': user_id, 'username': username} 
        for username, user_id in user_to_id.items()
    ])
    user_mapping_df.to_csv(user_mapping_file, index=False)
    
    movie_mapping_df = pd.DataFrame([
        {'movie_id': movie_id, 'tmdb_id': tmdb_id} 
        for tmdb_id, movie_id in movie_to_id.items()
    ])
    movie_mapping_df.to_csv(movie_mapping_file, index=False)
    
    print(f"📁 User mapping saved to: {user_mapping_file}")
    print(f"📁 Movie mapping saved to: {movie_mapping_file}")
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 CLEANING COMPLETE!")
    print("=" * 50)
    print(f"📊 Final dataset: {len(reviews_clean):,} reviews")
    print(f"👥 Users: {len(unique_users):,}")
    print(f"🎬 Movies: {len(unique_movies):,}")
    print(f"📁 Main file: {output_file}")
    
    print(f"\n🔧 Ready for LightFM:")
    print(f"  - User-item interactions: ✅")
    print(f"  - User IDs (0 to {len(unique_users)-1}): ✅")
    print(f"  - Movie IDs (0 to {len(unique_movies)-1}): ✅") 
    print(f"  - Ratings (explicit + implicit): ✅")
    print(f"  - Mapping files for interpretation: ✅")

def main():
    parser = argparse.ArgumentParser(description="Clean review data for LightFM recommendation model")
    parser.add_argument("reviews_file", help="Input reviews CSV file")
    parser.add_argument("metadata_file", help="Input metadata CSV file") 
    parser.add_argument("-o", "--output", required=True, help="Output cleaned reviews CSV file")
    
    args = parser.parse_args()
    
    clean_review_data(args.reviews_file, args.metadata_file, args.output)

if __name__ == "__main__":
    main()

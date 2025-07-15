#!/usr/bin/env python3

"""
Data Verification for LightFM

This script verifies that the cleaned movie metadata and review data are properly
prepared for training a LightFM recommendation model.

Usage:
    python verify_lightfm_data.py
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def verify_cleaned_data():
    """
    Verify that cleaned data is ready for LightFM training.
    """
    print("🔍 LightFM Data Verification")
    print("=" * 50)
    
    # Load all files
    print("📖 Loading cleaned data files...")
    
    try:
        reviews_df = pd.read_csv('datacleaning/cleaned_reviews.csv')
        metadata_df = pd.read_csv('datacleaning/cleaned_movie_metadata.csv')
        user_mapping_df = pd.read_csv('datacleaning/cleaned_reviews_user_mapping.csv')
        movie_mapping_df = pd.read_csv('datacleaning/cleaned_reviews_movie_mapping.csv')
        
        print(f"✅ Reviews: {len(reviews_df):,} interactions")
        print(f"✅ Metadata: {len(metadata_df):,} movies")
        print(f"✅ User mappings: {len(user_mapping_df):,} users")
        print(f"✅ Movie mappings: {len(movie_mapping_df):,} movies")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return False
    
    # Verify data integrity
    print(f"\n🔧 Data Integrity Checks:")
    
    # Check user IDs
    user_ids_reviews = set(reviews_df['user_id'].unique())
    user_ids_mapping = set(user_mapping_df['user_id'].unique())
    if user_ids_reviews == user_ids_mapping:
        print(f"✅ User IDs consistent: {len(user_ids_reviews)} users")
    else:
        print(f"❌ User ID mismatch!")
        return False
    
    # Check movie IDs
    movie_ids_reviews = set(reviews_df['movie_id'].unique())
    movie_ids_mapping = set(movie_mapping_df['movie_id'].unique())
    if movie_ids_reviews == movie_ids_mapping:
        print(f"✅ Movie IDs consistent: {len(movie_ids_reviews)} movies")
    else:
        print(f"❌ Movie ID mismatch!")
        return False
    
    # Check TMDB ID alignment
    tmdb_ids_reviews = set(reviews_df['tmdb_id'].unique())
    tmdb_ids_metadata = set(metadata_df['tmdb_id'].unique())
    overlap = len(tmdb_ids_reviews.intersection(tmdb_ids_metadata))
    print(f"✅ TMDB ID overlap: {overlap:,} / {len(tmdb_ids_reviews):,} movies ({overlap/len(tmdb_ids_reviews)*100:.1f}%)")
    
    # Check ratings
    rating_stats = reviews_df['rating'].describe()
    print(f"✅ Ratings range: {rating_stats['min']:.1f} - {rating_stats['max']:.1f}")
    print(f"✅ Average rating: {rating_stats['mean']:.2f}")
    
    # Rating type distribution
    rating_type_dist = reviews_df['rating_type'].value_counts()
    print(f"✅ Explicit ratings: {rating_type_dist.get('explicit', 0):,}")
    print(f"✅ Implicit ratings: {rating_type_dist.get('implicit', 0):,}")
    
    # Verify LightFM requirements
    print(f"\n🎯 LightFM Requirements Check:")
    
    # User IDs should be 0-indexed and contiguous
    max_user_id = reviews_df['user_id'].max()
    min_user_id = reviews_df['user_id'].min()
    unique_users = reviews_df['user_id'].nunique()
    if min_user_id == 0 and max_user_id == unique_users - 1:
        print(f"✅ User IDs: 0-indexed and contiguous (0 to {max_user_id})")
    else:
        print(f"❌ User IDs not properly formatted!")
        return False
    
    # Movie IDs should be 0-indexed and contiguous  
    max_movie_id = reviews_df['movie_id'].max()
    min_movie_id = reviews_df['movie_id'].min()
    unique_movies = reviews_df['movie_id'].nunique()
    if min_movie_id == 0 and max_movie_id == unique_movies - 1:
        print(f"✅ Movie IDs: 0-indexed and contiguous (0 to {max_movie_id})")
    else:
        print(f"❌ Movie IDs not properly formatted!")
        return False
    
    # Check for missing values in critical columns
    critical_review_columns = ['user_id', 'movie_id', 'rating']
    missing_data = False
    for col in critical_review_columns:
        missing_count = reviews_df[col].isna().sum()
        if missing_count > 0:
            print(f"❌ Missing values in {col}: {missing_count}")
            missing_data = True
        else:
            print(f"✅ No missing values in {col}")
    
    if missing_data:
        return False
    
    # Analyze sparsity
    sparsity = 1 - (len(reviews_df) / (unique_users * unique_movies))
    print(f"✅ Matrix sparsity: {sparsity*100:.2f}% (good for collaborative filtering)")
    
    # Analyze movie features for content-based filtering
    print(f"\n🎬 Movie Features Analysis:")
    
    feature_columns = ['runtime_bucket', 'budget_bucket', 'revenue_bucket', 'genres', 'directors', 'cast', 'original_language']
    for col in feature_columns:
        if col in metadata_df.columns:
            if col in ['genres', 'directors', 'cast']:
                # Count unique items in semicolon-separated lists
                all_items = []
                for entry in metadata_df[col].fillna(''):
                    if entry:
                        items = [item.strip() for item in entry.split(';')]
                        all_items.extend(items)
                unique_count = len(set(all_items))
                print(f"  {col}: {unique_count} unique values")
            else:
                unique_count = metadata_df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
    
    # Check data distribution
    print(f"\n📊 Data Distribution:")
    
    # User activity distribution
    user_activity = reviews_df['user_id'].value_counts()
    print(f"  Most active user: {user_activity.max():,} reviews")
    print(f"  Least active user: {user_activity.min():,} reviews")
    print(f"  Average reviews per user: {user_activity.mean():.1f}")
    
    # Movie popularity distribution
    movie_popularity = reviews_df['movie_id'].value_counts()
    print(f"  Most popular movie: {movie_popularity.max():,} reviews")
    print(f"  Least popular movie: {movie_popularity.min():,} reviews")
    print(f"  Average reviews per movie: {movie_popularity.mean():.1f}")
    
    # Recommend next steps
    print(f"\n🚀 Recommendations for LightFM:")
    print(f"  📈 Dataset size: {'Large' if len(reviews_df) > 100000 else 'Medium' if len(reviews_df) > 10000 else 'Small'}")
    print(f"  🎯 Recommendation: Use hybrid model (collaborative + content-based)")
    print(f"  ⚡ Algorithms to try: WARP, BPR, LightFM with features")
    print(f"  🔧 Feature engineering: Consider TF-IDF for text features")
    
    if sparsity > 0.95:
        print(f"  ⚠️  High sparsity detected - consider matrix factorization approaches")
    
    print(f"\n" + "=" * 50)
    print(f"🎉 DATA VERIFICATION COMPLETE!")
    print(f"✅ All checks passed - data is ready for LightFM training!")
    print(f"=" * 50)
    
    return True

def show_sample_data():
    """Show sample data for verification."""
    print(f"\n📋 Sample Data Preview:")
    
    # Sample reviews
    reviews_df = pd.read_csv('datacleaning/cleaned_reviews.csv')
    print(f"\nSample Reviews:")
    print(reviews_df[['user_id', 'movie_id', 'title', 'rating', 'rating_type']].head())
    
    # Sample metadata
    metadata_df = pd.read_csv('datacleaning/cleaned_movie_metadata.csv')
    print(f"\nSample Metadata:")
    print(metadata_df[['tmdb_id', 'title', 'runtime_bucket', 'genres', 'directors']].head())
    
    # Sample mappings
    user_mapping_df = pd.read_csv('datacleaning/cleaned_reviews_user_mapping.csv')
    print(f"\nSample User Mapping:")
    print(user_mapping_df.head())

def main():
    success = verify_cleaned_data()
    
    if success:
        show_sample_data()
        print(f"\n🎯 Ready to build LightFM model!")
        print(f"📝 Next steps:")
        print(f"   1. Create user and item feature matrices")
        print(f"   2. Split data into train/test sets")
        print(f"   3. Train LightFM model with hybrid approach")
        print(f"   4. Evaluate using precision@k, recall@k, AUC")
    else:
        print(f"\n❌ Data verification failed. Please fix issues before proceeding.")

if __name__ == "__main__":
    main()

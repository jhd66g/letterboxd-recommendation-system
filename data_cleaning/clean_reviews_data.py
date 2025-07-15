#!/usr/bin/env python3
"""
Script to clean reviews data to match filtered movie metadata.
"""

import pandas as pd
import argparse

def clean_reviews_data(reviews_file, metadata_file, output_file):
    """
    Clean reviews data to only include movies that are in the cleaned metadata.
    """
    print("🔄 Cleaning reviews data to match filtered metadata...")
    
    # Load data
    print("📖 Loading data...")
    reviews_df = pd.read_csv(reviews_file)
    metadata_df = pd.read_csv(metadata_file)
    
    print(f"Original reviews: {len(reviews_df)}")
    print(f"Movies in metadata: {len(metadata_df)}")
    
    # Get set of valid TMDB IDs from metadata
    valid_tmdb_ids = set(metadata_df['tmdb_id'].values)
    
    # Filter reviews to only include movies in metadata
    before_filter = len(reviews_df)
    reviews_filtered = reviews_df[reviews_df['tmdb_id'].isin(valid_tmdb_ids)].copy()
    after_filter = len(reviews_filtered)
    
    print(f"✂️  Removed {before_filter - after_filter} reviews for filtered movies")
    print(f"📊 Remaining reviews: {after_filter}")
    
    # Save cleaned reviews
    reviews_filtered.to_csv(output_file, index=False)
    print(f"💾 Saved cleaned reviews to: {output_file}")
    
    # Summary statistics
    print(f"\n📊 Summary:")
    print(f"  Users: {reviews_filtered['username'].nunique()}")
    print(f"  Movies: {reviews_filtered['tmdb_id'].nunique()}")
    print(f"  Reviews: {len(reviews_filtered)}")
    print(f"  Average ratings per movie: {len(reviews_filtered) / reviews_filtered['tmdb_id'].nunique():.1f}")

def main():
    parser = argparse.ArgumentParser(description="Clean reviews data to match filtered metadata")
    parser.add_argument("reviews_file", help="Input reviews CSV file")
    parser.add_argument("metadata_file", help="Cleaned metadata CSV file")
    parser.add_argument("-o", "--output", required=True, help="Output cleaned reviews CSV file")
    
    args = parser.parse_args()
    
    clean_reviews_data(args.reviews_file, args.metadata_file, args.output)

if __name__ == "__main__":
    main()

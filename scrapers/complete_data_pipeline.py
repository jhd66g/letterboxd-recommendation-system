#!/usr/bin/env python3

"""
Complete Letterboxd + TMDB Data Pipeline

This script combines the Letterboxd batch scraper with TMDB metadata collection
to create two comprehensive CSV files:
1. Reviews data: All user ratings and reviews from Letterboxd
2. Movie metadata: Detailed movie information from TMDB

Usage:
    python complete_data_pipeline.py --mode pipeline --pages 10 --output-prefix my_data
    python complete_data_pipeline.py --mode metadata-only --input reviews.csv --output-prefix my_data
"""

import argparse
import sys
import os
import time
from datetime import datetime
import pandas as pd

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datascraping.batch_scraper import run_full_pipeline, batch_scrape_users, load_usernames_from_file
from datascraping.get_users import get_popular_users, get_usernames_list
from datascraping.tmdb_metadata_scraper import extract_unique_movies, scrape_movie_metadata, save_metadata_to_csv

def create_output_filenames(prefix=None, timestamp=None):
    """
    Generate standardized output filenames.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if prefix is None:
        prefix = "letterboxd_data"
    
    reviews_file = f"{prefix}_reviews_{timestamp}.csv"
    metadata_file = f"{prefix}_metadata_{timestamp}.csv"
    
    return reviews_file, metadata_file

def run_complete_pipeline(pages=5, time_period="this/week", scraper_type="parallel", 
                         max_users=None, output_prefix=None, max_workers=10):
    """
    Run the complete pipeline: collect users -> scrape reviews -> get metadata.
    """
    print("🚀 Starting Complete Letterboxd + TMDB Data Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate output filenames
    reviews_file, metadata_file = create_output_filenames(output_prefix, timestamp)
    
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Reviews output: {reviews_file}")
    print(f"Metadata output: {metadata_file}")
    print("=" * 60)
    
    # Step 1: Run the Letterboxd batch scraper
    print("\n📋 Step 1: Collecting Letterboxd review data...")
    
    reviews_df = run_full_pipeline(
        max_pages=pages,
        time_period=time_period,
        scraper_type=scraper_type,
        output_file=reviews_file,
        save_users=True,
        max_users=max_users
    )
    
    if reviews_df.empty:
        print("❌ No review data collected. Exiting.")
        return None, None
    
    print(f"✅ Collected {len(reviews_df)} movie reviews")
    
    # Step 2: Extract unique movies and get TMDB metadata
    print(f"\n🎬 Step 2: Collecting TMDB metadata...")
    
    # Filter movies with at least 4 reviews to reduce TMDB API calls
    movies = extract_unique_movies(reviews_file, min_reviews=4)
    if not movies:
        print("❌ No movies found for metadata collection.")
        return reviews_df, None
    
    print(f"Found {len(movies)} unique movies with ≥4 reviews for metadata collection")
    
    metadata_list = scrape_movie_metadata(movies, max_workers=max_workers)
    
    if not metadata_list:
        print("❌ No metadata collected.")
        return reviews_df, None
    
    # Save metadata to CSV
    save_metadata_to_csv(metadata_list, metadata_file)
    metadata_df = pd.DataFrame(metadata_list)
    
    # Final summary
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n📊 Pipeline completed successfully!")
    print("=" * 60)
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    print(f"Review data: {len(reviews_df)} entries -> {reviews_file}")
    print(f"Metadata: {len(metadata_df)} movies -> {metadata_file}")
    
    # Show some statistics
    if not metadata_df.empty:
        found_count = metadata_df['found_in_tmdb'].sum() if 'found_in_tmdb' in metadata_df.columns else len(metadata_df)
        print(f"TMDB match rate: {found_count}/{len(metadata_df)} ({found_count/len(metadata_df)*100:.1f}%)")
    
    return reviews_df, metadata_df

def run_metadata_only(input_file, output_prefix=None, max_workers=10):
    """
    Run only the metadata collection step on existing review data.
    """
    print("🎬 TMDB Metadata Collection Only")
    print("=" * 50)
    
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate output filename
    if output_prefix is None:
        base_name = os.path.splitext(input_file)[0]
        output_prefix = f"{base_name}"
    
    _, metadata_file = create_output_filenames(output_prefix, timestamp)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {metadata_file}")
    print("=" * 50)
    
    # Extract unique movies with minimum review filter
    movies = extract_unique_movies(input_file, min_reviews=4)
    if not movies:
        print("❌ No movies found in input file.")
        return None
    
    print(f"Found {len(movies)} unique movies with ≥4 reviews for metadata collection")
    
    # Get metadata
    metadata_list = scrape_movie_metadata(movies, max_workers=max_workers)
    
    if not metadata_list:
        print("❌ No metadata collected.")
        return None
    
    # Save metadata to CSV
    save_metadata_to_csv(metadata_list, metadata_file)
    metadata_df = pd.DataFrame(metadata_list)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"📈 Throughput: {len(metadata_df) / execution_time:.2f} movies/second")
    
    return metadata_df

def analyze_combined_data(reviews_file, metadata_file):
    """
    Perform basic analysis on the combined dataset.
    """
    print("\n📊 Data Analysis")
    print("=" * 30)
    
    try:
        reviews_df = pd.read_csv(reviews_file)
        metadata_df = pd.read_csv(metadata_file)
        
        print(f"Reviews dataset: {len(reviews_df)} entries")
        print(f"Metadata dataset: {len(metadata_df)} movies")
        
        # Basic review statistics
        total_users = reviews_df['username'].nunique() if 'username' in reviews_df.columns else 0
        rated_reviews = reviews_df[reviews_df['rating'].notna()] if 'rating' in reviews_df.columns else pd.DataFrame()
        
        print(f"\nReview Statistics:")
        print(f"- Unique users: {total_users}")
        print(f"- Total reviews: {len(reviews_df)}")
        print(f"- Rated reviews: {len(rated_reviews)}")
        if not rated_reviews.empty:
            print(f"- Average rating: {rated_reviews['rating'].mean():.2f}")
        
        # Metadata statistics
        if 'found_in_tmdb' in metadata_df.columns:
            found_count = metadata_df['found_in_tmdb'].sum()
            print(f"\nMetadata Statistics:")
            print(f"- Movies found in TMDB: {found_count}/{len(metadata_df)} ({found_count/len(metadata_df)*100:.1f}%)")
        
        if 'genres' in metadata_df.columns:
            # Most common genres
            all_genres = []
            for genres_str in metadata_df['genres'].dropna():
                if genres_str:
                    all_genres.extend([g.strip() for g in genres_str.split(';')])
            
            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts().head(10)
                print(f"\nTop 10 Genres:")
                for genre, count in genre_counts.items():
                    print(f"- {genre}: {count} movies")
        
        # Data quality check
        print(f"\nData Quality:")
        reviews_missing_title = reviews_df['title'].isna().sum() if 'title' in reviews_df.columns else 0
        reviews_missing_year = reviews_df['year'].isna().sum() if 'year' in reviews_df.columns else 0
        
        print(f"- Reviews missing title: {reviews_missing_title}")
        print(f"- Reviews missing year: {reviews_missing_year}")
        
        if 'tmdb_id' in metadata_df.columns:
            metadata_missing_tmdb = metadata_df['tmdb_id'].isna().sum()
            print(f"- Movies missing TMDB ID: {metadata_missing_tmdb}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

def main():
    parser = argparse.ArgumentParser(description="Complete Letterboxd + TMDB data collection pipeline")
    
    # Mode selection
    parser.add_argument("--mode", choices=["pipeline", "metadata-only", "analyze"], default="pipeline",
                       help="Mode: 'pipeline' (full pipeline), 'metadata-only' (existing reviews), 'analyze' (analyze existing data)")
    
    # Pipeline options (for full pipeline mode)
    parser.add_argument("-p", "--pages", type=int, default=5,
                       help="Number of user pages to scrape (default: 5)")
    parser.add_argument("-t", "--time-period", default="this/week",
                       choices=["this/week", "this/month", "this/year", "all-time"],
                       help="Time period for user popularity (default: this/week)")
    parser.add_argument("-s", "--scraper", choices=["pro", "parallel"], default="parallel",
                       help="Letterboxd scraper type (default: parallel)")
    parser.add_argument("--max-users", type=int,
                       help="Maximum number of users to process")
    
    # Input/Output options
    parser.add_argument("-i", "--input",
                       help="Input CSV file (for metadata-only mode)")
    parser.add_argument("--output-prefix",
                       help="Prefix for output filenames")
    parser.add_argument("--reviews-file",
                       help="Reviews CSV file (for analyze mode)")
    parser.add_argument("--metadata-file",
                       help="Metadata CSV file (for analyze mode)")
    
    # Performance options
    parser.add_argument("-w", "--workers", type=int, default=10,
                       help="Number of TMDB API workers (default: 10)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: process only small subset")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        # Run complete pipeline
        if args.test:
            print("⚠️  TEST MODE: Limited processing")
            max_users = min(args.max_users or 5, 5)
            pages = min(args.pages, 1)
        else:
            max_users = args.max_users
            pages = args.pages
        
        reviews_df, metadata_df = run_complete_pipeline(
            pages=pages,
            time_period=args.time_period,
            scraper_type=args.scraper,
            max_users=max_users,
            output_prefix=args.output_prefix,
            max_workers=args.workers
        )
        
    elif args.mode == "metadata-only":
        # Run only metadata collection
        if not args.input:
            print("Error: --input is required for metadata-only mode")
            sys.exit(1)
        
        metadata_df = run_metadata_only(
            args.input,
            output_prefix=args.output_prefix,
            max_workers=args.workers
        )
        
    elif args.mode == "analyze":
        # Analyze existing data
        if not args.reviews_file or not args.metadata_file:
            print("Error: --reviews-file and --metadata-file are required for analyze mode")
            sys.exit(1)
        
        analyze_combined_data(args.reviews_file, args.metadata_file)
    
    print("\n🎉 Pipeline completed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Movie Metadata Cleaner for LightFM

This script cleans and preprocesses movie metadata for use in LightFM recommendation models.

Features:
- Filters out movies not found in TMDB
- Adds release year from release_date for movies missing input_year
- Creates runtime buckets (<90, 90-120, 120-150, >150 minutes)
- Creates budget buckets (Low, Medium, High)
- Creates revenue buckets (Low, Medium, High)
- Consolidates directors/cast: keeps frequent ones, moves others to "other" category
- Prepares categorical features suitable for LightFM

Usage:
    python clean_movie_metadata.py input_metadata.csv -o cleaned_metadata.csv
"""

import pandas as pd
import numpy as np
import argparse
import re
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def extract_year_from_date(date_str):
    """Extract year from TMDB release date format (YYYY-MM-DD)."""
    if pd.isna(date_str) or date_str == '' or date_str == 'Unknown':
        return None
    try:
        return int(str(date_str)[:4])
    except (ValueError, TypeError):
        return None

def create_runtime_buckets(runtime):
    """Create runtime buckets for categorical modeling."""
    if pd.isna(runtime) or runtime <= 0:
        return 'Unknown'
    elif runtime < 90:
        return '<90min'
    elif runtime <= 120:
        return '90-120min'
    elif runtime <= 150:
        return '120-150min'
    else:
        return '>150min'

def create_budget_buckets(budget):
    """Create budget buckets based on percentiles."""
    if pd.isna(budget) or budget <= 0:
        return 'Unknown'
    elif budget < 5000000:  # < 5M
        return 'Low'
    elif budget < 50000000:  # 5M - 50M
        return 'Medium'
    else:  # > 50M
        return 'High'

def create_revenue_buckets(revenue):
    """Create revenue buckets based on percentiles."""
    if pd.isna(revenue) or revenue <= 0:
        return 'Unknown'
    elif revenue < 10000000:  # < 10M
        return 'Low'
    elif revenue < 100000000:  # 10M - 100M
        return 'Medium'
    else:  # > 100M
        return 'High'

def create_year_buckets(year):
    """Create year buckets for categorical modeling."""
    if pd.isna(year):
        return 'Unknown'
    
    try:
        year_int = int(float(year))
        if year_int < 1930:
            return 'Pre-1930'
        elif year_int <= 1959:
            return '1930-1959'
        elif year_int <= 1979:
            return '1960-1979'
        elif year_int <= 1999:
            return '1980-1999'
        elif year_int <= 2019:
            return '2000-2019'
        else:
            return '2020-Present'
    except (ValueError, TypeError):
        return 'Unknown'

def analyze_and_clean_people(df, column_name, min_appearances=5):
    """
    Analyze directors/cast and keep only those with min_appearances or more.
    Move others to a '{column_name}_other' category.
    
    Returns the cleaned dataframe and analysis statistics.
    """
    print(f"\n📊 Analyzing {column_name}...")
    
    # Extract all unique people from the column
    all_people = []
    for entry in df[column_name].fillna(''):
        if entry and entry != '':
            people = [person.strip() for person in entry.split(';')]
            all_people.extend(people)
    
    # Count appearances
    people_counts = Counter(all_people)
    
    # Find frequent people (appearing in min_appearances or more films)
    frequent_people = {person for person, count in people_counts.items() 
                      if count >= min_appearances and person != ''}
    
    print(f"Total unique {column_name}: {len(people_counts)}")
    print(f"Frequent {column_name} (≥{min_appearances} films): {len(frequent_people)}")
    print(f"Moving to 'other' category: {len(people_counts) - len(frequent_people)}")
    
    # Show top 10 most frequent
    top_people = people_counts.most_common(10)
    print(f"\nTop 10 most frequent {column_name}:")
    for person, count in top_people:
        if person:  # Skip empty strings
            print(f"  {person}: {count} films")
    
    def clean_people_entry(entry):
        """Clean a single entry in the people column."""
        if pd.isna(entry) or entry == '':
            return ''
        
        people = [person.strip() for person in entry.split(';')]
        frequent = [person for person in people if person in frequent_people]
        others = [person for person in people if person not in frequent_people and person != '']
        
        result = []
        result.extend(frequent)
        if others:
            result.append(f"{column_name}_other")
        
        return '; '.join(result)
    
    # Apply cleaning
    df[column_name] = df[column_name].apply(clean_people_entry)
    
    return df, {
        'total_unique': len(people_counts),
        'frequent_count': len(frequent_people),
        'moved_to_other': len(people_counts) - len(frequent_people),
        'top_10': top_people
    }

def clean_metadata(input_file, output_file, reviews_file=None, min_movie_ratings=5, min_director_appearances=3, min_cast_appearances=5, min_production_company_appearances=5):
    """
    Main function to clean movie metadata for LightFM.
    """
    print("🎬 Movie Metadata Cleaner for LightFM")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    if reviews_file:
        print(f"Reviews file: {reviews_file}")
    print("=" * 50)
    
    # Read the data
    print("📖 Loading metadata...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} movies")
    
    # Remove movies not found in TMDB
    initial_count = len(df)
    df = df[df['found_in_tmdb'] == True].copy()
    removed_not_found = initial_count - len(df)
    print(f"✂️  Removed {removed_not_found} movies not found in TMDB")
    print(f"📊 Remaining movies: {len(df)}")
    
    if len(df) == 0:
        print("❌ No movies remaining after filtering!")
        return
    
    # Filter movies by minimum ratings (popularity filter)
    if reviews_file:
        print(f"\n🎯 Filtering movies with less than {min_movie_ratings} ratings...")
        
        # Load reviews data
        reviews_df = pd.read_csv(reviews_file)
        print(f"Loaded {len(reviews_df)} reviews")
        
        # Count ratings per movie (by tmdb_id)
        movie_rating_counts = reviews_df.groupby('tmdb_id').size().reset_index(name='rating_count')
        
        # Filter to movies with at least min_movie_ratings
        popular_movies = movie_rating_counts[movie_rating_counts['rating_count'] >= min_movie_ratings]
        
        print(f"📊 Movie popularity statistics:")
        print(f"  Total movies in reviews: {len(movie_rating_counts)}")
        print(f"  Movies with ≥{min_movie_ratings} ratings: {len(popular_movies)}")
        print(f"  Movies with <{min_movie_ratings} ratings: {len(movie_rating_counts) - len(popular_movies)}")
        
        # Filter metadata to only include popular movies
        before_filter = len(df)
        df = df[df['tmdb_id'].isin(popular_movies['tmdb_id'])].copy()
        after_filter = len(df)
        
        print(f"✂️  Removed {before_filter - after_filter} movies with <{min_movie_ratings} ratings")
        print(f"📊 Remaining movies: {after_filter}")
        
        if len(df) == 0:
            print("❌ No movies remaining after popularity filtering!")
            return
    
    # Fix missing input_year using release_date
    print("\n🔧 Fixing missing years...")
    missing_year_mask = (df['input_year'].isna()) | (df['input_year'] == 'Unknown')
    missing_year_count = missing_year_mask.sum()
    
    if missing_year_count > 0:
        print(f"Found {missing_year_count} movies with missing input_year")
        
        # Extract year from release_date
        df.loc[missing_year_mask, 'release_year'] = df.loc[missing_year_mask, 'release_date'].apply(extract_year_from_date)
        df.loc[missing_year_mask, 'input_year'] = df.loc[missing_year_mask, 'release_year']
        
        fixed_count = (~df.loc[missing_year_mask, 'input_year'].isna()).sum()
        print(f"✅ Fixed {fixed_count} missing years from release_date")
    
    # Create release_year column for all movies
    df['release_year'] = df['release_date'].apply(extract_year_from_date)
    
    # Create runtime buckets
    print("\n⏱️  Creating runtime buckets...")
    df['runtime_bucket'] = df['runtime'].apply(create_runtime_buckets)
    runtime_dist = df['runtime_bucket'].value_counts()
    print("Runtime distribution:")
    for bucket, count in runtime_dist.items():
        print(f"  {bucket}: {count} movies")
    
    # Create budget buckets
    print("\n💰 Creating budget buckets...")
    df['budget_bucket'] = df['budget'].apply(create_budget_buckets)
    budget_dist = df['budget_bucket'].value_counts()
    print("Budget distribution:")
    for bucket, count in budget_dist.items():
        print(f"  {bucket}: {count} movies")
    
    # Create revenue buckets
    print("\n💵 Creating revenue buckets...")
    df['revenue_bucket'] = df['revenue'].apply(create_revenue_buckets)
    revenue_dist = df['revenue_bucket'].value_counts()
    print("Revenue distribution:")
    for bucket, count in revenue_dist.items():
        print(f"  {bucket}: {count} movies")
    
    # Create year buckets
    print("\n📅 Creating year buckets...")
    df['year_bucket'] = df['release_year'].apply(create_year_buckets)
    year_dist = df['year_bucket'].value_counts()
    print("Year distribution:")
    for bucket, count in year_dist.items():
        print(f"  {bucket}: {count} movies")
    
    # Clean directors
    df, director_stats = analyze_and_clean_people(df, 'directors', min_director_appearances)
    
    # Clean cast
    df, cast_stats = analyze_and_clean_people(df, 'cast', min_cast_appearances)
    
    # Clean production companies
    df, prod_company_stats = analyze_and_clean_people(df, 'production_companies', min_production_company_appearances)
    
    # Select and reorder columns for LightFM
    output_columns = [
        'tmdb_id',
        'title',
        'input_year',
        'release_year',
        'year_bucket',
        'runtime_bucket',
        'budget_bucket', 
        'revenue_bucket',
        'genres',
        'directors',
        'cast',
        'keywords',
        'production_companies',
        'original_language'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    df_clean = df[available_columns].copy()
    
    # Fill NaN values appropriately
    string_columns = ['genres', 'directors', 'cast', 'keywords', 'production_companies']
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('')
    
    if 'original_language' in df_clean.columns:
        df_clean['original_language'] = df_clean['original_language'].fillna('unknown')
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned metadata...")
    df_clean.to_csv(output_file, index=False)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 CLEANING COMPLETE!")
    print("=" * 50)
    print(f"📊 Final dataset: {len(df_clean)} movies")
    print(f"📁 Saved to: {output_file}")
    
    print(f"\n📈 Data Distribution:")
    print(f"  Year buckets: {df_clean['year_bucket'].nunique()} categories")
    print(f"  Runtime buckets: {df_clean['runtime_bucket'].nunique()} categories")
    print(f"  Budget buckets: {df_clean['budget_bucket'].nunique()} categories") 
    print(f"  Revenue buckets: {df_clean['revenue_bucket'].nunique()} categories")
    print(f"  Languages: {df_clean['original_language'].nunique()} languages")
    
    if 'genres' in df_clean.columns:
        # Count total unique genres
        all_genres = []
        for entry in df_clean['genres'].fillna(''):
            if entry:
                genres = [g.strip() for g in entry.split(';')]
                all_genres.extend(genres)
        unique_genres = len(set(all_genres))
        print(f"  Genres: {unique_genres} unique genres")
    
    print(f"\n👥 People Statistics:")
    print(f"  Frequent directors: {director_stats['frequent_count']}")
    print(f"  Frequent cast members: {cast_stats['frequent_count']}")
    print(f"  Frequent production companies: {prod_company_stats['frequent_count']}")
    
    print(f"\n🔧 Prepared for LightFM with categorical features:")
    print(f"  - Movie features: year, runtime, budget, revenue, language, genres, directors, cast, keywords, production companies")
    print(f"  - Ready for feature engineering and model training!")

def main():
    parser = argparse.ArgumentParser(description="Clean movie metadata for LightFM recommendation model")
    parser.add_argument("input_file", help="Input metadata CSV file")
    parser.add_argument("-o", "--output", required=True, help="Output cleaned CSV file")
    parser.add_argument("-r", "--reviews", help="Reviews CSV file for popularity filtering")
    parser.add_argument("--min-movie-ratings", type=int, default=3,
                       help="Minimum ratings per movie to be kept (default: 3)")
    parser.add_argument("--min-director-appearances", type=int, default=3,
                       help="Minimum appearances for director to be kept (default: 3)")
    parser.add_argument("--min-cast-appearances", type=int, default=5,
                       help="Minimum appearances for cast member to be kept (default: 5)")
    parser.add_argument("--min-production-company-appearances", type=int, default=5,
                       help="Minimum appearances for production company to be kept (default: 5)")
    
    args = parser.parse_args()
    
    clean_metadata(
        args.input_file, 
        args.output,
        args.reviews,
        args.min_movie_ratings,
        args.min_director_appearances,
        args.min_cast_appearances,
        args.min_production_company_appearances
    )

if __name__ == "__main__":
    main()

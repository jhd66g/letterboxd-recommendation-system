#!/usr/bin/env python3

"""
TMDB Movie Metadata Scraper

This script takes a CSV file from the Letterboxd batch scraper (containing username, title, year, rating)
and creates a separate CSV file with movie metadata from TMDB API including:
- Genres
- Keywords  
- Director(s)
- Cast
- TMDB ID
- Release date
- Runtime
- Production companies
"""

import time
import json
import requests
import pandas as pd
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from datetime import datetime

# ====== CONFIGURATION ======
# You'll need to get your own TMDB Bearer Token from https://www.themoviedb.org/settings/api
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MjViZDI1OGY1ZjQ0ZjAyM2NjNjRjMTg5NDNmZjI4YyIsIm5iZiI6MTcyMzA2NDU0Mi4zNjQ0MDUsInN1YiI6IjY2N2RkMmU5NTljYTUwNzA1OTMyNjQ3ZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.ubYs4zAhZ97A1YtnsR-Ku4oIu1NMokbaVZvJ77lXZQA"

MAX_WORKERS = 10     # Number of concurrent API requests
RETRY_DELAY = 1      # Seconds to wait on rate limit
REQUEST_DELAY = 0.1  # Delay between requests to be respectful

# ====== HELPERS ======
def make_request(session, method, url, **kwargs):
    """Make a request with automatic retry on rate limiting."""
    while True:
        r = session.request(method, url, **kwargs)
        if r.status_code == 429:
            print(f"[429] Rate limit for {url!r}, retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            continue
        elif r.status_code == 404:
            # Movie not found, return None
            return None
        elif r.status_code != 200:
            print(f"[ERROR] Status {r.status_code} for {url}")
            return None
        
        # Small delay to be respectful to the API
        time.sleep(REQUEST_DELAY)
        return r

def extract_unique_movies(csv_file, min_reviews=1):
    """
    Extract unique movies from the Letterboxd CSV file.
    Returns a list of (title, year) tuples, filtering out movies with fewer than min_reviews.
    """
    print(f"Reading movies from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    if 'title' not in df.columns:
        print("Error: CSV file must have a 'title' column")
        return []
    
    # Handle NaN years by converting to string
    df['year'] = df['year'].fillna('Unknown')
    
    # Count reviews per movie (title, year combination)
    movie_counts = df.groupby(['title', 'year']).size().reset_index(name='review_count')
    
    # Filter movies with at least min_reviews
    filtered_movies = movie_counts[movie_counts['review_count'] >= min_reviews]
    
    print(f"Found {len(df)} total movie entries")
    print(f"Found {len(movie_counts)} unique movies")
    print(f"Found {len(filtered_movies)} unique movies with ≥{min_reviews} reviews")
    
    if min_reviews > 1:
        skipped = len(movie_counts) - len(filtered_movies)
        print(f"Skipped {skipped} movies with <{min_reviews} reviews (saves {skipped} TMDB API calls)")
    
    # Convert to list of tuples
    movies = [(row['title'], row['year']) for _, row in filtered_movies.iterrows()]
    
    return movies

def search_movie_tmdb(title, year, session):
    """
    Search for a movie on TMDB and return the best match.
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    # First try searching with year if available
    if year and year != 'Unknown':
        try:
            year_int = int(float(year))
            params = {
                "query": title,
                "year": year_int,
                "include_adult": False
            }
        except (ValueError, TypeError):
            params = {
                "query": title,
                "include_adult": False
            }
    else:
        params = {
            "query": title,
            "include_adult": False
        }
    
    url = "https://api.themoviedb.org/3/search/movie"
    r = make_request(session, "GET", url, headers=headers, params=params)
    
    if not r:
        return None
    
    data = r.json()
    results = data.get("results", [])
    
    if not results:
        # Try searching without year if we had one
        if year and year != 'Unknown':
            params = {
                "query": title,
                "include_adult": False
            }
            r = make_request(session, "GET", url, headers=headers, params=params)
            if r:
                data = r.json()
                results = data.get("results", [])
    
    if not results:
        return None
    
    # Return the first (most relevant) result
    return results[0]

def fetch_movie_details(movie_data, session):
    """
    Fetch detailed movie information from TMDB.
    """
    movie_id = movie_data["id"]
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    # Get detailed info including credits and keywords
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "append_to_response": "credits,keywords"
    }
    
    r = make_request(session, "GET", url, headers=headers, params=params)
    
    if not r:
        return None
    
    data = r.json()
    
    # Extract genres
    genres = [g["name"] for g in data.get("genres", [])]
    
    # Extract keywords
    keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]
    
    # Extract directors from crew
    crew = data.get("credits", {}).get("crew", [])
    directors = [person["name"] for person in crew if person.get("job") == "Director"]
    
    # Extract main cast (top 10)
    cast = data.get("credits", {}).get("cast", [])
    main_cast = [actor["name"] for actor in cast[:10]]
    
    # Extract production companies
    prod_companies = [company["name"] for company in data.get("production_companies", [])]
    
    # Compile all metadata
    metadata = {
        "tmdb_id": movie_id,
        "title": data.get("title"),
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "genres": "; ".join(genres),
        "keywords": "; ".join(keywords),
        "directors": "; ".join(directors),
        "cast": "; ".join(main_cast),
        "production_companies": "; ".join(prod_companies),
        "original_language": data.get("original_language"),
        "budget": data.get("budget"),
        "revenue": data.get("revenue")
    }
    
    return metadata

def process_movie(movie_tuple, session):
    """
    Process a single movie: search TMDB and get detailed metadata.
    """
    title, year = movie_tuple
    
    try:
        # Search for the movie
        search_result = search_movie_tmdb(title, year, session)
        
        if not search_result:
            # Return basic info if not found
            return {
                "input_title": title,
                "input_year": year,
                "tmdb_id": None,
                "title": title,
                "release_date": year if year != 'Unknown' else None,
                "genres": "",
                "keywords": "",
                "directors": "",
                "cast": "",
                "production_companies": "",
                "original_language": None,
                "budget": None,
                "revenue": None,
                "found_in_tmdb": False
            }
        
        # Get detailed metadata
        details = fetch_movie_details(search_result, session)
        
        if not details:
            # Fallback to search result data
            details = {
                "tmdb_id": search_result.get("id"),
                "title": search_result.get("title"),
                "release_date": search_result.get("release_date"),
                "runtime": None,
                "genres": "",
                "keywords": "",
                "directors": "",
                "cast": "",
                "production_companies": "",
                "original_language": search_result.get("original_language"),
                "budget": None,
                "revenue": None
            }
        
        # Add input info and found flag
        details["input_title"] = title
        details["input_year"] = year
        details["found_in_tmdb"] = True
        
        return details
        
    except Exception as e:
        print(f"Error processing movie '{title}' ({year}): {e}")
        return {
            "input_title": title,
            "input_year": year,
            "tmdb_id": None,
            "title": title,
            "release_date": year if year != 'Unknown' else None,
            "runtime": None,
            "genres": "",
            "keywords": "",
            "directors": "",
            "cast": "",
            "production_companies": "",
            "original_language": None,
            "budget": None,
            "revenue": None,
            "found_in_tmdb": False
        }

def scrape_movie_metadata(movies, max_workers=MAX_WORKERS):
    """
    Scrape metadata for all movies using concurrent requests.
    """
    print(f"Fetching metadata for {len(movies)} movies using {max_workers} workers...")
    
    metadata_list = []
    
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {BEARER_TOKEN}"})
        
        # Test API connection
        test_url = "https://api.themoviedb.org/3/configuration"
        test_r = make_request(session, "GET", test_url, headers={"Accept": "application/json"})
        if not test_r:
            print("Error: Could not connect to TMDB API. Check your bearer token.")
            return []
        
        print("✅ TMDB API connection successful")
        
        # Process movies with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_movie = {
                executor.submit(process_movie, movie, session): movie 
                for movie in movies
            }
            
            # Process results as they complete
            with tqdm(total=len(movies), desc="Processing movies") as pbar:
                for future in as_completed(future_to_movie):
                    movie = future_to_movie[future]
                    try:
                        metadata = future.result()
                        metadata_list.append(metadata)
                    except Exception as e:
                        title, year = movie
                        print(f"Error processing {title} ({year}): {e}")
                    finally:
                        pbar.update(1)
    
    return metadata_list

def save_metadata_to_csv(metadata_list, output_file):
    """
    Save metadata to CSV file.
    """
    if not metadata_list:
        print("No metadata to save.")
        return
    
    df = pd.DataFrame(metadata_list)
    
    # Reorder columns for better readability
    column_order = [
        "input_title", "input_year", "found_in_tmdb", "tmdb_id", 
        "title", "release_date", "runtime",
        "genres", "directors", "cast", "keywords",
        "production_companies", "original_language", "budget", "revenue"
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    found_count = df['found_in_tmdb'].sum() if 'found_in_tmdb' in df.columns else len(df)
    not_found_count = len(df) - found_count
    
    print(f"\n📊 Metadata Collection Summary:")
    print(f"Total movies processed: {len(df)}")
    print(f"Found in TMDB: {found_count}")
    print(f"Not found in TMDB: {not_found_count}")
    print(f"Success rate: {found_count/len(df)*100:.1f}%")
    
    if 'genres' in df.columns:
        genres_with_data = df[df['genres'].notna() & (df['genres'] != '')].shape[0]
        print(f"Movies with genre data: {genres_with_data}")
    
    if 'directors' in df.columns:
        directors_with_data = df[df['directors'].notna() & (df['directors'] != '')].shape[0]
        print(f"Movies with director data: {directors_with_data}")
    
    print(f"\n💾 Metadata saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract movie metadata from TMDB for Letterboxd data")
    parser.add_argument("input_csv", help="Input CSV file from batch scraper (with title, year columns)")
    parser.add_argument("-o", "--output", help="Output CSV filename for metadata")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS,
                       help=f"Number of concurrent workers (default: {MAX_WORKERS})")
    parser.add_argument("--test", action="store_true", help="Test mode: only process first 10 movies")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        base_name = os.path.splitext(args.input_csv)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"{base_name}_metadata_{timestamp}.csv"
    
    print("🎬 TMDB Movie Metadata Scraper")
    print("=" * 50)
    print(f"Input file: {args.input_csv}")
    print(f"Output file: {args.output}")
    print(f"Workers: {args.workers}")
    if args.test:
        print("⚠️  TEST MODE: Processing only first 10 movies")
    print("=" * 50)
    
    # Extract unique movies from input CSV with minimum review filter
    movies = extract_unique_movies(args.input_csv, min_reviews=4)
    
    if not movies:
        print("No movies found in input file. Exiting.")
        sys.exit(1)
    
    # Limit to first 10 movies in test mode
    if args.test:
        movies = movies[:10]
        print(f"Test mode: processing {len(movies)} movies")
    
    # Scrape metadata
    start_time = time.time()
    metadata = scrape_movie_metadata(movies, max_workers=args.workers)
    end_time = time.time()
    
    if not metadata:
        print("No metadata collected. Exiting.")
        sys.exit(1)
    
    # Save to CSV
    save_metadata_to_csv(metadata, args.output)
    
    execution_time = end_time - start_time
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📈 Throughput: {len(metadata) / execution_time:.2f} movies/second")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlined TMDB Metadata Fetcher

A clean, efficient TMDB metadata fetcher without the bloat.
Features:
- Simple, readable code
- Robust error handling
- Essential metadata only
- Fast parallel processing

Required metadata:
- TMDB ID, release date, cast, director, genre, original language, keywords, streaming services, poster path, budget, revenue

Functions:
    fetch_metadata_for_movies(unique_movies_df) -> pd.DataFrame
"""

import pandas as pd
import requests
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlinedTMDBFetcher:
    def __init__(self, api_key=None, bearer_token=None, max_workers=20):
        """Initialize streamlined TMDB fetcher."""
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        self.bearer_token = bearer_token or os.getenv('TMDB_BEARER_TOKEN')
        self.max_workers = max_workers
        
        if not self.bearer_token and not self.api_key:
            raise ValueError("TMDB API credentials not found. Set TMDB_BEARER_TOKEN or TMDB_API_KEY environment variable.")
        
        self.base_url = "https://api.themoviedb.org/3"
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        if self.bearer_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.bearer_token}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.params.update({'api_key': self.api_key})
        
        self.timeout = (5, 10)
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 20 requests per second
    
    def _rate_limit(self):
        """Simple rate limiting."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def search_movie(self, title, year=None):
        """Search for a movie by title and year."""
        self._rate_limit()
        
        url = f"{self.base_url}/search/movie"
        params = {'query': title}
        if year:
            params['year'] = year
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            if not results:
                return None
            
            # If year specified, try to find exact match
            if year:
                for result in results:
                    release_date = result.get('release_date', '')
                    if release_date and release_date.startswith(str(year)):
                        return result
            
            return results[0]
            
        except Exception as e:
            logger.debug(f"Search failed for '{title}' ({year}): {e}")
            return None
    
    def get_movie_details(self, movie_id):
        """Get detailed movie information."""
        self._rate_limit()
        
        url = f"{self.base_url}/movie/{movie_id}"
        params = {'append_to_response': 'credits,keywords,watch/providers'}
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Details failed for movie ID {movie_id}: {e}")
            return None
    
    def extract_streaming_services(self, details):
        """Extract US streaming services from movie details."""
        try:
            watch_providers = details.get('watch/providers', {})
            if not watch_providers:
                return []
            
            results = watch_providers.get('results', {})
            us_data = results.get('US', {})
            
            if not us_data:
                return []
            
            providers = []
            
            # Get all provider types
            for provider_type in ['flatrate', 'rent', 'buy']:
                if provider_type in us_data:
                    for provider in us_data[provider_type]:
                        if isinstance(provider, dict) and 'provider_name' in provider:
                            providers.append(provider['provider_name'])
            
            # Remove duplicates while preserving order
            return list(dict.fromkeys(providers))
            
        except Exception as e:
            logger.debug(f"Streaming services extraction failed: {e}")
            return []
    
    def fetch_movie_metadata(self, title, year=None):
        """Fetch complete metadata for a single movie."""
        # Search for the movie
        search_result = self.search_movie(title, year)
        if not search_result:
            return None
        
        movie_id = search_result['id']
        
        # Get detailed information
        details = self.get_movie_details(movie_id)
        if not details:
            return None
        
        try:
            # Extract credits
            credits = details.get('credits', {})
            directors = [
                person['name'] for person in credits.get('crew', [])
                if person.get('job') == 'Director'
            ]
            cast = [
                person['name'] for person in credits.get('cast', [])[:10]
                if person.get('name')
            ]
            
            # Extract other data
            genres = [g['name'] for g in details.get('genres', [])]
            keywords_data = details.get('keywords', {})
            keywords = [kw['name'] for kw in keywords_data.get('keywords', [])]
            streaming_services = self.extract_streaming_services(details)
            
            return {
                'tmdb_id': movie_id,
                'tmdb_title': details.get('title', ''),
                'release_date': details.get('release_date', ''),
                'directors': directors,
                'cast': cast,
                'genres': genres,
                'original_language': details.get('original_language', ''),
                'overview': details.get('overview', ''),
                'keywords': keywords,
                'streaming_services': streaming_services,
                'poster_path': details.get('poster_path', ''),
                'budget': details.get('budget', 0),
                'revenue': details.get('revenue', 0),
                'popularity': details.get('popularity', 0)
            }
            
        except Exception as e:
            logger.debug(f"Metadata extraction failed for {title}: {e}")
            return None

def process_movie_batch(fetcher, movies_batch):
    """Process a batch of movies."""
    results = []
    for movie in movies_batch:
        try:
            metadata = fetcher.fetch_movie_metadata(movie['title'], movie['year'])
            if metadata:
                result = {
                    'original_title': movie['original_title'],
                    'original_year': movie['original_year'],
                    **metadata
                }
            else:
                # Missing metadata placeholder
                result = {
                    'original_title': movie['original_title'],
                    'original_year': movie['original_year'],
                    'tmdb_id': None,
                    'tmdb_title': '',
                    'release_date': '',
                    'directors': [],
                    'cast': [],
                    'genres': [],
                    'original_language': '',
                    'overview': '',
                    'keywords': [],
                    'streaming_services': [],
                    'poster_path': '',
                    'budget': 0,
                    'revenue': 0,
                    'popularity': 0
                }
            results.append(result)
        except Exception as e:
            logger.warning(f"Error processing {movie['title']}: {e}")
            # Error placeholder
            results.append({
                'original_title': movie['original_title'],
                'original_year': movie['original_year'],
                'tmdb_id': None,
                'tmdb_title': '',
                'release_date': '',
                'directors': [],
                'cast': [],
                'genres': [],
                'original_language': '',
                'overview': '',
                'keywords': [],
                'streaming_services': [],
                'poster_path': '',
                'budget': 0,
                'revenue': 0,
                'popularity': 0
            })
    return results

def fetch_metadata_for_movies(unique_movies_df, max_workers=20):
    """
    Fetch TMDB metadata for all unique movies.
    
    Args:
        unique_movies_df (pd.DataFrame): DataFrame with 'title' and 'year' columns
        max_workers (int): Number of parallel workers
    
    Returns:
        pd.DataFrame: Movies with TMDB metadata added
    """
    
    start_time = time.time()
    logger.info(f"🚀 Starting streamlined TMDB metadata fetch for {len(unique_movies_df)} movies...")
    logger.info(f"   Using {max_workers} parallel workers")
    
    # Initialize fetcher
    try:
        fetcher = StreamlinedTMDBFetcher(max_workers=max_workers)
    except ValueError as e:
        logger.error(f"❌ {e}")
        return pd.DataFrame()
    
    # Prepare movies list
    movies_to_fetch = []
    for idx, row in unique_movies_df.iterrows():
        movies_to_fetch.append({
            'title': row['title'],
            'year': row['year'] if pd.notna(row['year']) else None,
            'original_title': row['title'],
            'original_year': row['year']
        })
    
    # Split into batches
    batch_size = max(1, len(movies_to_fetch) // max_workers)
    batches = [movies_to_fetch[i:i + batch_size] for i in range(0, len(movies_to_fetch), batch_size)]
    
    logger.info(f"   Processing {len(batches)} batches (avg {batch_size} movies per batch)")
    
    # Process batches in parallel
    all_results = []
    successful_fetches = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_movie_batch, fetcher, batch): batch
            for batch in batches
        }
        
        progress_bar = tqdm(total=len(movies_to_fetch), desc="🎭 Fetching metadata", unit="movies")
        
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                
                batch_successes = sum(1 for result in batch_results if result.get('tmdb_id') is not None)
                successful_fetches += batch_successes
                
                progress_bar.update(len(batch_results))
                
                elapsed_time = time.time() - start_time
                rate = len(all_results) / elapsed_time if elapsed_time > 0 else 0
                progress_bar.set_description(f"🎭 Fetching metadata ({rate:.1f} movies/sec)")
                
            except Exception as e:
                logger.warning(f"Batch processing error: {e}")
                # Add error placeholders for the entire batch
                for movie in batch:
                    all_results.append({
                        'original_title': movie['original_title'],
                        'original_year': movie['original_year'],
                        'tmdb_id': None,
                        'tmdb_title': '',
                        'release_date': '',
                        'directors': [],
                        'cast': [],
                        'genres': [],
                        'original_language': '',
                        'overview': '',
                        'keywords': [],
                        'streaming_services': [],
                        'poster_path': '',
                        'budget': 0,
                        'revenue': 0,
                        'popularity': 0
                    })
                progress_bar.update(len(batch))
        
        progress_bar.close()
    
    # Create results DataFrame
    metadata_df = pd.DataFrame(all_results)
    
    # Merge with original data
    final_df = unique_movies_df.merge(
        metadata_df,
        left_on=['title', 'year'],
        right_on=['original_title', 'original_year'],
        how='left'
    )
    
    # Clean up duplicate columns
    if 'original_title' in final_df.columns:
        final_df = final_df.drop(['original_title', 'original_year'], axis=1)
    
    # Final stats
    elapsed_time = time.time() - start_time
    rate = len(final_df) / elapsed_time
    
    logger.info(f"\n🚀 STREAMLINED TMDB METADATA FETCH COMPLETE")
    logger.info(f"   Total time: {elapsed_time:.1f} seconds")
    logger.info(f"   Rate: {rate:.1f} movies per second")
    logger.info(f"   Total processed: {len(final_df)}")
    logger.info(f"   Successfully fetched: {successful_fetches}")
    logger.info(f"   Success rate: {successful_fetches/len(final_df)*100:.1f}%")
    
    return final_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.csv', '_with_tmdb.csv')
        
        print(f"🚀 Loading movies from {input_file}...")
        unique_movies_df = pd.read_csv(input_file)
        
        print(f"🎭 Fetching TMDB metadata...")
        result_df = fetch_metadata_for_movies(unique_movies_df)
        
        print(f"💾 Saving to {output_file}...")
        result_df.to_csv(output_file, index=False)
        
        print(f"✅ Complete! Processed {len(result_df)} movies.")
    else:
        print("Usage: python streamlined_tmdb_fetcher.py <input_csv> [output_csv]")

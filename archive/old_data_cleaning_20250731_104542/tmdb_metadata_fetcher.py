#!/usr/bin/env python3
"""
TMDB Metadata Fetcher

Fetches movie metadata from TMDB API with high parallelization for speed.
Handles movies with and without years, matches based on title when year is missing.

Required metadata:
- TMDB ID
- Full release date
- Cast
- Director
- Genre
- Original language

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
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TMDBFetcher:
    def __init__(self, api_key=None, bearer_token=None):
        """Initialize TMDB fetcher with API credentials."""
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        self.bearer_token = bearer_token or os.getenv('TMDB_BEARER_TOKEN')
        
        if not self.bearer_token and not self.api_key:
            raise ValueError("TMDB API credentials not found. Set TMDB_BEARER_TOKEN or TMDB_API_KEY environment variable.")
        
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        
        if self.bearer_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.bearer_token}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.params.update({'api_key': self.api_key})
    
    def search_movie(self, title, year=None):
        """Search for a movie by title and optionally year."""
        url = f"{self.base_url}/search/movie"
        params = {'query': title}
        
        if year:
            params['year'] = year
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            if results:
                # If year is specified, try to find exact year match first
                if year:
                    for result in results:
                        release_date = result.get('release_date', '')
                        if release_date and release_date.startswith(str(year)):
                            return result
                
                # Return first result if no exact year match or no year specified
                return results[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"Search failed for '{title}' ({year}): {e}")
            return None
    
    def get_movie_details(self, movie_id):
        """Get detailed movie information including cast and crew."""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {'append_to_response': 'credits'}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get details for movie ID {movie_id}: {e}")
            return None
    
    def extract_directors(self, credits):
        """Extract directors from movie credits."""
        if not credits or 'crew' not in credits:
            return []
        
        directors = [
            person['name'] for person in credits['crew']
            if person.get('job') == 'Director'
        ]
        return directors
    
    def extract_cast(self, credits, max_cast=10):
        """Extract main cast from movie credits."""
        if not credits or 'cast' not in credits:
            return []
        
        cast = [
            person['name'] for person in credits['cast'][:max_cast]
            if person.get('name')
        ]
        return cast
    
    def fetch_movie_metadata(self, title, year=None):
        """
        Fetch complete metadata for a single movie.
        
        Returns dict with: tmdb_id, title, release_date, directors, cast, genres, original_language
        """
        # Search for the movie
        search_result = self.search_movie(title, year)
        if not search_result:
            return None
        
        movie_id = search_result['id']
        
        # Get detailed information
        details = self.get_movie_details(movie_id)
        if not details:
            return None
        
        # Extract information
        directors = self.extract_directors(details.get('credits', {}))
        cast = self.extract_cast(details.get('credits', {}))
        genres = [g['name'] for g in details.get('genres', [])]
        
        return {
            'tmdb_id': movie_id,
            'tmdb_title': details.get('title', ''),
            'release_date': details.get('release_date', ''),
            'directors': directors,
            'cast': cast,
            'genres': genres,
            'original_language': details.get('original_language', ''),
            'overview': details.get('overview', ''),
            'vote_average': details.get('vote_average', 0),
            'vote_count': details.get('vote_count', 0),
            'popularity': details.get('popularity', 0)
        }

def fetch_metadata_for_movies(unique_movies_df, max_workers=20):
    """
    Fetch TMDB metadata for all unique movies with high parallelization.
    
    Args:
        unique_movies_df (pd.DataFrame): DataFrame with 'title' and 'year' columns
        max_workers (int): Number of parallel threads for API calls
    
    Returns:
        pd.DataFrame: Movies with TMDB metadata added
    """
    
    logger.info(f"🎭 Starting TMDB metadata fetch for {len(unique_movies_df)} movies...")
    logger.info(f"   Using {max_workers} parallel workers")
    
    # Initialize TMDB fetcher
    try:
        fetcher = TMDBFetcher()
    except ValueError as e:
        logger.error(f"❌ {e}")
        return pd.DataFrame()
    
    # Prepare list of movies to fetch
    movies_to_fetch = []
    for idx, row in unique_movies_df.iterrows():
        movies_to_fetch.append({
            'index': idx,
            'title': row['title'],
            'year': row['year'] if pd.notna(row['year']) else None,
            'original_title': row['title'],
            'original_year': row['year']
        })
    
    # Fetch metadata in parallel
    results = []
    successful_fetches = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_movie = {
            executor.submit(fetcher.fetch_movie_metadata, movie['title'], movie['year']): movie
            for movie in movies_to_fetch
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_movie), total=len(movies_to_fetch), desc="Fetching metadata"):
            movie = future_to_movie[future]
            
            try:
                metadata = future.result()
                if metadata:
                    # Combine original movie info with metadata
                    result = {
                        'original_title': movie['original_title'],
                        'original_year': movie['original_year'],
                        **metadata
                    }
                    results.append(result)
                    successful_fetches += 1
                else:
                    # Add entry with missing metadata
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
                        'vote_average': 0,
                        'vote_count': 0,
                        'popularity': 0
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {movie['title']}: {e}")
                # Add entry with error
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
                    'vote_average': 0,
                    'vote_count': 0,
                    'popularity': 0
                })
    
    # Create results DataFrame
    metadata_df = pd.DataFrame(results)
    
    # Merge with original unique movies data
    final_df = unique_movies_df.merge(
        metadata_df,
        left_on=['title', 'year'],
        right_on=['original_title', 'original_year'],
        how='left'
    )
    
    # Clean up duplicate columns
    if 'original_title' in final_df.columns:
        final_df = final_df.drop(['original_title', 'original_year'], axis=1)
    
    logger.info(f"\\n✅ TMDB METADATA FETCH COMPLETE")
    logger.info(f"   Total movies processed: {len(final_df)}")
    logger.info(f"   Successfully fetched: {successful_fetches}")
    logger.info(f"   Missing metadata: {len(final_df) - successful_fetches}")
    logger.info(f"   Success rate: {successful_fetches/len(final_df)*100:.1f}%")
    
    return final_df

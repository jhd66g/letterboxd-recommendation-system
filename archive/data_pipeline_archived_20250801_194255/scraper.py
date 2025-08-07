#!/usr/bin/env python3
"""
Ultra-Fast Parallel Review Scraper

Optimized for maximum speed without compromising data volume.
Uses concurrent processing and optimized HTTP requests.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

class FastScraper:
    def __init__(self, max_workers=20):
        self.max_workers = max_workers
        self.session_local = threading.local()
        
    def get_session(self):
        """Get thread-local session with optimized settings."""
        if not hasattr(self.session_local, 'session'):
            session = requests.Session()
            session.headers.update(HEADERS)
            
            # Optimize connection pooling
            adapter = HTTPAdapter(
                pool_connections=100,
                pool_maxsize=100,
                max_retries=Retry(
                    total=3,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504]
                )
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            self.session_local.session = session
        
        return self.session_local.session

    def scrape_user_films(self, username, explicit_only=True):
        """Scrape ALL films for a user with optimized requests."""
        session = self.get_session()
        films = []
        url = f"https://letterboxd.com/{username}/films/"
        
        while url:
            try:
                resp = session.get(url, timeout=5)  # Reduced timeout
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                container = soup.find("ul", class_="poster-list")
                if not container:
                    break

                for block in container.find_all("li", class_="poster-container"):
                    poster_div = block.find("div", class_="film-poster")
                    if not poster_div:
                        continue
                    
                    title_elem = poster_div.find("img")
                    title = title_elem.get("alt") if title_elem else None
                    
                    # Extract year
                    year = None
                    if poster_div.get("data-film-year"):
                        year = int(poster_div.get("data-film-year"))
                    elif poster_div.get("data-film-slug"):
                        slug = poster_div.get("data-film-slug")
                        year_match = re.search(r'-(\d{4})$', slug)
                        if year_match:
                            year = int(year_match.group(1))
                        else:
                            year_matches = re.findall(r'\\b(19\\d{2}|20\\d{2})\\b', slug)
                            if year_matches:
                                year = int(year_matches[-1])
                    
                    # Extract rating - optimized
                    rating = None
                    rating_elem = block.find("span", class_="rating")
                    if rating_elem:
                        for cls in rating_elem.get("class", []):
                            if cls.startswith("rated-"):
                                rating = int(cls.split("-", 1)[1])
                                break
                    
                    # Only add if we have title and (if explicit_only=True) a rating
                    if title and (not explicit_only or rating is not None):
                        films.append({
                            "username": username,
                            "title": title,
                            "year": year,
                            "rating": rating,
                            "film_slug": poster_div.get("data-film-slug", "")
                        })

                # Check for next page
                next_link = soup.select_one("a.next")
                if next_link and next_link.get("href"):
                    url = "https://letterboxd.com" + next_link["href"]
                else:
                    url = None

            except Exception as e:
                logger.warning(f"Error scraping {username}: {e}")
                break

        return films

    def process_user(self, username, explicit_only=True):
        """Process a single user and return results."""
        try:
            films = self.scrape_user_films(username, explicit_only)
            return username, films, None
        except Exception as e:
            return username, [], str(e)

def scrape_reviews_ultra_fast(users_df, explicit_only=True, max_workers=20):
    """
    Ultra-fast parallel scraping of reviews for all users.
    
    Args:
        users_df (pd.DataFrame): DataFrame with 'username' column
        explicit_only (bool): If True, only collect reviews with explicit ratings
        max_workers (int): Number of parallel workers
    
    Returns:
        pd.DataFrame: Reviews with columns ['username', 'title', 'year', 'rating', 'film_slug']
    """
    
    logger.info(f"🚀 Ultra-fast parallel scraping for {len(users_df)} users (workers={max_workers}, explicit_only={explicit_only})...")
    
    scraper = FastScraper(max_workers=max_workers)
    all_reviews = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_user = {
            executor.submit(scraper.process_user, row['username'], explicit_only): row['username'] 
            for _, row in users_df.iterrows()
        }
        
        # Process results with progress bar
        completed = 0
        total = len(users_df)
        
        with tqdm(total=total, desc="Scraping users") as pbar:
            for future in as_completed(future_to_user):
                username = future_to_user[future]
                try:
                    username, films, error = future.result()
                    if error:
                        logger.warning(f"   ⚠️  Error with {username}: {error}")
                    else:
                        all_reviews.extend(films)
                        logger.info(f"   📖 {username}: {len(films)} reviews")
                    
                except Exception as exc:
                    logger.error(f"   ❌ {username} generated an exception: {exc}")
                
                completed += 1
                pbar.update(1)
                pbar.set_postfix({"Reviews": len(all_reviews), "Avg/user": len(all_reviews)/completed if completed > 0 else 0})
    
    # Create DataFrame
    df = pd.DataFrame(all_reviews)
    
    logger.info(f"\\n🎉 ULTRA-FAST SCRAPING COMPLETE")
    logger.info(f"   Total reviews: {len(df)}")
    logger.info(f"   Unique users: {df['username'].nunique()}")
    logger.info(f"   Unique movies: {df['title'].nunique()}")
    if 'rating' in df.columns:
        logger.info(f"   Reviews with ratings: {df['rating'].notna().sum()}")
    
    return df

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Ultra-fast Letterboxd review scraper")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file with users")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file for reviews")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--explicit-only", action="store_true", help="Only collect explicit ratings")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load users from CSV
        print(f"🚀 Loading users from {args.input}...")
        users_df = pd.read_csv(args.input)
        
        if 'username' not in users_df.columns:
            print("❌ Error: Input CSV must have 'username' column")
            sys.exit(1)
        
        print(f"   Found {len(users_df)} users")
        
        # Scrape reviews
        print(f"🎬 Starting review scraping...")
        print(f"   Workers: {args.workers}")
        print(f"   Explicit only: {args.explicit_only}")
        
        start_time = time.time()
        result = scrape_reviews_ultra_fast(
            users_df, 
            explicit_only=args.explicit_only, 
            max_workers=args.workers
        )
        end_time = time.time()
        
        # Save results
        print(f"💾 Saving results to {args.output}...")
        result.to_csv(args.output, index=False)
        
        # Summary
        print(f"\\n✅ Scraping completed successfully!")
        print(f"   Time taken: {end_time - start_time:.1f} seconds")
        print(f"   Reviews collected: {len(result):,}")
        print(f"   Unique users: {result['username'].nunique():,}")
        print(f"   Unique movies: {result['title'].nunique():,}")
        print(f"   Average reviews per user: {len(result) / len(users_df):.1f}")
        if 'rating' in result.columns:
            print(f"   Reviews with ratings: {result['rating'].notna().sum():,}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

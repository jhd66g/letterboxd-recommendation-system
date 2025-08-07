#!/usr/bin/env python3
"""
Fast Review Scraper - Based on rating_scraper_pro.py

Efficiently scrapes ALL reviews from Letterboxd user profiles with progress tracking.
Only collects reviews with explicit ratings.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def scrape_user_films(username, session, explicit_only=True):
    """
    Scrape ALL films for a user from Letterboxd using proper pagination.
    Only includes films with explicit ratings if explicit_only=True.
    """
    films = []
    url = f"https://letterboxd.com/{username}/films/"
    
    while url:
        try:
            resp = session.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            container = soup.find("ul", class_="poster-list")
            if not container:
                break

            for block in container.find_all("li", class_="poster-container"):
                # Extract title from the poster div
                poster_div = block.find("div", class_="film-poster")
                if not poster_div:
                    continue
                
                title_elem = poster_div.find("img")
                title = title_elem.get("alt") if title_elem else None
                
                # Extract year from the poster div's data or from the film slug
                year = None
                if poster_div.get("data-film-year"):
                    year = int(poster_div.get("data-film-year"))
                elif poster_div.get("data-film-slug"):
                    slug = poster_div.get("data-film-slug")
                    year_match = re.search(r'-(\d{4})$', slug)
                    if year_match:
                        year = int(year_match.group(1))
                    else:
                        # If no year at end, check for any 4-digit number that looks like a year
                        year_matches = re.findall(r'\\b(19\\d{2}|20\\d{2})\\b', slug)
                        if year_matches:
                            year = int(year_matches[-1])  # Take the last one if multiple found
                
                # Extract rating
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

        except requests.RequestException as e:
            logger.warning(f"Error fetching page {url}: {e}")
            break
        except Exception as e:
            logger.warning(f"Error parsing page {url}: {e}")
            break

    return films

def scrape_reviews_for_users_fast(users_df, explicit_only=True):
    """
    Fast scraping of reviews for all users in the DataFrame using the efficient method.
    
    Args:
        users_df (pd.DataFrame): DataFrame with 'username' column
        explicit_only (bool): If True, only collect reviews with explicit ratings
    
    Returns:
        pd.DataFrame: Reviews with columns ['username', 'title', 'year', 'rating', 'film_slug']
    """
    
    logger.info(f"📚 Fast-scraping reviews for {len(users_df)} users (explicit_only={explicit_only})...")
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    all_reviews = []
    
    # Use tqdm for progress bar
    for idx, row in tqdm(users_df.iterrows(), total=len(users_df), desc="Scraping users"):
        username = row['username']
        
        try:
            # Check if user exists first
            profile_url = f"https://letterboxd.com/{username}/"
            resp = session.get(profile_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Check if profile exists
            body = soup.find("body")
            if body and "error" in body.get("class", []):
                logger.warning(f"   ⚠️  User '{username}' not found, skipping")
                continue
            
            # Scrape all films for this user
            user_reviews = scrape_user_films(username, session, explicit_only)
            all_reviews.extend(user_reviews)
            
            logger.info(f"   📖 {username}: {len(user_reviews)} reviews")
            
            # Small delay to be respectful
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"   ❌ Error scraping {username}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_reviews)
    
    logger.info(f"\\n✅ FAST SCRAPING COMPLETE")
    logger.info(f"   Total reviews: {len(df)}")
    logger.info(f"   Unique users: {df['username'].nunique()}")
    logger.info(f"   Unique movies: {df['title'].nunique()}")
    if 'rating' in df.columns:
        logger.info(f"   Reviews with ratings: {df['rating'].notna().sum()}")
    
    return df

if __name__ == "__main__":
    # Test with a single user
    import argparse
    parser = argparse.ArgumentParser(description="Test fast review scraper")
    parser.add_argument("username", help="Letterboxd username to test")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create test DataFrame
    test_df = pd.DataFrame({'username': [args.username]})
    
    # Test scraping
    result = scrape_reviews_for_users_fast(test_df, explicit_only=True)
    print(f"\\nResult shape: {result.shape}")
    print(result.head())

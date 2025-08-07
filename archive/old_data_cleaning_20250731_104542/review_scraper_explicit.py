#!/usr/bin/env python3
"""
Updated Review Scraper - Explicit Ratings Only

Scrapes reviews from Letterboxd user profiles, only collecting entries with explicit ratings.
If a rating does not exist, the entry is not added to the reviews.

Functions:
    scrape_reviews_for_users(users_df, explicit_only=True) -> pd.DataFrame
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

def extract_rating(review_element):
    """Extract rating from review element (if any)."""
    rating_span = review_element.find('span', class_='rating')
    if rating_span:
        # Count filled stars
        stars = rating_span.find_all('span', class_='star')
        filled_stars = len([s for s in stars if 'filled' in s.get('class', [])])
        half_stars = len([s for s in stars if 'half' in s.get('class', [])])
        return filled_stars + (0.5 * half_stars)
    return None

def scrape_user_reviews(username, session, max_reviews=1000, explicit_only=True):
    """Scrape reviews for a single user, optionally filtering for explicit ratings only."""
    reviews = []
    page = 1
    
    logger.info(f"   📖 Scraping {username}...")
    
    while len(reviews) < max_reviews:
        if page == 1:
            url = f"https://letterboxd.com/{username}/films/"
        else:
            url = f"https://letterboxd.com/{username}/films/page/{page}/"
        
        try:
            response = session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find film entries
            film_items = soup.find_all('li', class_='poster-container')
            
            if not film_items:
                break
            
            for item in film_items:
                if len(reviews) >= max_reviews:
                    break
                
                # Extract film info
                film_link = item.find('div', class_='film-poster')
                if not film_link:
                    continue
                
                film_data = film_link.get('data-film-slug', '')
                if not film_data:
                    continue
                
                # Extract title from alt text
                img = film_link.find('img')
                if img:
                    title = img.get('alt', '').strip()
                else:
                    continue
                
                # Try to extract year from film slug or title
                year_match = re.search(r'-(\d{4})$', film_data)
                if year_match:
                    year = int(year_match.group(1))
                else:
                    # Try to extract from title
                    year_match = re.search(r'\\((\d{4})\\)', title)
                    if year_match:
                        year = int(year_match.group(1))
                        title = re.sub(r'\\s*\\(\\d{4}\\)', '', title).strip()
                    else:
                        year = None
                
                # Extract rating
                rating = extract_rating(item)
                
                # Only add if we have a title AND rating (when explicit_only=True)
                if title and (not explicit_only or rating is not None):
                    reviews.append({
                        'username': username,
                        'title': title,
                        'year': year,
                        'rating': rating,
                        'film_slug': film_data
                    })
            
            page += 1
            time.sleep(random.uniform(0.5, 1.5))  # Random delay
            
        except Exception as e:
            logger.warning(f"      Error on page {page}: {e}")
            break
    
    logger.info(f"      Found {len(reviews)} reviews")
    return reviews

def scrape_reviews_for_users(users_df, explicit_only=True, max_reviews_per_user=1000):
    """
    Scrape reviews for all users in the DataFrame.
    
    Args:
        users_df (pd.DataFrame): DataFrame with 'username' column
        explicit_only (bool): If True, only collect reviews with explicit ratings
        max_reviews_per_user (int): Maximum reviews to collect per user
    
    Returns:
        pd.DataFrame: Reviews with columns ['username', 'title', 'year', 'rating', 'film_slug']
    """
    
    logger.info(f"📚 Scraping reviews for {len(users_df)} users (explicit_only={explicit_only})...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    all_reviews = []
    
    for idx, row in users_df.iterrows():
        username = row['username']
        
        try:
            user_reviews = scrape_user_reviews(
                username, 
                session, 
                max_reviews_per_user, 
                explicit_only
            )
            all_reviews.extend(user_reviews)
            
            # Log progress
            if (idx + 1) % 50 == 0:
                logger.info(f"💾 Progress: {idx+1}/{len(users_df)} users, {len(all_reviews)} total reviews")
            
            # Add delay between users
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            logger.error(f"   ❌ Error scraping {username}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_reviews)
    
    logger.info(f"\\n✅ SCRAPING COMPLETE")
    logger.info(f"   Total reviews: {len(df)}")
    logger.info(f"   Unique users: {df['username'].nunique()}")
    logger.info(f"   Unique movies: {df['title'].nunique()}")
    if 'rating' in df.columns:
        logger.info(f"   Reviews with ratings: {df['rating'].notna().sum()}")
    
    return df

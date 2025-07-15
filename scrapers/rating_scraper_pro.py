#!/usr/local/bin/python3.11

import re
from bs4 import BeautifulSoup
import requests
import pandas as pd
import argparse
import sys

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_user_films(username):
    """
    Scrape user films from Letterboxd and return them as a list of dictionaries.
    """
    films = []
    url = f"https://letterboxd.com/{username}/films/"

    while url:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
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
                    year = poster_div.get("data-film-year")
                elif poster_div.get("data-film-slug"):
                    # Try to extract year from slug - look for 4-digit numbers
                    slug = poster_div.get("data-film-slug")
                    import re
                    year_match = re.search(r'-(\d{4})$', slug)
                    if year_match:
                        year = year_match.group(1)
                    else:
                        # If no year at end, check for any 4-digit number that looks like a year
                        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', slug)
                        if year_matches:
                            year = year_matches[-1]  # Take the last one if multiple found
                
                # Extract rating
                rating = None
                rating_elem = block.find("span", class_="rating")
                if rating_elem:
                    for cls in rating_elem.get("class", []):
                        if cls.startswith("rated-"):
                            rating = int(cls.split("-", 1)[1])
                            break
                
                films.append({
                    "username": username,
                    "title": title,
                    "year": year,
                    "rating": rating
                })

            # Check for next page
            next_link = soup.select_one("a.next")
            if next_link and next_link.get("href"):
                url = "https://letterboxd.com" + next_link["href"]
            else:
                url = None

        except requests.RequestException as e:
            print(f"Error fetching page {url}: {e}")
            break
        except Exception as e:
            print(f"Error parsing page {url}: {e}")
            break

    return films


def get_user_data(username):
    """
    Get user film data and return as a pandas DataFrame.
    """
    print(f"Scraping films for user: {username}")
    
    # Check if user exists by making a request to their profile
    profile_url = f"https://letterboxd.com/{username}/"
    try:
        resp = requests.get(profile_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Check if profile exists
        body = soup.find("body")
        if body and "error" in body.get("class", []):
            print(f"User '{username}' not found")
            return pd.DataFrame()
            
    except requests.RequestException as e:
        print(f"Error checking user profile: {e}")
        return pd.DataFrame()
    
    # Scrape the films
    films = scrape_user_films(username)
    
    if not films:
        print(f"No films found for user: {username}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(films)
    print(f"Successfully scraped {len(df)} films for user: {username}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Letterboxd user films")
    parser.add_argument("username", help="Letterboxd username to scrape")
    parser.add_argument("-o", "--output", help="Output CSV file path (optional)")
    
    args = parser.parse_args()
    
    # Get user data
    df = get_user_data(args.username)
    
    if df.empty:
        print("No data retrieved. Exiting.")
        sys.exit(1)
    
    # Display DataFrame info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save to CSV if output path provided
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nData saved to {args.output}")
    
    # Display summary statistics
    print(f"\nSummary:")
    print(f"Total films: {len(df)}")
    print(f"Films with ratings: {len(df[df['rating'].notna()])}")
    print(f"Films without ratings: {len(df[df['rating'].isna()])}")
    if not df[df['rating'].notna()].empty:
        print(f"Average rating: {df[df['rating'].notna()]['rating'].mean():.2f}")

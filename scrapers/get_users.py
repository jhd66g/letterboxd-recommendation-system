#!/usr/local/bin/python3.11

import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse
import sys
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def get_popular_users(max_pages=128, time_period="this/week"):
    """
    Scrape popular users from Letterboxd.
    
    Args:
        max_pages (int): Maximum number of pages to scrape
        time_period (str): Time period for popularity - options: 
                          "this/week", "this/month", "this/year", "all-time"
    
    Returns:
        list: List of dictionaries containing user information
    """
    base_url = f"https://letterboxd.com/members/popular/{time_period}/page/{{}}/"
    
    all_users = []
    
    pbar = tqdm(range(1, max_pages + 1))
    for page in pbar:
        pbar.set_description(f"Scraping page {page} of {max_pages} of popular users")
        
        try:
            r = requests.get(base_url.format(page), headers=HEADERS, timeout=10)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", attrs={"class": "person-table"})
            
            if not table:
                print(f"Warning: No user table found on page {page}")
                continue
                
            rows = table.findAll("td", attrs={"class": "table-person"})
            
            page_users = []
            for row in rows:
                try:
                    link = row.find("a")["href"]
                    username = link.strip('/')
                    display_name = row.find("a", attrs={"class": "name"}).text.strip()
                    
                    # Extract number of reviews
                    small_elem = row.find("small")
                    if small_elem and small_elem.find("a"):
                        num_reviews_text = small_elem.find("a").text.replace('\xa0', ' ').split()[0].replace(',', '')
                        try:
                            num_reviews = int(num_reviews_text)
                        except ValueError:
                            num_reviews = 0
                    else:
                        num_reviews = 0
                    
                    user = {
                        "username": username,
                        "display_name": display_name,
                        "num_reviews": num_reviews
                    }
                    
                    page_users.append(user)
                    
                except Exception as e:
                    print(f"Error parsing user row on page {page}: {e}")
                    continue
            
            all_users.extend(page_users)
            pbar.set_postfix({"Users found": len(all_users)})
            
            # If we got no users on this page, we might have reached the end
            if not page_users:
                print(f"No users found on page {page}, stopping pagination")
                break
                
        except requests.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            continue
        except Exception as e:
            print(f"Error processing page {page}: {e}")
            continue
    
    return all_users

def save_users_to_csv(users, filename="letterboxd_users.csv"):
    """
    Save users list to CSV file.
    
    Args:
        users (list): List of user dictionaries
        filename (str): Output filename
    """
    df = pd.DataFrame(users)
    df.to_csv(filename, index=False)
    print(f"Saved {len(users)} users to {filename}")
    return df

def get_usernames_list(users):
    """
    Extract just the usernames from the users list.
    
    Args:
        users (list): List of user dictionaries
        
    Returns:
        list: List of usernames
    """
    return [user['username'] for user in users]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape popular Letterboxd users")
    parser.add_argument("-p", "--pages", type=int, default=10, 
                       help="Maximum number of pages to scrape (default: 10)")
    parser.add_argument("-t", "--time-period", default="this/week",
                       choices=["this/week", "this/month", "this/year", "all-time"],
                       help="Time period for popularity (default: this/week)")
    parser.add_argument("-o", "--output", default="letterboxd_users.csv",
                       help="Output CSV filename (default: letterboxd_users.csv)")
    parser.add_argument("--usernames-only", action="store_true",
                       help="Output only usernames to a text file")
    
    args = parser.parse_args()
    
    print(f"Scraping popular users from Letterboxd ({args.time_period})")
    print(f"Maximum pages: {args.pages}")
    print("=" * 50)
    
    # Get users
    users = get_popular_users(max_pages=args.pages, time_period=args.time_period)
    
    if not users:
        print("No users found. Exiting.")
        sys.exit(1)
    
    # Save to CSV
    df = save_users_to_csv(users, args.output)
    
    # Also save usernames only if requested
    if args.usernames_only:
        usernames = get_usernames_list(users)
        usernames_file = args.output.replace('.csv', '_usernames.txt')
        with open(usernames_file, 'w') as f:
            for username in usernames:
                f.write(f"{username}\n")
        print(f"Saved {len(usernames)} usernames to {usernames_file}")
    
    # Display summary statistics
    print(f"\nSummary:")
    print(f"Total users: {len(users)}")
    print(f"Users with reviews: {len([u for u in users if u['num_reviews'] > 0])}")
    if users:
        avg_reviews = sum(u['num_reviews'] for u in users) / len(users)
        max_reviews = max(u['num_reviews'] for u in users)
        min_reviews = min(u['num_reviews'] for u in users)
        print(f"Average reviews per user: {avg_reviews:.1f}")
        print(f"Max reviews: {max_reviews}")
        print(f"Min reviews: {min_reviews}")
    
    print("\nFirst 10 users:")
    print(df.head(10).to_string(index=False))
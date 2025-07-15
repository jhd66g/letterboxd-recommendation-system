#!/usr/local/bin/python3.11

import pandas as pd
import argparse
import sys
import time
from datetime import datetime
import os
from tqdm import tqdm

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datascraping.get_users import get_popular_users, get_usernames_list
from datascraping.rating_scraper_pro import get_user_data as get_user_data_pro
from rating_scraper_parallel import get_user_data as get_user_data_parallel
from rating_scraper_parallel import scrape_multiple_users

def batch_scrape_users(usernames, scraper_type="pro", max_workers=None, chunk_size=50):
    """
    Scrape multiple users and combine results into a single DataFrame.
    
    Args:
        usernames (list): List of usernames to scrape
        scraper_type (str): Type of scraper to use ("pro" or "parallel")
        max_workers (int): Number of workers for parallel processing
        chunk_size (int): Number of users to process in each chunk
    
    Returns:
        pandas.DataFrame: Combined results from all users
    """
    all_results = []
    failed_users = []
    
    print(f"Scraping {len(usernames)} users using {scraper_type} scraper")
    print("=" * 60)
    
    if scraper_type == "parallel" and len(usernames) > 1:
        # Use the parallel batch scraper for multiple users
        try:
            print("Using parallel batch processing...")
            results = scrape_multiple_users(usernames)
            
            for username, df in results.items():
                if not df.empty:
                    all_results.append(df)
                else:
                    failed_users.append(username)
                    
        except Exception as e:
            print(f"Batch processing failed: {e}")
            print("Falling back to sequential processing...")
            return batch_scrape_sequential(usernames, scraper_type)
            
    else:
        # Use sequential processing
        return batch_scrape_sequential(usernames, scraper_type, chunk_size)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    # Report results
    print(f"\nScraping completed:")
    print(f"Successful users: {len(usernames) - len(failed_users)}")
    print(f"Failed users: {len(failed_users)}")
    if failed_users:
        print(f"Failed usernames: {', '.join(failed_users[:10])}" + 
              (f" and {len(failed_users) - 10} more..." if len(failed_users) > 10 else ""))
    
    return combined_df

def batch_scrape_sequential(usernames, scraper_type="pro", chunk_size=50):
    """
    Scrape users sequentially, processing in chunks.
    """
    all_results = []
    failed_users = []
    
    # Choose scraper function
    if scraper_type == "parallel":
        scraper_func = get_user_data_parallel
    else:
        scraper_func = get_user_data_pro
    
    # Process users in chunks
    total_chunks = (len(usernames) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(usernames))
        chunk_usernames = usernames[start_idx:end_idx]
        
        print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_usernames)} users)")
        
        # Process users in current chunk
        pbar = tqdm(chunk_usernames, desc=f"Chunk {chunk_idx + 1}")
        for username in pbar:
            pbar.set_description(f"Scraping {username}")
            
            try:
                df = scraper_func(username)
                if not df.empty:
                    all_results.append(df)
                else:
                    failed_users.append(username)
                    
            except Exception as e:
                print(f"Error scraping {username}: {e}")
                failed_users.append(username)
                continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    # Report results
    print(f"\nScraping completed:")
    print(f"Successful users: {len(usernames) - len(failed_users)}")
    print(f"Failed users: {len(failed_users)}")
    if failed_users:
        print(f"Failed usernames: {', '.join(failed_users[:10])}" + 
              (f" and {len(failed_users) - 10} more..." if len(failed_users) > 10 else ""))
    
    return combined_df

def run_full_pipeline(max_pages=5, time_period="this/week", scraper_type="pro", 
                     output_file=None, save_users=True, max_users=None):
    """
    Run the complete pipeline: get users, then scrape their ratings.
    
    Args:
        max_pages (int): Maximum pages of users to scrape
        time_period (str): Time period for user popularity
        scraper_type (str): Type of scraper to use
        output_file (str): Output CSV filename
        save_users (bool): Whether to save the users list
        max_users (int): Maximum number of users to process
    
    Returns:
        pandas.DataFrame: Combined ratings data
    """
    start_time = time.time()
    
    print("🚀 Starting Letterboxd Batch Scraping Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max pages: {max_pages}")
    print(f"Time period: {time_period}")
    print(f"Scraper type: {scraper_type}")
    if max_users:
        print(f"Max users: {max_users}")
    print("=" * 60)
    
    # Step 1: Get popular users
    print("\n📋 Step 1: Getting popular users...")
    users = get_popular_users(max_pages=max_pages, time_period=time_period)
    
    if not users:
        print("❌ No users found. Exiting.")
        return pd.DataFrame()
    
    # Limit users if requested
    if max_users and len(users) > max_users:
        users = users[:max_users]
        print(f"Limited to first {max_users} users")
    
    usernames = get_usernames_list(users)
    
    print(f"✅ Found {len(users)} users")
    
    # Save users list if requested
    if save_users:
        users_df = pd.DataFrame(users)
        users_filename = f"users_{time_period.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        users_df.to_csv(users_filename, index=False)
        print(f"💾 Saved users list to {users_filename}")
    
    # Step 2: Scrape ratings for all users
    print(f"\n🎬 Step 2: Scraping ratings for {len(usernames)} users...")
    ratings_df = batch_scrape_users(usernames, scraper_type=scraper_type)
    
    if ratings_df.empty:
        print("❌ No ratings data collected. Exiting.")
        return pd.DataFrame()
    
    # Step 3: Save results
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"letterboxd_ratings_{scraper_type}_{time_period.replace('/', '_')}_{timestamp}.csv"
    
    ratings_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Final statistics
    print(f"\n📊 Pipeline completed successfully!")
    print("=" * 60)
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    print(f"Total users processed: {len(usernames)}")
    print(f"Total films collected: {len(ratings_df)}")
    print(f"Films with ratings: {len(ratings_df[ratings_df['rating'].notna()])}")
    print(f"Films without ratings: {len(ratings_df[ratings_df['rating'].isna()])}")
    print(f"Average rating: {ratings_df[ratings_df['rating'].notna()]['rating'].mean():.2f}")
    print(f"Throughput: {len(ratings_df) / execution_time:.2f} films/second")
    print(f"Output saved to: {output_file}")
    
    return ratings_df

def load_usernames_from_file(filename):
    """
    Load usernames from a text file (one username per line).
    
    Args:
        filename (str): Path to file containing usernames
        
    Returns:
        list: List of usernames
    """
    try:
        with open(filename, 'r') as f:
            usernames = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(usernames)} usernames from {filename}")
        return usernames
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scrape Letterboxd users and ratings")
    
    # Mode selection
    parser.add_argument("--mode", choices=["pipeline", "scrape-only"], default="pipeline",
                       help="Mode: 'pipeline' (get users + scrape) or 'scrape-only' (use existing usernames)")
    
    # User collection options
    parser.add_argument("-p", "--pages", type=int, default=5,
                       help="Maximum number of user pages to scrape (default: 5)")
    parser.add_argument("-t", "--time-period", default="this/week",
                       choices=["this/week", "this/month", "this/year", "all-time"],
                       help="Time period for user popularity (default: this/week)")
    parser.add_argument("--max-users", type=int,
                       help="Maximum number of users to process")
    
    # Scraping options
    parser.add_argument("-s", "--scraper", choices=["pro", "parallel"], default="pro",
                       help="Scraper type to use (default: pro)")
    parser.add_argument("--chunk-size", type=int, default=50,
                       help="Users per chunk for sequential processing (default: 50)")
    
    # Input/Output options
    parser.add_argument("-u", "--usernames-file",
                       help="File containing usernames to scrape (one per line)")
    parser.add_argument("-o", "--output",
                       help="Output CSV filename (auto-generated if not provided)")
    parser.add_argument("--no-save-users", action="store_true",
                       help="Don't save the users list")
    
    # Performance options
    parser.add_argument("--max-workers", type=int,
                       help="Maximum number of workers for parallel processing")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        # Run full pipeline
        df = run_full_pipeline(
            max_pages=args.pages,
            time_period=args.time_period,
            scraper_type=args.scraper,
            output_file=args.output,
            save_users=not args.no_save_users,
            max_users=args.max_users
        )
        
    elif args.mode == "scrape-only":
        # Scrape ratings for existing list of usernames
        if not args.usernames_file:
            print("Error: --usernames-file is required for scrape-only mode")
            sys.exit(1)
        
        usernames = load_usernames_from_file(args.usernames_file)
        if not usernames:
            print("No usernames to process. Exiting.")
            sys.exit(1)
        
        # Limit users if requested
        if args.max_users and len(usernames) > args.max_users:
            usernames = usernames[:args.max_users]
            print(f"Limited to first {args.max_users} users")
        
        print(f"Scraping ratings for {len(usernames)} users...")
        start_time = time.time()
        
        df = batch_scrape_users(usernames, scraper_type=args.scraper, max_workers=args.max_workers)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if df.empty:
            print("No data collected. Exiting.")
            sys.exit(1)
        
        # Save results
        if args.output is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f"letterboxd_ratings_{args.scraper}_{timestamp}.csv"
        
        df.to_csv(args.output, index=False)
        
        # Display statistics
        print(f"\nScraping completed!")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Total films: {len(df)}")
        print(f"Films with ratings: {len(df[df['rating'].notna()])}")
        print(f"Throughput: {len(df) / execution_time:.2f} films/second")
        print(f"Output saved to: {args.output}")

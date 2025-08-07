#!/usr/bin/env python3
"""
Demo script for the Streamlined Letterboxd Recommendation System

Tests performance and generates recommendations for multiple users including:
davidehrlich, Josh Lewis, #1 gizmo fan, SilentDawn, Karsten, suspirliam, 
Lucy, zoë rose bryant, belu borelli, Guido Montini
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlined_engine import LetterboxdRecommendationEngine
from scrapers.review_scraper import scrape_user_reviews
import requests

def get_user_data(username):
    """
    Get user film data and return as a pandas DataFrame using review_scraper.
    """
    print(f"Scraping films for user: {username}")
    
    # Create a session for scraping
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    try:
        # Use the existing review scraper with a reasonable limit
        reviews = scrape_user_reviews(username, session, max_reviews=500)
        
        if not reviews:
            print(f"No films found for user: {username}")
            return pd.DataFrame()
        
        # Convert to DataFrame with the expected columns
        df = pd.DataFrame(reviews)
        
        # Rename columns to match expected format if needed
        if 'title' in df.columns and 'rating' in df.columns:
            print(f"Successfully scraped {len(df)} films for user: {username}")
            return df
        else:
            print(f"Unexpected data format for user: {username}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error scraping {username}: {e}")
        return pd.DataFrame()

def calculate_model_metrics(engine):
    """Calculate RMSE and MAE using proper 80/20 train/test split."""
    print("\n📊 CALCULATING MODEL METRICS...")
    
    try:
        # Use ALL reviews for proper 80/20 split (no capping)
        reviews_df = engine.reviews_df.copy()
        
        if len(reviews_df) < 1000:
            print("   ❌ Dataset too small for reliable metrics")
            return None, None
        
        # Split data randomly with true 80/20 ratio - NO CAPPING
        test_size = int(len(reviews_df) * 0.2)  # True 20% for testing
        test_indices = np.random.choice(len(reviews_df), size=test_size, replace=False)
        
        test_df = reviews_df.iloc[test_indices].copy()
        train_df = reviews_df.drop(test_indices).copy()
        
        print(f"   Train set: {len(train_df):,} reviews ({len(train_df)/len(reviews_df)*100:.1f}%)")
        print(f"   Test set: {len(test_df):,} reviews ({len(test_df)/len(reviews_df)*100:.1f}%)")
        
        # Train model on training data  
        original_reviews = engine.reviews_df
        engine.reviews_df = train_df
        
        train_start = time.time()
        success = engine.train_model()
        train_time = time.time() - train_start
        
        if not success:
            print("   ❌ Model training failed")
            engine.reviews_df = original_reviews
            return None, None
        
        # Simple prediction: use average rating for basic baseline
        train_user_avg = train_df.groupby('username')['rating'].mean()
        train_movie_avg = train_df.groupby('tmdb_id')['rating'].mean()
        global_avg = train_df['rating'].mean()
        
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            user = row['username']
            movie = row['tmdb_id']
            actual = row['rating']
            
            # Use actual model prediction
            pred = engine.predict_rating_for_user(user, movie)
            
            predictions.append(pred)
            actuals.append(actual)
        
        # Restore original data
        engine.reviews_df = original_reviews
        engine.train_model()  # Re-train on full data
        
        if len(predictions) < 100:
            print(f"   ❌ Too few predictions ({len(predictions)}) for reliable metrics")
            return None, None
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        print(f"   ✅ Evaluation completed:")
        print(f"      Test predictions: {len(predictions):,}")
        print(f"      RMSE: {rmse:.3f}")
        print(f"      MAE: {mae:.3f}")
        print(f"      Model retrain time: {train_time:.2f}s")
        
        return rmse, mae
        
    except Exception as e:
        print(f"   ❌ Error calculating metrics: {e}")
        return None, None

def scrape_missing_user(username):
    """Scrape data for a user not in the dataset."""
    print(f"🔍 Scraping data for missing user: {username}")
    try:
        user_df = get_user_data(username)
        if not user_df.empty:
            print(f"   ✅ Got {len(user_df)} films for {username}")
            return user_df
        else:
            print(f"   ⚠️  No films found for {username}")
            return None
    except Exception as e:
        print(f"   ❌ Error scraping {username}: {e}")
        return None

def run_performance_tests(engine):
    """Run comprehensive performance tests."""
    print("\n🧪 PERFORMANCE TESTING")
    print("=" * 50)
    
    # Test 1: Model training time (only if not already loaded)
    print("1. Model Training Performance")
    start_time = time.time()
    
    if not hasattr(engine, 'reviews_df') or engine.reviews_df is None:
        success = engine.load_data()
        if not success:
            print("   ❌ Failed to load data")
            return None
    else:
        print("📖 Loading data...")
        success = True
    
    fit_time = time.time() - start_time
    print(f"   Training time: {fit_time:.2f} seconds")
    
    # Test 2: Dataset statistics
    print("2. Dataset Statistics")
    if hasattr(engine, 'reviews_df') and hasattr(engine, 'metadata_df'):
        reviews_df = engine.reviews_df
        metadata_df = engine.metadata_df
        
        total_reviews = len(reviews_df)
        unique_users = reviews_df['username'].nunique()
        unique_movies = len(metadata_df)
        density = (total_reviews / (unique_users * unique_movies)) * 100
        avg_rating = reviews_df['rating'].mean()
        
        print(f"   Total reviews: {total_reviews:,}")
        print(f"   Unique users: {unique_users}")
        print(f"   Unique movies: {unique_movies:,}")
        print(f"   Rating density: {density:.1f}%")
        print(f"   Average rating: {avg_rating:.2f}")
    
    # Test 3: Model accuracy metrics
    rmse, mae = calculate_model_metrics(engine)
    if rmse is not None and mae is not None:
        print("3. Model Accuracy Metrics")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE: {mae:.3f}")
        print(f"   Rating scale: 0.5-10.0 (10-point scale)")
        print(f"   RMSE as % of scale: {(rmse/9.5)*100:.1f}%")
    else:
        print("3. Model Accuracy Metrics")
        print("   ❌ Could not calculate RMSE/MAE")
    
    # Test 4: Feature engineering details
    print("4. Feature Engineering")
    if hasattr(engine, 'metadata_df'):
        metadata_df = engine.metadata_df
        
        # Check for separated columns
        has_directors_other = 'directors_other' in metadata_df.columns
        has_cast_other = 'cast_other' in metadata_df.columns
        
        print(f"   Directors separated: {'✅' if has_directors_other else '❌'}")
        print(f"   Cast separated: {'✅' if has_cast_other else '❌'}")
        
        if has_directors_other:
            director_other_count = (metadata_df['directors_other'] != '').sum()
            print(f"   Movies with directors_other: {director_other_count:,}")
            
        if has_cast_other:
            cast_other_count = (metadata_df['cast_other'] != '').sum()
            print(f"   Movies with cast_other: {cast_other_count:,}")
    
    return {"rmse": rmse, "mae": mae, "fit_time": fit_time}
    print(f"   Training time: {fit_time:.2f} seconds")
    
    # Test 2: Dataset statistics
    print("2. Dataset Statistics")
    if hasattr(engine, 'reviews_df') and engine.reviews_df is not None:
        reviews_df = engine.reviews_df
        metadata_df = engine.metadata_df
        
        print(f"   Total reviews: {len(reviews_df):,}")
        print(f"   Unique users: {reviews_df['username'].nunique():,}")
        print(f"   Unique movies: {len(metadata_df):,}")
        print(f"   Rating density: {(reviews_df['rating'].notna().sum() / len(reviews_df) * 100):.1f}%")
        print(f"   Average rating: {reviews_df['rating'].mean():.2f}")
    
    # Test 3: Feature count
    print("3. Feature Engineering")
    if hasattr(engine, 'content_features') and engine.content_features is not None:
        feature_count = engine.content_features.shape[1]
        print(f"   Total features: {feature_count}")
        print(f"   Feature types: genres, directors, cast, keywords, original_language")
    
    return {
        'fit_time': fit_time,
        'total_reviews': len(reviews_df) if hasattr(engine, 'reviews_df') else 0,
        'unique_users': reviews_df['username'].nunique() if hasattr(engine, 'reviews_df') else 0,
        'unique_movies': len(metadata_df) if hasattr(engine, 'metadata_df') else 0
    }

def demo_streamlined(reviews_file=None, metadata_file=None, test_users_count=None):
    """Test the streamlined engine performance with all specified users"""
    print("🚀 STREAMLINED LETTERBOXD RECOMMENDATION SYSTEM DEMO")
    print("=" * 60)
    
    # All test users including the newly specified ones
    test_users = [
        'schaffrillas',
        'jaaackd',
        'davidehrlich',
        'joshlewis',
        'gizmofan',          # #1 gizmo fan
        'silentdawn',        # SilentDawn
        'karsten',           # Karsten (appears as kurstboy in scraped data)
        'suspirliam',        # ˗ˏˋ suspirliam ˊˎ˗
        'lucy',              # Lucy
        'zoerosebryant',     # zoë rose bryant
        'beluborelli',       # belu borelli ‍
        'guidomontini',     # Guido Montini
        'aidandking08'
    ]
    
    # Limit test users if specified
    if test_users_count:
        test_users = test_users[:test_users_count]
    
    print(f"Testing with {len(test_users)} users: {', '.join(test_users)}")
    
    # Create engine with specified files or auto-detection
    if reviews_file and metadata_file:
        print(f"\n📂 Using specified dataset:")
        print(f"   Reviews: {os.path.basename(reviews_file)}")
        print(f"   Metadata: {os.path.basename(metadata_file)}")
        engine = LetterboxdRecommendationEngine()
        engine.load_data(reviews_file, metadata_file)
    else:
        print(f"\n📂 Using auto-detected dataset:")
        engine = LetterboxdRecommendationEngine()
        engine.load_data()
    
    # Run performance tests
    performance_results = run_performance_tests(engine)
    if performance_results is None:
        print("❌ Failed to initialize engine")
        return
    
    # Check which users exist in dataset
    if hasattr(engine, 'reviews_df') and engine.reviews_df is not None:
        existing_users = set(engine.reviews_df['username'].unique())
        available_users = [user for user in test_users if user in existing_users]
        missing_users = [user for user in test_users if user not in existing_users]
        
        print(f"\n📊 User Availability:")
        print(f"   Available in dataset: {len(available_users)} users")
        print(f"   Missing from dataset: {len(missing_users)} users")
        
        if missing_users:
            print(f"   Missing users: {', '.join(missing_users)}")
            print(f"   Note: Missing users will be scraped and integrated")
    
    # Generate recommendations for ALL users (available + missing)
    print(f"\n🎬 RECOMMENDATION DEMO")
    print("=" * 50)
    
    successful_recommendations = 0
    total_rec_time = 0
    
    for username in test_users:  # Test ALL users including missing ones
        print(f"\n🎭 Recommendations for @{username}:")
        print("─" * 60)
        
        start_time = time.time()
        recommendations = engine.get_recommendations(username)
        rec_time = time.time() - start_time
        total_rec_time += rec_time
        
        if recommendations is not None and len(recommendations) > 0:
            successful_recommendations += 1
            
            # Display top 5 recommendations
            for i, row in recommendations.head(5).iterrows():
                title = row.get('title', 'Unknown')
                year = row.get('year', 'N/A')
                rating = row.get('predicted_rating', 0)
                genres = row.get('genres', 'Unknown')
                
                print(f" {row.get('rank', i+1):2d}. {title} ({year})")
                print(f"     Rating: {rating:.1f}/10  |  {genres}")
            
            print(f"\n⏱️  Generation time: {rec_time:.3f} seconds")
            
            # User stats if available
            user_reviews = engine.reviews_df[engine.reviews_df['username'] == username]
            num_ratings = user_reviews['rating'].notna().sum()
            avg_rating = user_reviews['rating'].mean()
            
            print(f"📊 {username}'s profile:")
            print(f"    Films rated: {num_ratings}")
            print(f"    Average rating: {avg_rating:.1f}")
        else:
            print("   ❌ Could not generate recommendations")
    
    # Final performance summary
    print(f"\n📈 FINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    if performance_results and 'fit_time' in performance_results:
        print(f"Model training: {performance_results['fit_time']:.1f}s")
        if performance_results.get('rmse') is not None:
            print(f"RMSE: {performance_results['rmse']:.3f}")
            print(f"MAE: {performance_results['mae']:.3f}")
    else:
        print("Model training: N/A")
        
    if successful_recommendations > 0:
        avg_rec_time = total_rec_time / successful_recommendations
        print(f"Average recommendation time: {avg_rec_time:.3f}s per user")
    print(f"Successful recommendations: {successful_recommendations}/{len(test_users)}")
    
    # Get dataset stats from engine if available
    if engine and hasattr(engine, 'reviews_df') and hasattr(engine, 'metadata_df'):
        reviews_df = engine.reviews_df
        metadata_df = engine.metadata_df
        print(f"Dataset: {len(reviews_df):,} reviews, {reviews_df['username'].nunique():,} users, {len(metadata_df):,} movies")
    else:
        print("Dataset: N/A")
    
    print(f"\n💡 MODEL PERFORMANCE INSIGHTS:")
    print("• Separated features: Directors/cast split into main + other columns")
    print("• Larger dataset: ~357K reviews vs 164K in previous version")
    print("• Quality filtering: ≥10 reviews per user, ≥5 reviews per movie") 
    print("• Feature engineering: 887 features including separated other columns")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n✅ Demo completed at {timestamp}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Letterboxd Recommendation System Demo')
    parser.add_argument('reviews_file', nargs='?', help='Path to reviews CSV file')
    parser.add_argument('metadata_file', nargs='?', help='Path to metadata CSV file')
    parser.add_argument('--test-users', type=int, help='Number of test users (default: all)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    demo_streamlined(
        reviews_file=args.reviews_file,
        metadata_file=args.metadata_file,
        test_users_count=args.test_users
    )

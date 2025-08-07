#!/usr/bin/env python3
"""
Demo for the Streamlined Letterboxd Recommendation Engine V2

Clean, focused demo that tests the recommendation system with real data.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlined_engine_v2 import LetterboxdRecommendationEngine


def calculate_model_metrics(engine, test_sample_size=5000):
    """Calculate RMSE and MAE for the trained model"""
    try:
        print("📊 Calculating model metrics...")
        
        # Sample a subset of reviews for testing (for speed)
        test_reviews = engine.reviews_df.sample(n=min(test_sample_size, len(engine.reviews_df)), 
                                               random_state=42)
        
        actual_ratings = []
        predicted_ratings = []
        
        for _, review in test_reviews.iterrows():
            username = review['username']
            tmdb_id = review['tmdb_id']
            actual_rating = review['rating']
            
            # Get prediction from our model
            predicted_rating = engine.predict_rating(username, tmdb_id)
            
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        
        print(f"   ✅ RMSE: {rmse:.3f}")
        print(f"   ✅ MAE: {mae:.3f}")
        print(f"   📊 Based on {len(actual_ratings)} predictions")
        
        return {'rmse': rmse, 'mae': mae, 'n_predictions': len(actual_ratings)}
        
    except Exception as e:
        print(f"   ❌ Error calculating metrics: {e}")
        return None


def test_user_recommendations(engine, test_users, scrape_missing=False):
    """Test recommendations for a list of users"""
    print("\n🎬 RECOMMENDATION DEMO")
    print("=" * 50)
    
    successful_recommendations = 0
    total_rec_time = 0
    
    for username in test_users:
        print(f"\n🎭 Recommendations for @{username}:")
        print("─" * 60)
        
        # Check if user exists in dataset
        if username not in engine.user_mapping:
            if scrape_missing:
                print(f"   🔍 User not found, attempting to scrape...")
                success = engine.scrape_and_add_user(username, max_reviews=500)
                if success:
                    # Retrain model with new user
                    print("   🔄 Retraining model with new user data...")
                    engine.train_model()
                else:
                    print(f"   ❌ Could not scrape data for {username}")
                    continue
            else:
                print(f"   ❌ User {username} not found in training data")
                continue
        
        # Generate recommendations
        start_time = time.time()
        recommendations = engine.get_recommendations(username, n_recommendations=5)
        rec_time = time.time() - start_time
        total_rec_time += rec_time
        
        if len(recommendations) > 0:
            successful_recommendations += 1
            
            # Display recommendations
            for i, rec in recommendations.iterrows():
                print(f"  {i+1:2d}. {rec['title']} ({rec['year']})")
                print(f"      Rating: {rec['predicted_rating']}/10  |  {rec['genres']}")
            
            print(f"\n⏱️  Generation time: {rec_time:.3f} seconds")
            
            # Show user stats
            user_stats = engine.get_user_stats(username)
            if user_stats:
                print(f"📊 {username}'s profile:")
                print(f"    Films rated: {user_stats['num_reviews']}")
                print(f"    Average rating: {user_stats['avg_rating']:.1f}")
        else:
            print("   ❌ Could not generate recommendations")
    
    return {
        'successful': successful_recommendations,
        'total': len(test_users),
        'avg_time': total_rec_time / len(test_users) if test_users else 0
    }


def main():
    """Main demo function"""
    print("🚀 LETTERBOXD RECOMMENDATION ENGINE V2 DEMO")
    print("=" * 60)
    
    # Define test users
    test_users = [
        'schaffrillas', 'jaaackd', 'davidehrlich', 'joshlewis', 'gizmofan',
        'silentdawn', 'karsten', 'suspirliam', 'lucy', 'zoerosebryant',
        'beluborelli', 'guidomontini', 'aidandking08'
    ]
    
    print(f"Testing with {len(test_users)} users: {', '.join(test_users)}")
    
    # Auto-detect latest datasets
    data_dir = 'data'
    reviews_file = os.path.join(data_dir, 'cleaned_reviews_explicit_20250730_135737.csv')
    metadata_file = os.path.join(data_dir, 'final_metadata_improved_20250731_082236.csv')
    
    if not os.path.exists(reviews_file) or not os.path.exists(metadata_file):
        print("❌ Required data files not found!")
        print(f"   Reviews: {reviews_file}")
        print(f"   Metadata: {metadata_file}")
        return
    
    print(f"\n📂 Using datasets:")
    print(f"   Reviews: {os.path.basename(reviews_file)}")
    print(f"   Metadata: {os.path.basename(metadata_file)}")
    
    # Initialize and load data
    engine = LetterboxdRecommendationEngine(
        alpha=0.7,                    # 70% collaborative filtering, 30% content-based
        num_recommendations=25,       # Default number of recommendations
        num_components=25            # SVD components
    )
    
    if not engine.load_data(reviews_file, metadata_file):
        print("❌ Failed to load data")
        return
    
    # Train model
    print("\n🧪 PERFORMANCE TESTING")
    print("=" * 50)
    
    print("1. Model Training Performance")
    train_start = time.time()
    
    if not engine.train_model():
        print("❌ Failed to train model")
        return
    
    train_time = time.time() - train_start
    print(f"   ✅ Training completed in {train_time:.2f} seconds")
    
    # Display model statistics
    print("\n2. Model Statistics")
    stats = engine.get_model_stats()
    print(f"   Total reviews: {stats['num_reviews']:,}")
    print(f"   Unique users: {stats['num_users']:,}")
    print(f"   Unique movies: {stats['num_movies']:,}")
    print(f"   Rating density: {stats['density']*100:.3f}%")
    print(f"   Average rating: {stats['avg_rating']:.2f}")
    print(f"   SVD components: {stats['svd_components']}")
    print(f"   Content features: {stats['content_features']}")
    
    # Calculate model accuracy
    print("\n3. Model Accuracy")
    metrics = calculate_model_metrics(engine, test_sample_size=10000)
    
    # Check user availability
    print("\n4. User Availability")
    available_users = [user for user in test_users if user in engine.user_mapping]
    missing_users = [user for user in test_users if user not in engine.user_mapping]
    
    print(f"   Available in dataset: {len(available_users)} users")
    print(f"   Missing from dataset: {len(missing_users)} users")
    if missing_users:
        print(f"   Missing users: {', '.join(missing_users)}")
    
    # Test recommendations
    rec_results = test_user_recommendations(engine, test_users, scrape_missing=False)
    
    # Final summary
    print("\n📈 FINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Model training: {train_time:.1f}s")
    if metrics:
        print(f"Model accuracy: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")
    print(f"Average recommendation time: {rec_results['avg_time']:.3f}s per user")
    print(f"Successful recommendations: {rec_results['successful']}/{rec_results['total']}")
    print(f"Dataset: {stats['num_reviews']:,} reviews, {stats['num_users']:,} users, {stats['num_movies']:,} movies")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n✅ Demo completed at {timestamp}")


if __name__ == "__main__":
    main()

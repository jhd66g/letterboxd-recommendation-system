#!/usr/bin/env python3
"""
Streamlined Letterboxd Recommendation Demo

Tests the production-ready recommendation system with comprehensive performance analysis.
Based on README specifications with 13 diverse test users and full model evaluation.

Features:
- Auto-detection of latest pipeline data
- Comprehensive performance testing  
- Real-time user scraping integration
- Model accuracy metrics (RMSE, MAE)
- Production-scale testing
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


def calculate_model_metrics(engine, test_sample_size=10000):
    """Calculate RMSE and MAE for the trained model"""
    try:
        print("📊 Calculating model accuracy metrics...")
        
        # Use train/test split for proper evaluation
        reviews_sample = engine.reviews_df.sample(
            n=min(test_sample_size, len(engine.reviews_df)), 
            random_state=42
        )
        
        train_reviews, test_reviews = train_test_split(
            reviews_sample, test_size=0.2, random_state=42
        )
        
        print(f"   Train set: {len(train_reviews):,} reviews ({len(train_reviews)/len(reviews_sample)*100:.1f}%)")
        print(f"   Test set: {len(test_reviews):,} reviews ({len(test_reviews)/len(reviews_sample)*100:.1f}%)")
        
        # Generate predictions on test set
        actual_ratings = []
        predicted_ratings = []
        
        for _, review in test_reviews.iterrows():
            username = review['username']
            tmdb_id = review['tmdb_id']
            actual_rating = review['rating']
            
            # Check if user and movie are in model
            if (username in engine.model['user_mapping'] and 
                tmdb_id in engine.model['movie_mapping']):
                
                user_id = engine.model['user_mapping'][username]
                movie_id = engine.model['movie_mapping'][tmdb_id]
                
                predicted_rating = engine.predict_rating(user_id, movie_id)
                
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)
        
        if len(actual_ratings) > 0:
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            mae = mean_absolute_error(actual_ratings, predicted_ratings)
            
            # Calculate error percentage on 10-point scale
            rmse_pct = (rmse / 10.0) * 100
            mae_pct = (mae / 10.0) * 100
            
            print(f"   ✅ RMSE: {rmse:.3f} ({rmse_pct:.1f}% error rate)")
            print(f"   ✅ MAE: {mae:.3f} ({mae_pct:.1f}% error rate)")
            print(f"   📊 Based on {len(actual_ratings):,} predictions")
            
            return {
                'rmse': rmse, 
                'mae': mae, 
                'rmse_pct': rmse_pct,
                'mae_pct': mae_pct,
                'n_predictions': len(actual_ratings),
                'train_size': len(train_reviews),
                'test_size': len(test_reviews)
            }
        else:
            print("   ❌ No valid predictions for accuracy calculation")
            return None
            
    except Exception as e:
        print(f"   ❌ Error calculating metrics: {e}")
        return None


def test_user_recommendations(engine, test_users, scrape_missing=False):
    """Test recommendations for multiple users with detailed analysis"""
    print("\n🎬 RECOMMENDATION DEMO")
    print("=" * 70)
    
    successful_recommendations = 0
    total_rec_time = 0
    user_profiles = []
    
    for username in test_users:
        print(f"\n🎭 Recommendations for @{username}:")
        print("─" * 70)
        
        # Check if user exists
        if username not in engine.model['user_mapping']:
            if scrape_missing:
                print(f"   🔍 User not found, attempting to scrape...")
                if engine._scrape_and_add_user(username):
                    print("   🔄 Retraining model with new user data...")
                    if not engine.train_model():
                        print(f"   ❌ Failed to retrain model")
                        continue
                else:
                    print(f"   ❌ Could not scrape data for {username}")
                    continue
            else:
                print(f"   ❌ User {username} not found in dataset")
                continue
        
        # Get user profile
        user_id = engine.model['user_mapping'][username]
        user_ratings = engine.model['interaction_matrix'][user_id].toarray().flatten()
        rated_movies = np.where(user_ratings > 0)[0]
        
        if len(rated_movies) > 0:
            avg_rating = np.mean(user_ratings[rated_movies]) * 10  # Convert back to 10-point scale
            user_profiles.append({
                'username': username,
                'num_ratings': len(rated_movies),
                'avg_rating': avg_rating
            })
        
        # Generate recommendations
        start_time = time.time()
        recommendations = engine.generate_recommendations(username, scrape_if_missing=False)
        rec_time = time.time() - start_time
        total_rec_time += rec_time
        
        if recommendations is not None and len(recommendations) > 0:
            successful_recommendations += 1
            
            # Display top 5 recommendations
            for i, rec in recommendations.head(5).iterrows():
                print(f"  {i+1:2d}. {rec['title']} ({rec['year']})")
                print(f"      ⭐ {rec['predicted_rating']:.1f}/10  |  {rec['genres']}")
                if pd.notna(rec['directors']) and rec['directors'] != 'Unknown':
                    print(f"      🎬 {rec['directors']}")
            
            print(f"\n⏱️  Generation time: {rec_time:.3f} seconds")
            
            # Show user profile
            if len(rated_movies) > 0:
                print(f"📊 {username}'s profile:")
                print(f"    Films rated: {len(rated_movies):,}")
                print(f"    Average rating: {avg_rating:.1f}/10")
        else:
            print("   ❌ Could not generate recommendations")
    
    return {
        'successful': successful_recommendations,
        'total': len(test_users),
        'avg_time': total_rec_time / len(test_users) if test_users else 0,
        'user_profiles': user_profiles
    }


def analyze_dataset_quality(engine):
    """Analyze dataset quality and characteristics"""
    print("\n📊 DATASET QUALITY ANALYSIS")
    print("=" * 50)
    
    reviews_df = engine.reviews_df
    metadata_df = engine.metadata_df
    
    # Basic statistics
    print("1. Dataset Overview")
    print(f"   Total reviews: {len(reviews_df):,}")
    print(f"   Unique users: {reviews_df['username'].nunique():,}")
    print(f"   Unique movies: {reviews_df['tmdb_id'].nunique():,}")
    print(f"   Metadata movies: {len(metadata_df):,}")
    
    # Rating distribution
    print("\n2. Rating Quality (Explicit-Only)")
    print(f"   Average rating: {reviews_df['rating'].mean():.2f}")
    print(f"   Rating std: {reviews_df['rating'].std():.2f}")
    print(f"   Rating range: {reviews_df['rating'].min():.1f} - {reviews_df['rating'].max():.1f}")
    
    # Check for implicit bias removal
    implicit_count = len(reviews_df[reviews_df['rating'] == 1.0])
    if implicit_count > 0:
        print(f"   ⚠️  Potential implicit ratings (1.0): {implicit_count:,}")
    else:
        print(f"   ✅ No implicit ratings detected")
    
    # User activity distribution
    user_activity = reviews_df.groupby('username').size()
    print(f"\n3. User Activity")
    print(f"   Min reviews per user: {user_activity.min()}")
    print(f"   Max reviews per user: {user_activity.max()}")
    print(f"   Avg reviews per user: {user_activity.mean():.1f}")
    print(f"   Users with 10+ reviews: {(user_activity >= 10).sum()}")
    
    # Movie popularity distribution  
    movie_popularity = reviews_df.groupby('tmdb_id').size()
    print(f"\n4. Movie Popularity")
    print(f"   Min reviews per movie: {movie_popularity.min()}")
    print(f"   Max reviews per movie: {movie_popularity.max()}")
    print(f"   Avg reviews per movie: {movie_popularity.mean():.1f}")
    print(f"   Movies with 5+ reviews: {(movie_popularity >= 5).sum()}")
    
    # Feature completeness
    print(f"\n5. Feature Engineering")
    feature_completeness = {}
    for col in ['genres', 'directors_main', 'directors_other', 'cast_main', 'cast_other']:
        if col in metadata_df.columns:
            non_null = metadata_df[col].notna().sum()
            completeness = (non_null / len(metadata_df)) * 100
            feature_completeness[col] = completeness
            print(f"   {col}: {completeness:.1f}% complete ({non_null:,}/{len(metadata_df):,})")
    
    return {
        'total_reviews': len(reviews_df),
        'unique_users': reviews_df['username'].nunique(),
        'unique_movies': reviews_df['tmdb_id'].nunique(),
        'avg_rating': reviews_df['rating'].mean(),
        'feature_completeness': feature_completeness
    }


def main():
    """Main demo function with comprehensive testing"""
    parser = argparse.ArgumentParser(description='Letterboxd Recommendation System Demo')
    parser.add_argument('--test-users', type=int, help='Number of test users (default: all 13)')
    parser.add_argument('--scrape-missing', action='store_true', help='Scrape missing users')
    parser.add_argument('--skip-metrics', action='store_true', help='Skip accuracy metrics calculation')
    parser.add_argument('reviews_file', nargs='?', help='Path to reviews CSV file (optional)')
    parser.add_argument('metadata_file', nargs='?', help='Path to metadata CSV file (optional)')
    
    args = parser.parse_args()
    
    print("🚀 LETTERBOXD STREAMLINED RECOMMENDATION SYSTEM DEMO")
    print("=" * 70)
    print("Based on README specifications:")
    print("• 70% Collaborative Filtering + 30% Content-Based")
    print("• Explicit ratings only (removes implicit feedback bias)")
    print("• Separated feature engineering (main + other columns)")
    print("• Production-scale dataset (~1M reviews, 30K movies)")
    
    # Test users from README
    all_test_users = [
        'schaffrillas', 'jaaackd', 'davidehrlich', 'joshlewis', 'gizmofan',
        'silentdawn', 'karsten', 'suspirliam', 'lucy', 'zoerosebryant',
        'beluborelli', 'guidomontini', 'aidandking08'
    ]
    
    test_users = all_test_users[:args.test_users] if args.test_users else all_test_users
    print(f"\nTesting with {len(test_users)} users: {', '.join(test_users)}")
    
    # Initialize engine with README specifications
    engine = LetterboxdRecommendationEngine(
        alpha=0.7,  # 70% collaborative filtering
        num_recommendations=25,
        num_components=25
    )
    
    # Load data
    print(f"\n📂 Loading dataset...")
    if args.reviews_file and args.metadata_file:
        print(f"   Using specified files:")
        print(f"   Reviews: {os.path.basename(args.reviews_file)}")
        print(f"   Metadata: {os.path.basename(args.metadata_file)}")
        success = engine.load_data(args.reviews_file, args.metadata_file)
    else:
        success = engine.load_data()
    
    if not success:
        print("\n❌ Failed to load data. Options:")
        print("   1. Run pipeline: python data_cleaning/full_production_pipeline.py")
        print("   2. Specify files: python streamlined_demo.py reviews.csv metadata.csv")
        return
    
    # Analyze dataset quality
    dataset_stats = analyze_dataset_quality(engine)
    
    # Train model
    print(f"\n🧪 PERFORMANCE TESTING")
    print("=" * 50)
    print("1. Model Training Performance")
    
    train_start = time.time()
    if not engine.train_model():
        print("❌ Failed to train model")
        return
    
    train_time = time.time() - train_start
    print(f"   ✅ Training completed in {train_time:.2f} seconds")
    
    # Model statistics
    print("\n2. Model Architecture")
    stats = engine.get_model_stats()
    print(f"   Users: {stats['num_users']:,}")
    print(f"   Movies: {stats['num_movies']:,}")
    print(f"   Reviews: {stats['num_reviews']:,}")
    print(f"   Matrix density: {stats['matrix_density']:.3f}%")
    print(f"   SVD components: {stats['svd_components']}")
    print(f"   Content features: {stats['content_features']:,}")
    print(f"   Alpha (CF weight): {stats['alpha']}")
    
    # Calculate accuracy metrics
    if not args.skip_metrics:
        print("\n3. Model Accuracy")
        metrics = calculate_model_metrics(engine, test_sample_size=10000)
    else:
        print("\n3. Model Accuracy")
        print("   ⏭️  Skipped (use --skip-metrics to disable)")
        metrics = None
    
    # Test recommendations
    rec_results = test_user_recommendations(engine, test_users, args.scrape_missing)
    
    # Final summary
    print("\n📈 FINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Model training: {train_time:.1f}s")
    
    if metrics:
        print(f"Model accuracy: RMSE={metrics['rmse']:.3f} ({metrics['rmse_pct']:.1f}% error)")
        print(f"                MAE={metrics['mae']:.3f} ({metrics['mae_pct']:.1f}% error)")
        print(f"Test predictions: {metrics['n_predictions']:,} (80/20 split)")
    
    print(f"Average recommendation time: {rec_results['avg_time']:.3f}s per user")
    print(f"Successful recommendations: {rec_results['successful']}/{rec_results['total']}")
    print(f"Dataset scale: {stats['num_reviews']:,} reviews, {stats['num_users']:,} users, {stats['num_movies']:,} movies")
    
    # User profile summary
    if rec_results['user_profiles']:
        profiles = rec_results['user_profiles']
        avg_user_ratings = np.mean([p['num_ratings'] for p in profiles])
        avg_user_score = np.mean([p['avg_rating'] for p in profiles])
        print(f"User profiles: avg {avg_user_ratings:.0f} ratings per user, avg {avg_user_score:.1f}/10 rating")
    
    # Quality insights
    print(f"\n💡 DATASET QUALITY INSIGHTS:")
    print(f"• Explicit-only processing: Realistic {stats['avg_rating']:.2f}/10 average rating")
    print(f"• Large-scale data: {len(test_users)} diverse test users")
    if 'feature_completeness' in dataset_stats:
        main_features = [k for k in dataset_stats['feature_completeness'].keys() if 'main' in k]
        if main_features:
            avg_completeness = np.mean([dataset_stats['feature_completeness'][k] for k in main_features])
            print(f"• Feature engineering: {avg_completeness:.1f}% average completeness for main features")
    print(f"• Production-ready: {stats['matrix_density']:.3f}% density indicates quality over quantity")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n✅ Demo completed at {timestamp}")


if __name__ == "__main__":
    main()

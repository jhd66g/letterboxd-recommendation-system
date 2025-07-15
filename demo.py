#!/usr/bin/env python3
"""
Demo script for the Letterboxd Recommendation System
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import LetterboxdRecommendationEngine

def demo_basic_usage():
    """Demonstrate basic usage of the recommendation engine"""
    print("🎬 LETTERBOXD RECOMMENDATION SYSTEM DEMO")
    print("=" * 50)
    
    # Test users
    test_users = ['schaffrillas', 'jaaackd']
    
    # Create engine
    print("\n1. Creating recommendation engine...")
    engine = LetterboxdRecommendationEngine()
    
    # Load data
    print("\n2. Loading data...")
    if not engine.load_data():
        print("❌ Failed to load data. Please ensure data files exist.")
        return
    
    # Generate recommendations for test users
    for username in test_users:
        print(f"\n3. Generating recommendations for {username}...")
        recommendations = engine.generate_recommendations(username)
        
        if recommendations is not None:
            print(f"\n🎬 Top 10 Recommendations for {username}:")
            print("-" * 60)
            
            for i, row in recommendations.head(10).iterrows():
                print(f"{row['rank']:2d}. {row['title']} ({row['year']})")
                print(f"    Rating: {row['predicted_rating']:.1f}/10")
                print(f"    Genres: {row['genres']}")
                print()
        else:
            print(f"❌ Failed to generate recommendations for {username}")

def main():
    """Main demo function"""
    print("🚀 Starting Letterboxd Recommendation System Demo...")
    
    try:
        demo_basic_usage()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("=" * 50)
        
        print("\n💡 Usage:")
        print("python recommendation_engine.py <username>")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check that all required files are in the correct locations.")

if __name__ == "__main__":
    main()

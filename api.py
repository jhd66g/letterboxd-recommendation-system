#!/usr/bin/env python3
"""
FastAPI service for Letterboxd movie recommendations.

Wraps the recommendation engine and handles new users by scraping their data.
"""

from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import RecommendationEngine
from data_pipeline.scraper import FastScraper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Letterboxd Recommendation API", version="1.0.0")

class RecommendationAPI:
    def __init__(self, mode='hybrid'):
        self.engine = None
        self.reviews_file = None
        self.metadata_file = None
        self.features_file = None
        self.mode = mode
        self.scraper = FastScraper(max_workers=1)  # Single user scraping
        self._load_latest_data()
        self._initialize_engine(mode)
    
    def _load_latest_data(self):
        """Find and load the latest data files."""
        # Find latest data files
        reviews_files = list(Path(".").glob("**/final_reviews_*.csv"))
        metadata_files = list(Path(".").glob("**/final_metadata_*.csv"))
        features_files = list(Path(".").glob("**/final_features_*.csv"))
        
        if not reviews_files or not metadata_files or not features_files:
            raise FileNotFoundError("Required data files not found")
        
        self.reviews_file = max(reviews_files, key=lambda p: p.stat().st_mtime)
        self.metadata_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        self.features_file = max(features_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Using data files:")
        logger.info(f"  Reviews: {self.reviews_file}")
        logger.info(f"  Metadata: {self.metadata_file}")
        logger.info(f"  Features: {self.features_file}")
    
    def _initialize_engine(self, mode='hybrid'):
        """Initialize the recommendation engine."""
        self.engine = RecommendationEngine()
        self.engine.load_data(str(self.reviews_file), str(self.features_file))
        
        # Try to load existing model for the specified mode
        model_files = list(Path("models").glob(f"lightfm_{mode}_*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            self.engine.load_model(latest_model)
            logger.info(f"Loaded model: {latest_model}")
        else:
            logger.warning(f"No existing {mode} model found, training new model...")
            self.engine.train_model(mode=mode)
            self.engine.save_model(mode)
    
    def _user_exists(self, username):
        """Check if user exists in the dataset."""
        return username in self.engine.user_mapping
    
    def _scrape_and_add_user(self, username, epochs=10):
        """Scrape a new user and add them to the dataset."""
        logger.info(f"Scraping new user: {username}")
        
        # Scrape user reviews
        _, films, error = self.scraper.process_user(username, explicit_only=True)
        
        if error or not films:
            raise HTTPException(status_code=404, detail=f"User '{username}' has no reviews or could not be scraped")
        
        # Convert to DataFrame
        new_reviews_df = pd.DataFrame(films)
        
        # Load metadata to map titles to tmdb_ids
        metadata_df = pd.read_csv(self.metadata_file)
        
        # Map reviews to tmdb_ids based on title and year
        new_reviews_with_tmdb = []
        for _, review in new_reviews_df.iterrows():
            # Try to match by title and year first
            if pd.notna(review['year']):
                matches = metadata_df[
                    (metadata_df['title'].str.lower() == review['title'].lower()) &
                    (metadata_df['year'] == review['year'])
                ]
            else:
                # If no year, match by title only (take the first match)
                matches = metadata_df[
                    metadata_df['title'].str.lower() == review['title'].lower()
                ]
            
            if not matches.empty:
                tmdb_id = matches.iloc[0]['tmdb_id']
                new_reviews_with_tmdb.append({
                    'username': review['username'],
                    'tmdb_id': tmdb_id,
                    'rating': review['rating']
                })
        
        if not new_reviews_with_tmdb:
            raise HTTPException(status_code=404, detail=f"No movies from user '{username}' could be matched to existing dataset")
        
        # Add to existing reviews file
        existing_reviews = pd.read_csv(self.reviews_file)
        new_reviews_df = pd.DataFrame(new_reviews_with_tmdb)
        updated_reviews = pd.concat([existing_reviews, new_reviews_df], ignore_index=True)
        
        # Save updated reviews
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_reviews_file = self.reviews_file.parent / f"final_reviews_{timestamp}.csv"
        updated_reviews.to_csv(new_reviews_file, index=False)
        
        # Update internal reference
        self.reviews_file = new_reviews_file
        
        # Reload engine with new data
        self.engine = RecommendationEngine()
        self.engine.load_data(str(self.reviews_file), str(self.features_file))
        
        # Retrain model since we have a new user (existing model won't have new user mapping)
        logger.info(f"Retraining model with new user data using {epochs} epochs...")
        self.engine.train_model(mode=self.mode, epochs=epochs)
        model_file = self.engine.save_model(self.mode)
        logger.info(f"Retrained model saved: {model_file}")
        
        logger.info(f"Added user '{username}' with {len(new_reviews_with_tmdb)} matched reviews")
        return len(new_reviews_with_tmdb)
    
    def get_recommendations(self, username, top_n=25, epochs=10):
        """Get recommendations for a user, scraping if necessary."""
        # Check if user exists
        if not self._user_exists(username):
            self._scrape_and_add_user(username, epochs)
        
        # Generate recommendations
        recommendations = self.engine.recommend(username, top_n=top_n, mode=self.mode)
        
        # Load metadata for enrichment
        metadata_df = pd.read_csv(self.metadata_file)
        
        # Load user's existing reviews to check for already watched movies
        reviews_df = pd.read_csv(self.reviews_file)
        user_reviewed_tmdb_ids = set(reviews_df[reviews_df['username'] == username]['tmdb_id'].tolist())
        
        # Enrich recommendations with metadata
        enriched_recommendations = []
        for rec in recommendations:
            tmdb_id = rec['tmdb_id']
            metadata = metadata_df[metadata_df['tmdb_id'] == tmdb_id]
            
            if not metadata.empty:
                meta = metadata.iloc[0]
                enriched_rec = {
                    "rank": rec['rank'],
                    "tmdb_id": rec['tmdb_id'],
                    "score": rec['score'],
                    "title": meta.get('title', 'Unknown'),
                    "year": int(meta['year']) if pd.notna(meta.get('year')) else None,
                    "runtime": int(meta['runtime']) if pd.notna(meta.get('runtime')) else None,
                    "directors": eval(meta['directors']) if pd.notna(meta.get('directors')) else [],
                    "cast": eval(meta['cast']) if pd.notna(meta.get('cast')) else [],
                    "genres": eval(meta['genres']) if pd.notna(meta.get('genres')) else [],
                    "language": meta.get('original_language', ''),
                    "overview": meta.get('overview', ''),
                    "budget": int(meta['budget']) if pd.notna(meta.get('budget')) else None,
                    "revenue": int(meta['revenue']) if pd.notna(meta.get('revenue')) else None,
                    "already_reviewed": rec['tmdb_id'] in user_reviewed_tmdb_ids
                }
                enriched_recommendations.append(enriched_rec)
            else:
                # Fallback to basic recommendation data
                enriched_recommendations.append(rec)
        
        return enriched_recommendations

# Initialize global API instance for web server mode (hybrid only)
api_instance = RecommendationAPI(mode='hybrid')

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Letterboxd Recommendation API", "status": "healthy"}

@app.get("/recommendations/{username}")
async def get_recommendations(username: str, top_n: int = 25, epochs: int = 10):
    """
    Get top-N movie recommendations for a user.
    
    Args:
        username: Letterboxd username
        top_n: Number of recommendations to return (default: 25)
        epochs: Number of training epochs for new users (default: 10)
    
    Returns:
        JSON array of ranked movie recommendations
    """
    try:
        if top_n <= 0 or top_n > 100:
            raise HTTPException(status_code=400, detail="top_n must be between 1 and 100")
        
        if epochs <= 0 or epochs > 100:
            raise HTTPException(status_code=400, detail="epochs must be between 1 and 100")
        
        recommendations = api_instance.get_recommendations(username, top_n, epochs)
        
        return {
            "username": username,
            "recommendations": recommendations,
            "total": len(recommendations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/user/{username}/status")
async def user_status(username: str):
    """Check if a user exists in the dataset."""
    exists = api_instance._user_exists(username)
    return {
        "username": username,
        "exists_in_dataset": exists,
        "total_users": len(api_instance.engine.user_mapping)
    }

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Letterboxd Recommendation API")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "cf_only", "cb_only"],
                        help="Recommendation mode (default: hybrid)")
    parser.add_argument("--user", type=str, help="Username to generate recommendations for")
    parser.add_argument("--top_n", type=int, default=25, help="Number of recommendations")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for new users")
    parser.add_argument("--serve", action="store_true", help="Start web server (default if no --user)")
    
    args = parser.parse_args()
    
    # Initialize API with the specified mode
    try:
        api_instance = RecommendationAPI(mode=args.mode)
        logger.info("✅ Recommendation API initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        sys.exit(1)
    
    if args.user:
        # CLI mode - generate recommendations and print
        try:
            recommendations = api_instance.get_recommendations(args.user, args.top_n, args.epochs)
            
            # Print in readable format
            print(f"\n🎬 Top {len(recommendations)} recommendations for '{args.user}':")
            print("=" * 60)
            
            for rec in recommendations:
                already_reviewed = " ⭐" if rec.get('already_reviewed', False) else ""
                print(f"{rec['rank']:2}. {rec['title']} ({rec['year']}) - Score: {rec['score']:.2f}{already_reviewed}")
                if rec.get('directors'):
                    print(f"    Director(s): {', '.join(rec['directors'])}")
                if rec.get('genres'):
                    print(f"    Genres: {', '.join(rec['genres'])}")
                print()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Web server mode
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

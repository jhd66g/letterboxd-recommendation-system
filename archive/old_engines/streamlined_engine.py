#!/usr/bin/env python3
"""
Streamlined Letterboxd Recommendation Engine

A fast, lightweight recommendation engine using hybrid collaborative filtering and content-based features.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix
import os
import re
import sys
import time
import requests
import subprocess


class LetterboxdRecommendationEngine:
    def __init__(self, alpha=0.7, num_recommendations=25, num_components=25):
        
        # Model parameters
        self.alpha = alpha  # Weight for collaborative filtering vs content-based
        self.num_recommendations = num_recommendations
        self.num_components = num_components
        
        # Data containers
        self.reviews_df = None
        self.metadata_df = None
        self.cf_model = None
        self.user_mapping = {}
        self.movie_mapping = {}
        self.user_embeddings = None
        self.item_embeddings = None
        self.interaction_matrix = None
        
        # Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.scrapers_dir = os.path.join(self.base_dir, 'scrapers')

    def load_data(self, reviews_file=None, metadata_file=None):
        """Load reviews and metadata from CSV files"""
        try:
            if reviews_file:
                self.reviews_df = pd.read_csv(reviews_file)
            if metadata_file:
                self.metadata_df = pd.read_csv(metadata_file)
            
            print(f"✅ Loaded {len(self.reviews_df)} reviews and {len(self.metadata_df)} movies")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def scrape_user_data(self, username: str) -> bool:
        """Scrape user data and integrate into reviews dataset"""
        try:
            print(f"🔍 Scraping data for {username}...")
            
            # Use the get_user_data function from the demo
            from scrapers.review_scraper import scrape_user_reviews
            import requests
            
            # Create a session for scraping
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            
            # Scrape user reviews
            reviews = scrape_user_reviews(username, session, max_reviews=500)
            
            if not reviews:
                print(f"❌ No films found for user: {username}")
                return False
            
            # Convert to DataFrame
            user_data = pd.DataFrame(reviews)
            print(f"✅ Scraped {len(user_data)} ratings for {username}")
            
            user_data_cleaned = self._clean_user_data(user_data, username)
            
            # Remove existing data for this user and add new data
            self.reviews_df = self.reviews_df[self.reviews_df['username'] != username]
            self.reviews_df = pd.concat([self.reviews_df, user_data_cleaned], ignore_index=True)
            
            print(f"✅ Integrated {len(user_data_cleaned)} ratings for {username}")
            return True
            
        except Exception as e:
            print(f"❌ Error scraping user data: {e}")
            return False

    def _clean_user_data(self, raw_data: pd.DataFrame, username: str) -> pd.DataFrame:
        """Clean scraped user data to match reviews dataset format"""
        cleaned_data = []
        skipped_no_match = 0
        skipped_no_rating = 0
        
        print(f"🧹 Cleaning {len(raw_data)} scraped ratings for {username}...")
        
        for _, row in raw_data.iterrows():
            title = row['title']
            year = row['year']
            rating = row.get('rating')
            
            # Skip if no rating (only include explicit ratings)
            if pd.isna(rating) or rating is None:
                skipped_no_rating += 1
                continue
            
            # Try to find matching movie in metadata using the correct column name
            movie_match = self.metadata_df[
                (self.metadata_df['title'] == title) & 
                (self.metadata_df['year'] == year)
            ]
            
            # If no exact match, try just title (more permissive)
            if len(movie_match) == 0:
                movie_match = self.metadata_df[self.metadata_df['title'] == title]
            
            # If still no match, try year +/- 1 (sometimes release dates differ)
            if len(movie_match) == 0 and pd.notna(year):
                for year_offset in [-1, 1]:
                    test_year = year + year_offset
                    movie_match = self.metadata_df[
                        (self.metadata_df['title'] == title) & 
                        (self.metadata_df['year'] == test_year)
                    ]
                    if len(movie_match) > 0:
                        break
            
            if len(movie_match) > 0:
                # Use first match if multiple found
                movie_info = movie_match.iloc[0]
                
                cleaned_data.append({
                    'username': username,
                    'tmdb_id': movie_info['tmdb_id'],
                    'title': movie_info['title'],
                    'year': movie_info['year'],
                    'rating': rating
                })
            else:
                skipped_no_match += 1
        
        result_df = pd.DataFrame(cleaned_data)
        
        print(f"   ✅ Integrated: {len(result_df)} ratings")
        print(f"   ⚠️  Skipped (no metadata found): {skipped_no_match}")
        print(f"   ⚠️  Skipped (no rating): {skipped_no_rating}")
        
        return result_df

    def train_model(self):
        """Train the streamlined hybrid recommendation model"""
        try:
            print("🚀 Training hybrid model...")
            
            # Clean data: remove NaN ratings
            print("🧹 Cleaning data...")
            initial_reviews = len(self.reviews_df)
            self.reviews_df = self.reviews_df.dropna(subset=['rating'])
            cleaned_reviews = len(self.reviews_df)
            if initial_reviews > cleaned_reviews:
                print(f"   Removed {initial_reviews - cleaned_reviews} reviews with missing ratings")
            
            # Prepare collaborative filtering data
            print("📊 Preparing collaborative filtering...")
            
            users = self.reviews_df['username'].unique()
            movies = self.reviews_df['tmdb_id'].unique()
            
            user_mapping = {user: idx for idx, user in enumerate(users)}
            movie_mapping = {movie: idx for idx, movie in enumerate(movies)}
            
            # Build interaction matrix efficiently
            n_users, n_movies = len(users), len(movies)
            interaction_matrix = lil_matrix((n_users, n_movies))
            
            for _, row in self.reviews_df.iterrows():
                user_idx = user_mapping[row['username']]
                movie_idx = movie_mapping[row['tmdb_id']]
                rating = row['rating']
                interaction_matrix[user_idx, movie_idx] = rating
            
            # Train collaborative filtering model using SVD
            print("🤝 Training collaborative filtering...")
            self.cf_model = TruncatedSVD(n_components=self.num_components, random_state=42)
            
            # Fit the model on the interaction matrix
            interaction_csr = interaction_matrix.tocsr()
            self.cf_model.fit(interaction_csr)
            
            # Get user and item embeddings
            self.user_embeddings = self.cf_model.transform(interaction_csr)
            self.item_embeddings = self.cf_model.components_.T
            
            # Store mappings
            self.user_mapping = user_mapping
            self.movie_mapping = movie_mapping
            self.interaction_matrix = interaction_csr
            
            # Calculate CF score distribution for debugging
            cf_scores = []
            for user_idx in range(min(100, n_users)):  # Sample for speed
                for movie_idx in range(min(100, n_movies)):
                    # Compute prediction using dot product of embeddings
                    score = np.dot(self.user_embeddings[user_idx], self.item_embeddings[movie_idx])
                    cf_scores.append(score)
            
            cf_mean = np.mean(cf_scores)
            cf_std = np.std(cf_scores)
            print(f"   CF score distribution: Mean={cf_mean:.3f}, Std={cf_std:.3f}")
            
            # Prepare content-based features
            print("🎬 Preparing content-based features...")
            self._prepare_content_features()
            
            print("✅ Model trained successfully!")
            print(f"   - Users: {n_users}, Movies: {n_movies}")
            print(f"   - Features: {len(self.content_features.columns) if hasattr(self, 'content_features') else 0}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False

    def _prepare_content_features(self):
        """Prepare content-based features for movies"""
        features = []
        
        # Process each movie
        for _, movie in self.metadata_df.iterrows():
            movie_features = {}
            
            # Basic features
            movie_features['year'] = movie.get('year', 2000)
            movie_features['vote_average'] = movie.get('vote_average', 6.0)
            movie_features['popularity'] = movie.get('popularity', 1.0)
            
            # Genre features (one-hot encoded)
            genres = str(movie.get('genres', '')).split('|') if pd.notna(movie.get('genres')) else []
            for genre in ['Action', 'Drama', 'Comedy', 'Thriller', 'Horror', 'Romance', 'Sci-Fi']:
                movie_features[f'genre_{genre.lower()}'] = 1 if genre in genres else 0
            
            # Director features
            directors = str(movie.get('directors_main', '')).split('|') if pd.notna(movie.get('directors_main')) else []
            movie_features['has_known_director'] = 1 if directors else 0
            
            # Cast features  
            cast = str(movie.get('cast_main', '')).split('|') if pd.notna(movie.get('cast_main')) else []
            movie_features['has_known_cast'] = 1 if cast else 0
            
            features.append(movie_features)
        
        self.content_features = pd.DataFrame(features)
        self.content_features.index = self.metadata_df['tmdb_id']

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-movie pair using matrix indices"""
        try:
            # Get collaborative filtering score using SVD embeddings
            if user_id < len(self.user_embeddings) and movie_id < len(self.item_embeddings):
                cf_score = np.dot(self.user_embeddings[user_id], self.item_embeddings[movie_id])
            else:
                cf_score = 6.0  # Default for unknown users/movies
            
            # Get content-based score (simple average) - need to convert movie_id to tmdb_id
            # Get the actual tmdb_id for this matrix index
            tmdb_ids = list(self.movie_mapping.keys())
            if movie_id < len(tmdb_ids):
                actual_tmdb_id = tmdb_ids[movie_id]
                if hasattr(self, 'content_features') and actual_tmdb_id in self.content_features.index:
                    cb_score = self.content_features.loc[actual_tmdb_id].mean()
                else:
                    cb_score = 6.0  # Default
            else:
                cb_score = 6.0  # Default
            
            # Combine scores
            final_score = self.alpha * cf_score + (1 - self.alpha) * cb_score
            
            # Scale to rating range (0.5 to 10.0)
            final_score = max(0.5, min(10.0, final_score))
            
            return final_score
            
        except Exception as e:
            return 6.0  # Default rating

    def predict_rating_for_user(self, username: str, tmdb_id: int) -> float:
        """Predict rating for a user-movie pair using actual identifiers"""
        try:
            if username not in self.user_mapping or tmdb_id not in self.movie_mapping:
                return 6.0  # Default for unknown users/movies
            
            user_idx = self.user_mapping[username]
            movie_idx = self.movie_mapping[tmdb_id]
            
            return self.predict_rating(user_idx, movie_idx)
            
        except Exception as e:
            return 6.0  # Default rating

    def get_recommendations(self, username: str, n_recommendations: int = None) -> pd.DataFrame:
        """Get movie recommendations for a user"""
        if n_recommendations is None:
            n_recommendations = self.num_recommendations
        
        if username not in self.user_mapping:
            print(f"❌ User {username} not found in training data")
            return pd.DataFrame()
        
        user_idx = self.user_mapping[username]
        
        # Get movies user hasn't rated
        user_movies = set(self.reviews_df[self.reviews_df['username'] == username]['tmdb_id'])
        candidate_movies = [mid for mid in self.movie_mapping.keys() if mid not in user_movies]
        
        # Score candidate movies
        scores = []
        for movie_id in candidate_movies:
            movie_idx = self.movie_mapping[movie_id]
            score = self.predict_rating(user_idx, movie_idx)
            scores.append((movie_id, score))
        
        # Sort by score and get top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_movies = scores[:n_recommendations]
        
        # Create recommendations DataFrame
        recommendations = []
        for movie_id, score in top_movies:
            movie_info = self.metadata_df[self.metadata_df['tmdb_id'] == movie_id].iloc[0]
            recommendations.append({
                'tmdb_id': movie_id,
                'title': movie_info['title'],
                'year': movie_info['year'],
                'predicted_rating': round(score, 2),
                'genres': movie_info.get('genres', ''),
                'directors': movie_info.get('directors_main', '')
            })
        
        return pd.DataFrame(recommendations)

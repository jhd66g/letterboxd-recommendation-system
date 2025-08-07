#!/usr/bin/env python3
"""
Streamlined Letterboxd Recommendation Engine V2

A clean, fast recommendation engine using hybrid collaborative filtering and content-based features.
Built specifically for the Letterboxd dataset format.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os
import time
import requests


class LetterboxdRecommendationEngine:
    def __init__(self, alpha=0.7, num_recommendations=25, num_components=25):
        """
        Initialize the recommendation engine
        
        Args:
            alpha: Weight for collaborative filtering vs content-based (0.7 = 70% collaborative)
            num_recommendations: Default number of recommendations to generate
            num_components: Number of SVD components for dimensionality reduction
        """
        # Model parameters
        self.alpha = alpha
        self.num_recommendations = num_recommendations
        self.num_components = num_components
        
        # Data containers
        self.reviews_df = None
        self.metadata_df = None
        self.cf_model = None
        self.user_mapping = {}
        self.movie_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_movie_mapping = {}
        self.user_embeddings = None
        self.item_embeddings = None
        self.interaction_matrix = None
        self.content_features = None
        
        # Stats
        self.training_time = 0
        self.is_trained = False

    def load_data(self, reviews_file, metadata_file):
        """Load reviews and metadata from CSV files"""
        try:
            print(f"📂 Loading data...")
            
            # Load datasets
            self.reviews_df = pd.read_csv(reviews_file)
            self.metadata_df = pd.read_csv(metadata_file)
            
            # Clean the reviews data
            initial_count = len(self.reviews_df)
            self.reviews_df = self.reviews_df.dropna(subset=['rating', 'tmdb_id', 'username'])
            self.reviews_df = self.reviews_df[self.reviews_df['rating'].notna()]
            cleaned_count = len(self.reviews_df)
            
            if initial_count > cleaned_count:
                print(f"   🧹 Removed {initial_count - cleaned_count} invalid reviews")
            
            print(f"✅ Loaded {len(self.reviews_df)} reviews and {len(self.metadata_df)} movies")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def train_model(self):
        """Train the hybrid recommendation model"""
        start_time = time.time()
        
        try:
            print("🚀 Training hybrid model...")
            
            # Create user mapping from reviews
            users = sorted(self.reviews_df['username'].unique())
            self.user_mapping = {user: idx for idx, user in enumerate(users)}
            self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
            
            # Create movie mapping from ALL metadata (not just reviewed movies)
            all_movies = sorted(self.metadata_df['tmdb_id'].dropna().unique())
            self.movie_mapping = {movie: idx for idx, movie in enumerate(all_movies)}
            self.reverse_movie_mapping = {idx: movie for movie, idx in self.movie_mapping.items()}
            
            n_users, n_movies = len(users), len(all_movies)
            print(f"   👥 Users: {n_users}, 🎬 Movies: {n_movies} (full catalog)")
            
            # Build interaction matrix
            print("📊 Building interaction matrix...")
            interaction_matrix = csr_matrix((n_users, n_movies))
            
            # Use more efficient method for building sparse matrix
            row_indices = []
            col_indices = []
            ratings = []
            
            for _, row in self.reviews_df.iterrows():
                username = row['username']
                tmdb_id = row['tmdb_id']
                rating = float(row['rating'])
                
                # Only include if both user and movie are in our mappings
                if username in self.user_mapping and tmdb_id in self.movie_mapping:
                    user_idx = self.user_mapping[username]
                    movie_idx = self.movie_mapping[tmdb_id]
                    
                    row_indices.append(user_idx)
                    col_indices.append(movie_idx)
                    ratings.append(rating)
            
            self.interaction_matrix = csr_matrix(
                (ratings, (row_indices, col_indices)), 
                shape=(n_users, n_movies)
            )
            
            print(f"   📊 Built matrix: {n_users}x{n_movies} with {len(ratings)} ratings")
            
            # Train collaborative filtering model
            print("🤝 Training collaborative filtering (SVD)...")
            self.cf_model = TruncatedSVD(
                n_components=min(self.num_components, min(n_users, n_movies) - 1),
                random_state=42
            )
            
            # Fit SVD on interaction matrix
            self.cf_model.fit(self.interaction_matrix)
            
            # Get embeddings
            self.user_embeddings = self.cf_model.transform(self.interaction_matrix)
            self.item_embeddings = self.cf_model.components_.T
            
            print(f"   ✅ SVD trained with {self.cf_model.n_components} components")
            
            # Prepare content-based features
            print("🎬 Preparing content features...")
            self._prepare_content_features()
            
            self.training_time = time.time() - start_time
            self.is_trained = True
            
            print(f"✅ Model trained successfully in {self.training_time:.2f}s!")
            print(f"   📊 Matrix density: {(len(ratings) / (n_users * n_movies) * 100):.3f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_content_features(self):
        """Prepare content-based features from movie metadata"""
        features = []
        
        # Create a mapping from tmdb_id to movie info for faster lookup
        metadata_dict = {}
        for _, movie in self.metadata_df.iterrows():
            metadata_dict[movie['tmdb_id']] = movie
        
        # Process movies that are in our interaction matrix
        for movie_id in self.movie_mapping.keys():
            movie_features = {}
            
            if movie_id in metadata_dict:
                movie = metadata_dict[movie_id]
                
                # Numerical features
                movie_features['year'] = float(movie.get('year', 2000))
                movie_features['vote_average'] = float(movie.get('vote_average', 6.0))
                movie_features['popularity'] = float(movie.get('popularity', 1.0))
                movie_features['vote_count'] = float(movie.get('vote_count', 0))
                
                # Genre features (one-hot encoding)
                genres_str = str(movie.get('genres', ''))
                for genre in ['Action', 'Drama', 'Comedy', 'Thriller', 'Horror', 'Romance', 'Sci-Fi', 'Animation']:
                    movie_features[f'genre_{genre.lower()}'] = 1 if genre in genres_str else 0
                
                # Director and cast features
                movie_features['has_directors'] = 1 if pd.notna(movie.get('directors_main')) else 0
                movie_features['has_cast'] = 1 if pd.notna(movie.get('cast_main')) else 0
                
            else:
                # Default values for movies not in metadata
                movie_features = {
                    'year': 2000.0, 'vote_average': 6.0, 'popularity': 1.0, 'vote_count': 0.0,
                    'genre_action': 0, 'genre_drama': 0, 'genre_comedy': 0, 'genre_thriller': 0,
                    'genre_horror': 0, 'genre_romance': 0, 'genre_sci-fi': 0, 'genre_animation': 0,
                    'has_directors': 0, 'has_cast': 0
                }
            
            features.append(movie_features)
        
        # Create DataFrame with movie indices as index
        self.content_features = pd.DataFrame(features)
        self.content_features.index = list(self.movie_mapping.keys())
        
        # Normalize numerical features
        for col in ['year', 'vote_average', 'popularity', 'vote_count']:
            if col in self.content_features.columns:
                mean_val = self.content_features[col].mean()
                std_val = self.content_features[col].std()
                if std_val > 0:
                    self.content_features[col] = (self.content_features[col] - mean_val) / std_val
        
        print(f"   ✅ Content features: {len(self.content_features.columns)} features")

    def predict_rating(self, username, tmdb_id):
        """Predict rating for a user-movie pair"""
        if not self.is_trained:
            return 6.0
        
        try:
            # Check if user and movie exist in training data
            if username not in self.user_mapping or tmdb_id not in self.movie_mapping:
                return 6.0  # Default rating
            
            user_idx = self.user_mapping[username]
            movie_idx = self.movie_mapping[tmdb_id]
            
            # Get collaborative filtering score
            cf_score = np.dot(self.user_embeddings[user_idx], self.item_embeddings[movie_idx])
            
            # Get content-based score
            if tmdb_id in self.content_features.index:
                cb_score = self.content_features.loc[tmdb_id].mean()
            else:
                cb_score = 6.0
            
            # Combine scores
            final_score = self.alpha * cf_score + (1 - self.alpha) * cb_score
            
            # Scale to reasonable rating range (1-10)
            final_score = np.clip(final_score, 1.0, 10.0)
            
            return float(final_score)
            
        except Exception as e:
            return 6.0

    def get_recommendations(self, username, n_recommendations=None):
        """Generate movie recommendations for a user"""
        if n_recommendations is None:
            n_recommendations = self.num_recommendations
        
        if not self.is_trained:
            print("❌ Model not trained yet")
            return pd.DataFrame()
        
        if username not in self.user_mapping:
            print(f"❌ User '{username}' not found in training data")
            return pd.DataFrame()
        
        user_idx = self.user_mapping[username]
        
        # Get movies the user has already rated
        user_rated_movies = set(
            self.reviews_df[self.reviews_df['username'] == username]['tmdb_id'].values
        )
        
        # Get candidate movies (not rated by user)
        candidate_movies = [
            tmdb_id for tmdb_id in self.movie_mapping.keys() 
            if tmdb_id not in user_rated_movies
        ]
        
        if not candidate_movies:
            print(f"❌ No new movies to recommend for {username}")
            return pd.DataFrame()
        
        # Score all candidate movies
        scores = []
        for tmdb_id in candidate_movies:
            score = self.predict_rating(username, tmdb_id)
            scores.append((tmdb_id, score))
        
        # Sort by score and get top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_movies = scores[:n_recommendations]
        
        # Create recommendations DataFrame with metadata
        recommendations = []
        metadata_dict = {row['tmdb_id']: row for _, row in self.metadata_df.iterrows()}
        
        for tmdb_id, score in top_movies:
            if tmdb_id in metadata_dict:
                movie = metadata_dict[tmdb_id]
                recommendations.append({
                    'tmdb_id': tmdb_id,
                    'title': movie['title'],
                    'year': movie['year'],
                    'predicted_rating': round(score, 2),
                    'genres': movie.get('genres', ''),
                    'directors': movie.get('directors_main', ''),
                    'vote_average': movie.get('vote_average', 0),
                    'popularity': movie.get('popularity', 0)
                })
        
        return pd.DataFrame(recommendations)

    def scrape_and_add_user(self, username, max_reviews=500):
        """Scrape a user's data and add to the dataset"""
        try:
            print(f"🔍 Scraping data for {username}...")
            
            from scrapers.review_scraper import scrape_user_reviews
            
            # Create session
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            
            # Scrape reviews
            reviews = scrape_user_reviews(username, session, max_reviews=max_reviews)
            
            if not reviews:
                print(f"❌ No reviews found for {username}")
                return False
            
            # Convert to DataFrame and clean
            user_df = pd.DataFrame(reviews)
            user_df = user_df.dropna(subset=['rating'])
            
            # Match with our metadata
            matched_reviews = []
            metadata_dict = {(row['title'], row['year']): row['tmdb_id'] 
                           for _, row in self.metadata_df.iterrows()}
            
            for _, review in user_df.iterrows():
                title = review['title']
                year = review['year']
                rating = review['rating']
                
                # Try to find movie in our metadata
                tmdb_id = None
                if (title, year) in metadata_dict:
                    tmdb_id = metadata_dict[(title, year)]
                else:
                    # Try without year
                    for (meta_title, meta_year), meta_id in metadata_dict.items():
                        if meta_title == title:
                            tmdb_id = meta_id
                            year = meta_year
                            break
                
                if tmdb_id:
                    matched_reviews.append({
                        'username': username,
                        'tmdb_id': tmdb_id,
                        'rating': rating,
                        'title': title,
                        'year': year
                    })
            
            if matched_reviews:
                # Add to our reviews dataset
                new_reviews_df = pd.DataFrame(matched_reviews)
                
                # Remove existing reviews for this user
                self.reviews_df = self.reviews_df[self.reviews_df['username'] != username]
                
                # Add new reviews
                self.reviews_df = pd.concat([self.reviews_df, new_reviews_df], ignore_index=True)
                
                print(f"✅ Added {len(matched_reviews)} reviews for {username}")
                print(f"   📊 Total reviews now: {len(self.reviews_df)}")
                
                return True
            else:
                print(f"❌ No matching movies found for {username}")
                return False
                
        except Exception as e:
            print(f"❌ Error scraping user data: {e}")
            return False

    def get_user_stats(self, username):
        """Get statistics for a user"""
        if username not in self.user_mapping:
            return None
        
        user_reviews = self.reviews_df[self.reviews_df['username'] == username]
        
        return {
            'username': username,
            'num_reviews': len(user_reviews),
            'avg_rating': user_reviews['rating'].mean(),
            'rating_std': user_reviews['rating'].std(),
            'min_rating': user_reviews['rating'].min(),
            'max_rating': user_reviews['rating'].max()
        }

    def get_model_stats(self):
        """Get overall model statistics"""
        if not self.is_trained:
            return {}
        
        return {
            'num_users': len(self.user_mapping),
            'num_movies': len(self.movie_mapping),
            'num_reviews': len(self.reviews_df),
            'density': len(self.reviews_df) / (len(self.user_mapping) * len(self.movie_mapping)),
            'avg_rating': self.reviews_df['rating'].mean(),
            'training_time': self.training_time,
            'svd_components': self.cf_model.n_components if self.cf_model else 0,
            'content_features': len(self.content_features.columns) if self.content_features is not None else 0
        }

#!/usr/bin/env python3
"""
Letterboxd Hybrid Recommendation Engine

A clean, simple implementation that combines collaborative and content-based filtering.
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ML libraries
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix, hstack

class LetterboxdRecommendationEngine:
    """Simple hybrid recommendation engine for Letterboxd users"""
    
    def __init__(self, alpha=0.7, num_recommendations=25, num_components=25, 
                 l2_reg=1.0, cv_folds=5, optimize_alpha=True):
        self.alpha = alpha
        self.num_recommendations = num_recommendations
        self.num_components = num_components
        self.l2_reg = l2_reg
        self.cv_folds = cv_folds
        self.optimize_alpha = optimize_alpha
        
        # Directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.scrapers_dir = os.path.join(self.base_dir, 'scrapers')
        self.output_dir = os.path.join(self.base_dir, 'output')
        
        # Data files
        self.reviews_file = os.path.join(self.data_dir, 'cleaned_reviews_v5.csv')
        self.metadata_file = os.path.join(self.data_dir, 'cleaned_movie_metadata_v5.csv')
        
        # Model components
        self.model = None
        self.reviews_df = None
        self.metadata_df = None
        self.content_model = None
        self.optimal_alpha = None
        self.optimal_l2_reg = None
        
    def load_data(self):
        """Load cleaned data"""
        try:
            print("📖 Loading data...")
            self.reviews_df = pd.read_csv(self.reviews_file)
            self.metadata_df = pd.read_csv(self.metadata_file)
            
            print(f"✅ Loaded {len(self.reviews_df)} reviews and {len(self.metadata_df)} movies")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def scrape_user_data(self, username: str) -> bool:
        """Scrape user data and integrate into reviews dataset"""
        try:
            print(f"🔍 Scraping data for {username}...")
            
            # Run scraper
            scraper_path = os.path.join(self.scrapers_dir, 'rating_scraper_pro.py')
            output_file = os.path.join(self.data_dir, f'{username}_ratings.csv')
            
            result = subprocess.run([
                sys.executable, scraper_path, username, '--output', output_file
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"❌ Scraper failed: {result.stderr}")
                return False
            
            # Load scraped data
            user_data = pd.read_csv(output_file)
            print(f"✅ Scraped {len(user_data)} ratings for {username}")
            
            # Clean and format to match reviews dataset
            user_data_cleaned = self._clean_user_data(user_data, username)
            
            # Remove existing data for this user
            self.reviews_df = self.reviews_df[self.reviews_df['username'] != username]
            
            # Add new data
            self.reviews_df = pd.concat([self.reviews_df, user_data_cleaned], ignore_index=True)
            
            # Clean up temp file
            os.remove(output_file)
            
            print(f"✅ Integrated {len(user_data_cleaned)} ratings for {username}")
            return True
            
        except Exception as e:
            print(f"❌ Error scraping user data: {e}")
            return False
    
    def _clean_user_data(self, raw_data: pd.DataFrame, username: str) -> pd.DataFrame:
        """Clean scraped user data to match reviews dataset format"""
        cleaned_data = []
        
        for _, row in raw_data.iterrows():
            # Match with metadata using title and year
            title = row['title']
            year = row['year']
            
            # Try to find matching movie in metadata
            movie_match = self.metadata_df[
                (self.metadata_df['title'] == title) & 
                (self.metadata_df['release_year'] == year)
            ]
            
            # If no exact match, try just title
            if len(movie_match) == 0:
                movie_match = self.metadata_df[self.metadata_df['title'] == title]
            
            if len(movie_match) > 0:
                # Use the first match
                movie_info = movie_match.iloc[0]
                
                # Extract rating or mark as implicit feedback
                rating = row.get('rating', 0)
                has_rating = rating > 0
                
                cleaned_data.append({
                    'username': username,
                    'tmdb_id': movie_info['tmdb_id'],
                    'title': title,
                    'rating': rating if has_rating else 0,
                    'has_rating': has_rating
                })
        
        return pd.DataFrame(cleaned_data)
    
    def train_model(self):
        """Train the hybrid recommendation model"""
        try:
            print("🚀 Training hybrid model...")
            
            # Prepare collaborative filtering data
            print("📊 Preparing collaborative filtering...")
            
            # Create user-movie interaction matrix
            users = self.reviews_df['username'].unique()
            movies = self.reviews_df['tmdb_id'].unique()
            
            user_mapping = {user: idx for idx, user in enumerate(users)}
            movie_mapping = {movie: idx for idx, movie in enumerate(movies)}
            
            # Build interaction matrix
            n_users, n_movies = len(users), len(movies)
            interaction_matrix = csr_matrix((n_users, n_movies))
            
            for _, row in self.reviews_df.iterrows():
                user_idx = user_mapping[row['username']]
                movie_idx = movie_mapping[row['tmdb_id']]
                
                # Use explicit rating if available, otherwise implicit (0.1)
                rating = row['rating'] / 10.0 if row['has_rating'] else 0.1
                interaction_matrix[user_idx, movie_idx] = rating
            
            # Train SVD model
            print("🤝 Training collaborative filtering...")
            svd_model = TruncatedSVD(n_components=self.num_components, random_state=42)
            user_factors = svd_model.fit_transform(interaction_matrix)
            movie_factors = svd_model.components_
            
            # Prepare content-based features
            print("🎬 Preparing content-based features...")
            
            # Text features (TF-IDF)
            text_features = []
            for feature in ['genres', 'directors', 'cast', 'keywords', 'production_companies']:
                if feature in self.metadata_df.columns:
                    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
                    feature_matrix = vectorizer.fit_transform(
                        self.metadata_df[feature].fillna('').astype(str)
                    )
                    text_features.append(feature_matrix)
            
            # Categorical features
            categorical_features = []
            for feature in ['year_bucket', 'runtime_bucket', 'budget_bucket', 'revenue_bucket', 'original_language']:
                if feature in self.metadata_df.columns:
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(self.metadata_df[feature].fillna('unknown'))
                    # One-hot encode
                    n_categories = len(encoder.classes_)
                    category_matrix = csr_matrix((len(encoded), n_categories))
                    for i, cat in enumerate(encoded):
                        category_matrix[i, cat] = 1
                    categorical_features.append(category_matrix)
            
            # Combine all features
            all_features = text_features + categorical_features
            if all_features:
                combined_features = hstack(all_features)
            else:
                combined_features = csr_matrix((len(self.metadata_df), 1))
            
            # Calculate movie similarity matrix
            movie_similarity = cosine_similarity(combined_features)
            
            # Store model
            self.model = {
                'svd_model': svd_model,
                'user_factors': user_factors,
                'movie_factors': movie_factors,
                'movie_similarity': movie_similarity,
                'interaction_matrix': interaction_matrix,
                'user_mapping': user_mapping,
                'movie_mapping': movie_mapping,
                'users': users,
                'movies': movies
            }
            
            print(f"✅ Model trained successfully!")
            print(f"   - Users: {len(users)}, Movies: {len(movies)}")
            print(f"   - SVD components: {self.num_components}")
            print(f"   - Features: {combined_features.shape[1]}")
            
            # Hyperparameter optimization
            if self.optimize_alpha:
                self._optimize_hyperparameters(interaction_matrix, combined_features)
            
            # Train content model with L2 regularization
            self.content_model = self._train_content_model(interaction_matrix, combined_features)
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def _optimize_hyperparameters(self, interaction_matrix, combined_features):
        """Optimize alpha and L2 regularization using cross-validation"""
        try:
            print("🔧 Optimizing hyperparameters with cross-validation...")
            
            # Create validation set
            n_users, n_movies = interaction_matrix.shape
            
            # Sample validation interactions
            validation_interactions = []
            for user_idx in range(min(50, n_users)):  # Sample up to 50 users
                user_ratings = interaction_matrix[user_idx].toarray().flatten()
                rated_movies = np.where(user_ratings > 0)[0]
                
                if len(rated_movies) >= 5:  # Need at least 5 ratings for validation
                    # Hold out 20% of ratings for validation
                    n_holdout = max(1, len(rated_movies) // 5)
                    holdout_indices = np.random.choice(rated_movies, n_holdout, replace=False)
                    
                    for movie_idx in holdout_indices:
                        validation_interactions.append({
                            'user_idx': user_idx,
                            'movie_idx': movie_idx,
                            'rating': user_ratings[movie_idx]
                        })
            
            if len(validation_interactions) < 10:
                print("⚠️  Not enough validation data, using default parameters")
                self.optimal_alpha = self.alpha
                self.optimal_l2_reg = self.l2_reg
                return
            
            # Grid search for optimal parameters
            alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
            l2_values = [0.1, 0.5, 1.0, 2.0, 5.0]
            
            best_alpha = self.alpha
            best_l2 = self.l2_reg
            best_score = float('inf')
            
            print(f"   Testing {len(alpha_values)} alpha values and {len(l2_values)} L2 values...")
            
            for alpha in alpha_values:
                for l2_reg in l2_values:
                    # Calculate validation score
                    score = self._validate_parameters(
                        interaction_matrix, combined_features, 
                        validation_interactions, alpha, l2_reg
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_alpha = alpha
                        best_l2 = l2_reg
            
            self.optimal_alpha = best_alpha
            self.optimal_l2_reg = best_l2
            
            print(f"✅ Optimal parameters: alpha={best_alpha:.2f}, L2={best_l2:.2f}, RMSE={best_score:.3f}")
            
        except Exception as e:
            print(f"⚠️  Hyperparameter optimization failed: {e}")
            self.optimal_alpha = self.alpha
            self.optimal_l2_reg = self.l2_reg
    
    def _validate_parameters(self, interaction_matrix, combined_features, 
                           validation_interactions, alpha, l2_reg):
        """Validate parameters on held-out data"""
        try:
            # Train models with current parameters
            svd_model = TruncatedSVD(n_components=self.num_components, random_state=42)
            user_factors = svd_model.fit_transform(interaction_matrix)
            movie_factors = svd_model.components_
            
            # Train content-based model with L2 regularization
            content_model = Ridge(alpha=l2_reg, random_state=42)
            
            # Prepare training data for content model
            X_content = []
            y_content = []
            
            for interaction in validation_interactions:
                user_idx = interaction['user_idx']
                movie_idx = interaction['movie_idx']
                rating = interaction['rating']
                
                # Get user's collaborative features
                user_features = user_factors[user_idx]
                
                # Get movie's content features
                movie_features = combined_features[movie_idx].toarray().flatten()
                
                # Combine features
                combined_vector = np.concatenate([user_features, movie_features])
                
                X_content.append(combined_vector)
                y_content.append(rating)
            
            if len(X_content) == 0:
                return float('inf')
            
            # Train content model
            content_model.fit(X_content, y_content)
            
            # Calculate validation RMSE
            predictions = []
            actuals = []
            
            for interaction in validation_interactions:
                user_idx = interaction['user_idx']
                movie_idx = interaction['movie_idx']
                actual_rating = interaction['rating']
                
                # Collaborative score
                cf_score = np.dot(user_factors[user_idx], movie_factors[:, movie_idx])
                
                # Content-based score
                user_features = user_factors[user_idx]
                movie_features = combined_features[movie_idx].toarray().flatten()
                combined_vector = np.concatenate([user_features, movie_features])
                cb_score = content_model.predict([combined_vector])[0]
                
                # Hybrid score
                hybrid_score = alpha * cf_score + (1 - alpha) * cb_score
                
                predictions.append(hybrid_score)
                actuals.append(actual_rating)
            
            return np.sqrt(mean_squared_error(actuals, predictions))
            
        except Exception as e:
            return float('inf')
    
    def _train_content_model(self, interaction_matrix, combined_features):
        """Train content-based model with L2 regularization"""
        try:
            print("🎯 Training content-based model with L2 regularization...")
            
            # Prepare training data
            X_train = []
            y_train = []
            
            n_users, n_movies = interaction_matrix.shape
            
            # Sample training interactions
            for user_idx in range(min(100, n_users)):  # Sample up to 100 users
                user_ratings = interaction_matrix[user_idx].toarray().flatten()
                rated_movies = np.where(user_ratings > 0)[0]
                
                if len(rated_movies) > 0:
                    # Sample up to 50 movies per user
                    sample_size = min(50, len(rated_movies))
                    sampled_movies = np.random.choice(rated_movies, sample_size, replace=False)
                    
                    for movie_idx in sampled_movies:
                        # Get user's collaborative features (from SVD)
                        user_features = self.model['user_factors'][user_idx]
                        
                        # Get movie's content features
                        movie_features = combined_features[movie_idx].toarray().flatten()
                        
                        # Combine features
                        combined_vector = np.concatenate([user_features, movie_features])
                        
                        X_train.append(combined_vector)
                        y_train.append(user_ratings[movie_idx])
            
            if len(X_train) == 0:
                print("⚠️  No training data for content model")
                return None
            
            # Train Ridge regression model
            content_model = Ridge(alpha=self.optimal_l2_reg, random_state=42)
            content_model.fit(X_train, y_train)
            
            print(f"✅ Content model trained on {len(X_train)} interactions")
            return content_model
            
        except Exception as e:
            print(f"⚠️  Content model training failed: {e}")
            return None
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user-movie pair using hybrid approach"""
        # Collaborative filtering score
        cf_score = np.dot(self.model['user_factors'][user_id], 
                         self.model['movie_factors'][:, movie_id])
        
        # Content-based score
        user_ratings = self.model['interaction_matrix'][user_id].toarray().flatten()
        rated_movies = np.where(user_ratings > 0)[0]
        
        if len(rated_movies) > 0:
            # Average similarity to user's rated movies
            similarities = self.model['movie_similarity'][movie_id, rated_movies]
            ratings = user_ratings[rated_movies]
            
            if np.sum(similarities) > 0:
                cb_score = np.average(ratings, weights=similarities)
            else:
                cb_score = np.mean(ratings)
        else:
            cb_score = 0.5  # Default for new users
        
        # Hybrid combination
        hybrid_score = self.alpha * cf_score + (1 - self.alpha) * cb_score
        
        # Convert to 1-10 scale
        return np.clip(hybrid_score * 10, 1, 10)
    
    def generate_recommendations(self, username: str) -> Optional[pd.DataFrame]:
        """Generate recommendations for a user"""
        try:
            print(f"🎬 Generating recommendations for {username}...")
            
            # Scrape and integrate user data
            if not self.scrape_user_data(username):
                return None
            
            # Train model with updated data
            if not self.train_model():
                return None
            
            # Check if user exists
            if username not in self.model['user_mapping']:
                print(f"❌ User {username} not found after scraping")
                return None
            
            user_id = self.model['user_mapping'][username]
            
            # Get user's rated movies
            user_ratings = self.model['interaction_matrix'][user_id].toarray().flatten()
            rated_movies = set(np.where(user_ratings > 0)[0])
            
            # Generate predictions for unrated movies
            predictions = []
            for movie_idx in range(len(self.model['movies'])):
                if movie_idx not in rated_movies:
                    tmdb_id = self.model['movies'][movie_idx]
                    predicted_rating = self.predict_rating(user_id, movie_idx)
                    
                    # Get movie metadata
                    movie_info = self.metadata_df[self.metadata_df['tmdb_id'] == tmdb_id]
                    if len(movie_info) > 0:
                        movie_info = movie_info.iloc[0]
                        
                        predictions.append({
                            'tmdb_id': tmdb_id,
                            'title': movie_info.get('title', 'Unknown'),
                            'year': movie_info.get('release_year', 'Unknown'),
                            'genres': movie_info.get('genres', 'Unknown'),
                            'directors': movie_info.get('directors', 'Unknown'),
                            'predicted_rating': predicted_rating
                        })
            
            # Sort by predicted rating and take top recommendations
            predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
            top_predictions = predictions[:self.num_recommendations]
            
            # Create DataFrame
            recommendations_df = pd.DataFrame(top_predictions)
            recommendations_df['rank'] = range(1, len(recommendations_df) + 1)
            
            # Save recommendations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f'recommendations_{username}_{timestamp}.csv')
            recommendations_df.to_csv(output_file, index=False)
            
            print(f"✅ Generated {len(recommendations_df)} recommendations")
            print(f"📁 Saved to: {output_file}")
            
            return recommendations_df
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return None

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python recommendation_engine.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    
    # Create engine
    engine = LetterboxdRecommendationEngine()
    
    # Load data
    if not engine.load_data():
        sys.exit(1)
    
    # Generate recommendations
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

if __name__ == "__main__":
    main()

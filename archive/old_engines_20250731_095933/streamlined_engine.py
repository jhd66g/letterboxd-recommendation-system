#!/usr/bin/env python3
"""
Letterboxd Streamlined Recommendation Engine

Production-ready hybrid recommendation system optimized for explicit ratings only.
Combines collaborative filtering (SVD) and content-based filtering with separated features.

Based on README specifications:
- 70% collaborative filtering + 30% content-based filtering
- Explicit ratings only (removes implicit feedback bias)
- Separated feature engineering (main + other columns)
- TruncatedSVD with 25 components
- TF-IDF vectorization for content features
- Auto-detection of latest pipeline data
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ML libraries
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix, hstack


class LetterboxdRecommendationEngine:
    """Streamlined hybrid recommendation engine for Letterboxd users"""
    
    def __init__(self, alpha=0.7, num_recommendations=25, num_components=25):
        """
        Initialize the recommendation engine with optimized parameters from README
        
        Args:
            alpha: Weight for collaborative filtering (0.7 = 70% CF, 30% CB)
            num_recommendations: Number of recommendations to generate
            num_components: SVD components for dimensionality reduction
        """
        # Fixed optimal parameters from README
        self.alpha = alpha
        self.num_recommendations = num_recommendations
        self.num_components = num_components
        
        # Directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.scrapers_dir = os.path.join(self.base_dir, 'scrapers')
        self.output_dir = os.path.join(self.base_dir, 'output')
        
        # Auto-detect latest pipeline data
        self.reviews_file, self.metadata_file = self._find_latest_dataset()
        
        # Model components
        self.model = None
        self.reviews_df = None
        self.metadata_df = None
        self.training_time = 0
        
    def _find_latest_dataset(self) -> Tuple[str, str]:
        """Find the latest dataset from pipeline or fallback to existing"""
        
        # Look for latest production pipeline data (priority order)
        patterns = [
            ('production_cleaned_reviews_*.csv', 'production_cleaned_metadata_*.csv'),
            ('final_reviews_improved_*.csv', 'final_metadata_improved_*.csv'),
            ('cleaned_reviews_explicit_*.csv', 'final_metadata_improved_*.csv'),
            ('cleaned_reviews_explicit_*.csv', 'cleaned_movie_metadata_*.csv')
        ]
        
        for reviews_pattern, metadata_pattern in patterns:
            reviews_files = glob.glob(os.path.join(self.data_dir, reviews_pattern))
            metadata_files = glob.glob(os.path.join(self.data_dir, metadata_pattern))
            
            if reviews_files and metadata_files:
                # Use most recent files
                latest_reviews = max(reviews_files, key=os.path.getctime)
                latest_metadata = max(metadata_files, key=os.path.getctime)
                
                print(f"🎯 Auto-detected latest dataset:")
                print(f"   Reviews: {os.path.basename(latest_reviews)}")
                print(f"   Metadata: {os.path.basename(latest_metadata)}")
                
                return latest_reviews, latest_metadata
        
        # If no files found, return placeholders (will be created by pipeline)
        print("⚠️  No dataset found - run pipeline first or specify files manually")
        return (
            os.path.join(self.data_dir, 'production_cleaned_reviews_latest.csv'),
            os.path.join(self.data_dir, 'production_cleaned_metadata_latest.csv')
        )
    
    def load_data(self, reviews_file: str = None, metadata_file: str = None) -> bool:
        """Load cleaned data from specified files or auto-detected latest"""
        try:
            # Use specified files or auto-detected ones
            reviews_path = reviews_file or self.reviews_file
            metadata_path = metadata_file or self.metadata_file
            
            if not os.path.exists(reviews_path) or not os.path.exists(metadata_path):
                print(f"❌ Data files not found:")
                print(f"   Reviews: {reviews_path}")
                print(f"   Metadata: {metadata_path}")
                print(f"   Run the pipeline first: python data_cleaning/full_production_pipeline.py")
                return False
            
            print("📖 Loading data...")
            self.reviews_df = pd.read_csv(reviews_path)
            self.metadata_df = pd.read_csv(metadata_path)
            
            # Validate explicit ratings only
            initial_reviews = len(self.reviews_df)
            self.reviews_df = self.reviews_df[self.reviews_df['rating'] > 0.5]  # Remove implicit ratings
            final_reviews = len(self.reviews_df)
            
            if initial_reviews > final_reviews:
                print(f"   🧹 Filtered {initial_reviews - final_reviews} implicit ratings")
            
            print(f"✅ Loaded {len(self.reviews_df):,} explicit reviews and {len(self.metadata_df):,} movies")
            print(f"   Users: {self.reviews_df['username'].nunique():,}")
            print(f"   Avg rating: {self.reviews_df['rating'].mean():.2f}")
            print(f"   Rating range: {self.reviews_df['rating'].min():.1f} - {self.reviews_df['rating'].max():.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def train_model(self) -> bool:
        """Train the streamlined hybrid recommendation model"""
        start_time = time.time()
        
        try:
            print("🚀 Training hybrid model...")
            
            # Prepare collaborative filtering data
            print("📊 Preparing collaborative filtering...")
            
            users = sorted(self.reviews_df['username'].unique())
            movies = sorted(self.reviews_df['tmdb_id'].unique())
            
            user_mapping = {user: idx for idx, user in enumerate(users)}
            movie_mapping = {movie: idx for idx, movie in enumerate(movies)}
            
            # Build interaction matrix efficiently
            n_users, n_movies = len(users), len(movies)
            print(f"   👥 Users: {n_users:,}, 🎬 Movies: {n_movies:,}")
            
            interaction_matrix = lil_matrix((n_users, n_movies))
            
            for _, row in self.reviews_df.iterrows():
                user_idx = user_mapping[row['username']]
                movie_idx = movie_mapping[row['tmdb_id']]
                # Normalize ratings to 0-1 scale for SVD
                rating = row['rating'] / 10.0
                interaction_matrix[user_idx, movie_idx] = rating
            
            # Convert to CSR for SVD
            interaction_matrix = interaction_matrix.tocsr()
            density = (interaction_matrix.nnz / (n_users * n_movies)) * 100
            print(f"   📊 Matrix density: {density:.3f}%")
            
            # Train SVD model
            print("🤝 Training collaborative filtering (SVD)...")
            svd_model = TruncatedSVD(n_components=self.num_components, random_state=42)
            user_factors = svd_model.fit_transform(interaction_matrix)
            movie_factors = svd_model.components_
            
            # Calculate CF normalization parameters
            cf_scores_sample = []
            for i in range(min(100, len(user_factors))):
                for j in range(min(100, movie_factors.shape[1])):
                    cf_scores_sample.append(np.dot(user_factors[i], movie_factors[:, j]))
            
            cf_mean = np.mean(cf_scores_sample)
            cf_std = np.std(cf_scores_sample)
            print(f"   CF score distribution: Mean={cf_mean:.3f}, Std={cf_std:.3f}")
            
            # Prepare content-based features with separated engineering
            print("🎬 Preparing content-based features...")
            content_features = self._prepare_content_features()
            
            # Calculate movie similarity matrix
            print("🔗 Computing content similarity...")
            movie_similarity = cosine_similarity(content_features)
            
            # Store model
            self.model = {
                'user_factors': user_factors,
                'movie_factors': movie_factors,
                'movie_similarity': movie_similarity,
                'interaction_matrix': interaction_matrix,
                'user_mapping': user_mapping,
                'movie_mapping': movie_mapping,
                'users': users,
                'movies': movies,
                'cf_mean': cf_mean,
                'cf_std': cf_std,
                'content_features': content_features
            }
            
            self.training_time = time.time() - start_time
            
            print(f"✅ Model trained successfully in {self.training_time:.2f}s!")
            print(f"   - SVD components: {self.num_components}")
            print(f"   - Content features: {content_features.shape[1]:,}")
            print(f"   - Alpha (CF weight): {self.alpha}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_content_features(self):
        """Prepare content-based features with separated engineering as per README"""
        text_features = []
        
        # Core features with separated main/other engineering
        feature_configs = [
            ('genres', 300, 1.0),  # Full weight for genres
            ('directors_main', 200, 1.0),  # Main directors
            ('directors_other', 50, 0.3),  # Other directors (lower weight)
            ('cast_main', 200, 1.0),  # Main cast
            ('cast_other', 50, 0.3),  # Other cast (lower weight)
            ('keywords', 300, 0.8),  # Keywords with slight reduction
        ]
        
        # Process text features
        for feature, max_feat, weight in feature_configs:
            if feature in self.metadata_df.columns:
                try:
                    vectorizer = TfidfVectorizer(
                        max_features=max_feat, 
                        stop_words='english',
                        min_df=2,  # Remove very rare features
                        max_df=0.95  # Remove very common features
                    )
                    feature_matrix = vectorizer.fit_transform(
                        self.metadata_df[feature].fillna('').astype(str)
                    )
                    
                    # Apply weight to separated features
                    if weight != 1.0:
                        feature_matrix = feature_matrix * weight
                    
                    text_features.append(feature_matrix)
                    print(f"   ✅ {feature}: {feature_matrix.shape[1]} features (weight: {weight})")
                    
                except Exception as e:
                    print(f"   ⚠️  Skipped {feature}: {e}")
        
        # Original language encoding
        if 'original_language' in self.metadata_df.columns:
            try:
                # Group rare languages as 'other' 
                lang_counts = self.metadata_df['original_language'].value_counts()
                common_langs = lang_counts[lang_counts >= 10].index.tolist()
                
                languages = self.metadata_df['original_language'].fillna('unknown')
                languages = languages.apply(lambda x: x if x in common_langs else 'other')
                
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(languages)
                n_categories = len(encoder.classes_)
                
                # One-hot encoding
                category_matrix = lil_matrix((len(encoded), n_categories))
                for i, cat in enumerate(encoded):
                    category_matrix[i, cat] = 1
                
                text_features.append(category_matrix.tocsr())
                print(f"   ✅ original_language: {n_categories} languages")
                
            except Exception as e:
                print(f"   ⚠️  Skipped original_language: {e}")
        
        # Combine all features
        if text_features:
            combined_features = hstack(text_features)
        else:
            # Fallback to identity matrix if no features
            combined_features = lil_matrix((len(self.metadata_df), 1)).tocsr()
        
        return combined_features
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating using optimized hybrid approach from README"""
        try:
            # Collaborative filtering score
            cf_score_raw = np.dot(
                self.model['user_factors'][user_id], 
                self.model['movie_factors'][:, movie_id]
            )
            
            # Normalize CF score as per README approach
            cf_score = (cf_score_raw - self.model['cf_mean']) / self.model['cf_std']
            cf_score = max(0, min(1, (cf_score + 3) / 6))  # Map to 0-1
            
            # Content-based score
            user_ratings = self.model['interaction_matrix'][user_id].toarray().flatten()
            rated_movies = np.where(user_ratings > 0)[0]
            
            if len(rated_movies) > 0:
                similarities = self.model['movie_similarity'][movie_id, rated_movies]
                ratings = user_ratings[rated_movies]
                
                if np.sum(similarities) > 0:
                    cb_score = np.average(ratings, weights=similarities)
                else:
                    cb_score = np.mean(ratings)
            else:
                cb_score = 0.5  # Default for new users
            
            # Hybrid combination with fixed optimal alpha from README
            hybrid_score = self.alpha * cf_score + (1 - self.alpha) * cb_score
            
            # Convert to 1-10 scale
            return np.clip(hybrid_score * 10, 1, 10)
            
        except Exception as e:
            return 6.0  # Default rating
    
    def generate_recommendations(self, username: str, scrape_if_missing: bool = True) -> Optional[pd.DataFrame]:
        """Generate recommendations for a user"""
        try:
            print(f"🎬 Generating recommendations for {username}...")
            
            # Check if user exists in dataset
            if username not in self.model['user_mapping']:
                if scrape_if_missing:
                    print(f"🔍 User {username} not found, attempting to scrape...")
                    if self._scrape_and_add_user(username):
                        # Retrain model with new user
                        print("🔄 Retraining model with new user data...")
                        if not self.train_model():
                            return None
                    else:
                        print(f"❌ Could not scrape data for {username}")
                        return None
                else:
                    print(f"❌ User {username} not found in dataset")
                    return None
            
            user_id = self.model['user_mapping'][username]
            
            # Get user's rated movies
            user_ratings = self.model['interaction_matrix'][user_id].toarray().flatten()
            rated_movies = set(np.where(user_ratings > 0)[0])
            
            print(f"   📊 User profile: {len(rated_movies)} rated movies")
            
            # Generate predictions for unrated movies
            predictions = []
            
            for movie_idx in range(len(self.model['movies'])):
                if movie_idx not in rated_movies:
                    tmdb_id = self.model['movies'][movie_idx]
                    
                    # Check if movie exists in metadata
                    movie_info_rows = self.metadata_df[self.metadata_df['tmdb_id'] == tmdb_id]
                    if len(movie_info_rows) == 0:
                        continue
                    
                    movie_info = movie_info_rows.iloc[0]
                    predicted_rating = self.predict_rating(user_id, movie_idx)
                    
                    predictions.append({
                        'tmdb_id': tmdb_id,
                        'title': movie_info.get('title', 'Unknown'),
                        'year': movie_info.get('year', 'Unknown'),
                        'genres': movie_info.get('genres', 'Unknown'),
                        'directors': movie_info.get('directors_main', 'Unknown'),
                        'predicted_rating': predicted_rating
                    })
            
            # Sort and take top recommendations
            predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
            top_predictions = predictions[:self.num_recommendations]
            
            # Create DataFrame
            recommendations_df = pd.DataFrame(top_predictions)
            if len(recommendations_df) > 0:
                recommendations_df['rank'] = range(1, len(recommendations_df) + 1)
                
                # Save recommendations
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(self.output_dir, exist_ok=True)
                output_file = os.path.join(self.output_dir, f'recommendations_{username}_{timestamp}.csv')
                recommendations_df.to_csv(output_file, index=False)
                
                print(f"✅ Generated {len(recommendations_df)} recommendations")
                print(f"   💾 Saved to: {os.path.basename(output_file)}")
            
            return recommendations_df
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _scrape_and_add_user(self, username: str) -> bool:
        """Scrape user data and integrate into dataset"""
        try:
            # Import scraper
            sys.path.append(self.scrapers_dir)
            from review_scraper import scrape_user_reviews
            import requests
            
            # Create session
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            
            # Scrape reviews
            reviews = scrape_user_reviews(username, session, max_reviews=500)
            
            if not reviews:
                return False
            
            # Convert to DataFrame and clean
            user_df = pd.DataFrame(reviews)
            user_df = user_df[user_df['rating'] > 0.5]  # Explicit ratings only
            
            # Match with metadata and add to dataset
            matched_reviews = []
            for _, review in user_df.iterrows():
                # Try to find matching movie in metadata
                movie_matches = self.metadata_df[
                    (self.metadata_df['title'] == review['title']) &
                    (self.metadata_df['year'] == review['year'])
                ]
                
                if len(movie_matches) > 0:
                    movie_info = movie_matches.iloc[0]
                    matched_reviews.append({
                        'username': username,
                        'tmdb_id': movie_info['tmdb_id'],
                        'title': review['title'],
                        'year': review['year'],
                        'rating': review['rating']
                    })
            
            if matched_reviews:
                # Add to reviews dataset
                new_reviews_df = pd.DataFrame(matched_reviews)
                self.reviews_df = pd.concat([self.reviews_df, new_reviews_df], ignore_index=True)
                
                print(f"   ✅ Added {len(matched_reviews)} reviews for {username}")
                return True
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error scraping user: {e}")
            return False
    
    def get_model_stats(self) -> dict:
        """Get model statistics"""
        if not self.model:
            return {}
        
        return {
            'num_users': len(self.model['users']),
            'num_movies': len(self.model['movies']),
            'num_reviews': len(self.reviews_df),
            'avg_rating': self.reviews_df['rating'].mean(),
            'rating_std': self.reviews_df['rating'].std(),
            'matrix_density': (self.model['interaction_matrix'].nnz / 
                             (len(self.model['users']) * len(self.model['movies']))) * 100,
            'training_time': self.training_time,
            'svd_components': self.num_components,
            'content_features': self.model['content_features'].shape[1],
            'alpha': self.alpha
        }


def main():
    """Command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python streamlined_engine.py <username>")
        print("Example: python streamlined_engine.py jaaackd")
        sys.exit(1)
    
    username = sys.argv[1]
    
    # Create engine with README specifications
    engine = LetterboxdRecommendationEngine(
        alpha=0.7,  # 70% collaborative filtering
        num_recommendations=25,
        num_components=25
    )
    
    # Load data
    if not engine.load_data():
        print("❌ Failed to load data. Run the pipeline first:")
        print("   python data_cleaning/full_production_pipeline.py")
        sys.exit(1)
    
    # Train model
    if not engine.train_model():
        print("❌ Failed to train model")
        sys.exit(1)
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(username)
    
    if recommendations is not None and len(recommendations) > 0:
        print(f"\n🎬 Top 10 Recommendations for @{username}:")
        print("─" * 70)
        
        for i, row in recommendations.head(10).iterrows():
            print(f"{row['rank']:2d}. {row['title']} ({row['year']})")
            print(f"    ⭐ {row['predicted_rating']:.1f}/10  |  {row['genres']}")
            if pd.notna(row['directors']) and row['directors'] != 'Unknown':
                print(f"    🎬 {row['directors']}")
            print()
        
        # Show model stats
        stats = engine.get_model_stats()
        print(f"📊 Model Stats: {stats['num_users']:,} users, {stats['num_movies']:,} movies")
        print(f"   Training time: {stats['training_time']:.1f}s")
        print(f"   Features: {stats['content_features']:,} content + {stats['svd_components']} SVD")
        
    else:
        print(f"❌ Failed to generate recommendations for {username}")


if __name__ == "__main__":
    main()

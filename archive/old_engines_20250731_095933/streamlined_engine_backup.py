#!/usr/bin/env python3
"""
Streamlined Letterboxd Hybrid Recommendation Engine

Optimized for speed and simplicity based on optimal parameters:
- Alpha = 0.7 (collaborative filtering weighted higher)
- No L2 regularization (L2 = 0.0)
- Removed deprecated features and redundant code
"""

import os
import sys
import time
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
from sklearn.linear_model import LinearRegression
from scipy.sparse import lil_matrix, hstack

class LetterboxdRecommendationEngine:
    """Streamlined hybrid recommendation engine for Letterboxd users"""
    
    def __init__(self, alpha=0.7, num_recommendations=25, num_components=25, 
                 reviews_file=None, metadata_file=None):
        self.alpha = alpha  # Fixed optimal value
        self.num_recommendations = num_recommendations
        self.num_components = num_components
        
        # Directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.scrapers_dir = os.path.join(self.base_dir, 'scrapers')
        self.output_dir = os.path.join(self.base_dir, 'output')
        
        # Data files - use specified files or auto-detect latest explicit dataset
        if reviews_file and metadata_file:
            self.reviews_file = reviews_file
            self.metadata_file = metadata_file
        else:
            self.reviews_file, self.metadata_file = self._find_latest_dataset()
        
        # Model components
        self.model = None
        self.reviews_df = None
        self.metadata_df = None
        
    def _find_latest_dataset(self):
        """Find the latest explicit dataset from pipeline or fallback to existing"""
        import glob
        
        # Look for latest explicit dataset from pipeline
        explicit_files = glob.glob(os.path.join(self.data_dir, 'cleaned_reviews_explicit_*.csv'))
        
        if explicit_files:
            # Use most recent explicit dataset
            latest_reviews = max(explicit_files, key=os.path.getctime)
            
            # Find corresponding metadata file
            timestamp = latest_reviews.split('_')[-1].replace('.csv', '')
            metadata_file = os.path.join(self.data_dir, f'cleaned_movie_metadata_{timestamp}.csv')
            
            if os.path.exists(metadata_file):
                print(f"🎯 Using latest explicit dataset: {os.path.basename(latest_reviews)}")
                return latest_reviews, metadata_file
        
        # Fallback to existing files
        reviews_fallback = os.path.join(self.data_dir, 'cleaned_reviews_v12_explicit.csv')
        metadata_fallback = os.path.join(self.data_dir, 'cleaned_movie_metadata_v9.csv')
        
        if os.path.exists(reviews_fallback):
            print(f"📂 Using fallback explicit dataset: {os.path.basename(reviews_fallback)}")
            return reviews_fallback, metadata_fallback
        
        # Final fallback to any explicit dataset
        explicit_files = glob.glob(os.path.join(self.data_dir, '*explicit*.csv'))
        if explicit_files:
            latest_reviews = max(explicit_files, key=os.path.getctime)
            # Use any metadata file as fallback
            metadata_files = glob.glob(os.path.join(self.data_dir, 'cleaned_movie_metadata*.csv'))
            if metadata_files:
                metadata_file = max(metadata_files, key=os.path.getctime)
                print(f"🔄 Using fallback explicit dataset: {os.path.basename(latest_reviews)}")
                return latest_reviews, metadata_file
        
        # Last resort - original files
        return (os.path.join(self.data_dir, 'cleaned_reviews_v12.csv'),
                os.path.join(self.data_dir, 'cleaned_movie_metadata_v9.csv'))
        
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
    
    def _extract_year(self, movie_info):
        """Extract year from movie info, trying multiple sources."""
        # Try input_year first
        input_year = movie_info.get('input_year')
        if input_year and str(input_year) != 'Unknown' and str(input_year).replace('.0', '').isdigit():
            return int(float(input_year))
        
        # Try release_date
        release_date = movie_info.get('release_date')
        if release_date and str(release_date) != 'Unknown':
            try:
                # Extract year from date string (YYYY-MM-DD format)
                year_str = str(release_date)[:4]
                if year_str.isdigit():
                    return int(year_str)
            except:
                pass
        
        # Try release_year as fallback
        release_year = movie_info.get('release_year')
        if release_year and str(release_year) != 'Unknown' and str(release_year).replace('.0', '').isdigit():
            return int(float(release_year))
        
        return 'Unknown'
    
    def _fetch_missing_metadata(self, title, year):
        """Fetch TMDB metadata for a missing movie with better error handling"""
        try:
            import requests
            import os
            import time
            from dotenv import load_dotenv
            
            load_dotenv()
            bearer_token = os.getenv("TMDB_BEARER_TOKEN")
            
            if not bearer_token:
                return None
            
            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'Content-Type': 'application/json;charset=utf-8'
            }
            
            # Search for movie with retry logic
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'query': title,
                'include_adult': 'false'
            }
            
            # Add year if available
            if isinstance(year, (int, float)) and not pd.isna(year):
                params['year'] = int(year)
            
            # Retry logic for rate limiting
            for attempt in range(3):
                try:
                    response = requests.get(search_url, headers=headers, params=params, timeout=15)
                    
                    if response.status_code == 429:  # Rate limited
                        time.sleep(1)
                        continue
                    elif response.status_code != 200:
                        return None
                    
                    data = response.json()
                    if not data['results']:
                        return None
                    
                    movie = data['results'][0]  # Take first result
                    movie_id = movie['id']
                    
                    # Get detailed info with retry
                    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
                    keywords_url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords"
                    
                    details_resp = requests.get(details_url, headers=headers, timeout=15)
                    if details_resp.status_code == 429:
                        time.sleep(1)
                        continue
                    
                    credits_resp = requests.get(credits_url, headers=headers, timeout=15)
                    if credits_resp.status_code == 429:
                        time.sleep(1)
                        continue
                        
                    keywords_resp = requests.get(keywords_url, headers=headers, timeout=15)
                    if keywords_resp.status_code == 429:
                        time.sleep(1)
                        continue
                    
                    if all(r.status_code == 200 for r in [details_resp, credits_resp, keywords_resp]):
                        details = details_resp.json()
                        credits = credits_resp.json()
                        keywords = keywords_resp.json()
                        
                        # Extract metadata
                        genres = '; '.join([g['name'] for g in details.get('genres', [])])
                        directors = ', '.join([
                            c['name'] for c in credits.get('crew', []) 
                            if c['job'] == 'Director'
                        ][:3])  # Limit to 3
                        cast = ', '.join([
                            c['name'] for c in credits.get('cast', [])
                        ][:5])  # Limit to 5
                        keywords_str = ', '.join([
                            k['name'] for k in keywords.get('keywords', [])
                        ][:10])  # Limit to 10
                        
                        release_year = details.get('release_date', '')[:4] if details.get('release_date') else year
                        if release_year:
                            try:
                                release_year = int(release_year)
                            except:
                                release_year = year
                        
                        return {
                            'tmdb_id': movie_id,
                            'title': title,
                            'release_year': release_year,
                            'input_year': year,
                            'genres': genres,
                            'directors': directors,
                            'cast': cast,
                            'keywords': keywords_str,
                            'original_language': details.get('original_language', 'en')
                        }
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.Timeout:
                    if attempt == 2:  # Last attempt
                        return None
                    time.sleep(0.5)
                    continue
                except Exception:
                    return None
            
        except Exception as e:
            # Silently fail - don't spam console during batch processing
            return None
        
        return None

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
        cleaned_data = []
        skipped_no_match = 0
        skipped_no_rating = 0
        movies_to_fetch = []  # Collect movies that need fetching
        
        print(f"🧹 Cleaning {len(raw_data)} scraped ratings for {username}...")
        
        # First pass: identify movies that need metadata fetching
        for _, row in raw_data.iterrows():
            title = row['title']
            year = row['year']
            rating = row.get('rating', 0)
            
            # Skip if no rating (implicit feedback)
            if rating <= 0:
                skipped_no_rating += 1
                continue
            
            # Try to find matching movie in metadata
            year_col = 'input_year' if 'input_year' in self.metadata_df.columns else 'release_year'
            
            movie_match = self.metadata_df[
                (self.metadata_df['title'] == title) & 
                (self.metadata_df[year_col] == year)
            ]
            
            # If no exact match, try just title
            if len(movie_match) == 0:
                movie_match = self.metadata_df[self.metadata_df['title'] == title]
            
            # If still no match, try year +/- 1
            if len(movie_match) == 0:
                if isinstance(year, (int, float)) and not pd.isna(year):
                    for year_offset in [-1, 1]:
                        test_year = year + year_offset
                        movie_match = self.metadata_df[
                            (self.metadata_df['title'] == title) & 
                            (self.metadata_df[year_col] == test_year)
                        ]
                        if len(movie_match) > 0:
                            break
            
            if len(movie_match) > 0:
                # Movie found in metadata
                movie_info = movie_match.iloc[0]
                cleaned_data.append({
                    'username': username,
                    'tmdb_id': movie_info['tmdb_id'],
                    'title': title,
                    'rating': rating
                })
            else:
                # Movie not found - need to fetch metadata
                movies_to_fetch.append({'title': title, 'year': year, 'rating': rating})
        
        # Batch fetch missing metadata with progress bar
        if movies_to_fetch:
            print(f"   🔍 Fetching metadata for {len(movies_to_fetch)} missing movies...")
            
            # Split into smaller batches for better rate limiting
            batch_size = 25  # Smaller batches to avoid overwhelming TMDB
            total_batches = (len(movies_to_fetch) + batch_size - 1) // batch_size
            total_fetched = 0
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(movies_to_fetch))
                batch = movies_to_fetch[start_idx:end_idx]
                
                print(f"   📦 Batch {batch_num + 1}/{total_batches}: {len(batch)} movies")
                
                # Use tqdm for progress bar
                try:
                    from tqdm import tqdm
                    pbar = tqdm(batch, desc=f"   📡 TMDB API (Batch {batch_num + 1})", unit="movie")
                except ImportError:
                    pbar = batch
                    print(f"   📡 Fetching batch {batch_num + 1} from TMDB API...")
                
                batch_fetched = 0
                
                for movie_data in pbar:
                    title = movie_data['title']
                    year = movie_data['year']
                    rating = movie_data['rating']
                    
                    new_metadata = self._fetch_missing_metadata(title, year)
                    if new_metadata:
                        # Add to metadata_df
                        new_row = pd.DataFrame([new_metadata])
                        self.metadata_df = pd.concat([self.metadata_df, new_row], ignore_index=True)
                        
                        # Add to cleaned data
                        cleaned_data.append({
                            'username': username,
                            'tmdb_id': new_metadata['tmdb_id'],
                            'title': title,
                            'rating': rating
                        })
                        batch_fetched += 1
                        total_fetched += 1
                        
                        if hasattr(pbar, 'set_postfix'):
                            pbar.set_postfix({'✅ Fetched': batch_fetched, '❌ Failed': len(batch) - batch_fetched, 'Total': total_fetched})
                    else:
                        skipped_no_match += 1
                
                if hasattr(pbar, 'close'):
                    pbar.close()
                
                # Brief pause between batches to be nice to TMDB
                if batch_num < total_batches - 1:
                    time.sleep(1)
            
            print(f"   🆕 Successfully fetched: {total_fetched}/{len(movies_to_fetch)} movies")
        
        print(f"   ✅ Integrated: {len(cleaned_data)} ratings")
        print(f"   ⚠️  Skipped (no metadata found): {skipped_no_match}")
        print(f"   ⚠️  Skipped (no rating): {skipped_no_rating}")
        
        return pd.DataFrame(cleaned_data)
        
        return pd.DataFrame(cleaned_data)
    
    def train_model(self):
        """Train the streamlined hybrid recommendation model"""
        try:
            print("🚀 Training hybrid model...")
            
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
                # Since we're using explicit-only dataset, all ratings are valid
                rating = row['rating'] / 10.0  # Normalize to 0-1
                interaction_matrix[user_idx, movie_idx] = rating
            
            # Convert to CSR for SVD
            interaction_matrix = interaction_matrix.tocsr()
            
            # Train SVD model
            print("🤝 Training collaborative filtering...")
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
            
            # Prepare content-based features (optimized core set)
            print("🎬 Preparing content-based features...")
            
            # Core features including separated other columns
            text_features = []
            core_features = ['genres', 'directors', 'cast', 'keywords']
            other_features = ['directors_other', 'cast_other', 'keywords_other']
            
            # Process main features
            for feature in core_features:
                if feature in self.metadata_df.columns:
                    max_feat = 300 if feature == 'keywords' else 200
                    vectorizer = TfidfVectorizer(max_features=max_feat, stop_words='english')
                    feature_matrix = vectorizer.fit_transform(
                        self.metadata_df[feature].fillna('').astype(str)
                    )
                    text_features.append(feature_matrix)
            
            # Process separated "other" features with lower weight
            for feature in other_features:
                if feature in self.metadata_df.columns:
                    # Lower max_features for "other" columns to reduce noise
                    max_feat = 50
                    vectorizer = TfidfVectorizer(max_features=max_feat, stop_words='english')
                    feature_matrix = vectorizer.fit_transform(
                        self.metadata_df[feature].fillna('').astype(str)
                    )
                    # Scale down the "other" features to reduce their impact
                    feature_matrix = feature_matrix * 0.3
                    text_features.append(feature_matrix)
            
            # Keep original_language as it's computationally cheap and helps
            if 'original_language' in self.metadata_df.columns:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(self.metadata_df['original_language'].fillna('unknown'))
                n_categories = len(encoder.classes_)
                category_matrix = lil_matrix((len(encoded), n_categories))
                for i, cat in enumerate(encoded):
                    category_matrix[i, cat] = 1
                text_features.append(category_matrix.tocsr())
            
            # Combine features
            if text_features:
                combined_features = hstack(text_features)
            else:
                combined_features = lil_matrix((len(self.metadata_df), 1)).tocsr()
            
            # Calculate movie similarity matrix
            movie_similarity = cosine_similarity(combined_features)
            
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
                'content_features': combined_features
            }
            
            print(f"✅ Model trained successfully!")
            print(f"   - Users: {len(users)}, Movies: {len(movies)}")
            print(f"   - Features: {combined_features.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating using optimized hybrid approach"""
        # Normalized collaborative filtering score
        cf_score_raw = np.dot(self.model['user_factors'][user_id], 
                             self.model['movie_factors'][:, movie_id])
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
        
        # Hybrid combination with fixed optimal alpha
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
            
            # Only consider movies that exist in both the model and metadata
            valid_movie_indices = []
            for movie_idx in range(len(self.model['movies'])):
                tmdb_id = self.model['movies'][movie_idx]
                # Check if movie exists in metadata and content features
                if (tmdb_id in self.metadata_df['tmdb_id'].values and 
                    movie_idx < self.model['content_features'].shape[0]):
                    valid_movie_indices.append(movie_idx)
            
            for movie_idx in valid_movie_indices:
                if movie_idx not in rated_movies:
                    tmdb_id = self.model['movies'][movie_idx]
                    
                    predicted_rating = self.predict_rating(user_id, movie_idx)
                    movie_info = self.metadata_df[self.metadata_df['tmdb_id'] == tmdb_id].iloc[0]
                    
                    predictions.append({
                        'tmdb_id': tmdb_id,
                        'title': movie_info.get('title', 'Unknown'),
                        'year': self._extract_year(movie_info),
                        'genres': movie_info.get('genres', 'Unknown'),
                        'predicted_rating': predicted_rating
                    })
            
            # Sort and take top recommendations
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
        print("Usage: python streamlined_engine.py <username>")
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

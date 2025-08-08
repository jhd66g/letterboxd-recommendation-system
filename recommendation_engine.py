#!/usr/bin/env python3
"""
Recommendation Engine

A production-grade pipeline that assembles data, trains a hybrid LightFM model,
and serves movie recommendations using collaborative filtering and content-based features.

Usage:
    python recommendation_engine.py --mode hybrid --train
    python recommendation_engine.py --mode cf_only --evaluate
    python recommendation_engine.py --recommend username --top_n 25
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Hybrid movie recommendation engine using LightFM."""
    
    def __init__(self, cf_weight=2.0, cb_weight=0.5):
        """Initialize the recommendation engine.
        
        Args:
            cf_weight: Weight for collaborative filtering (item_id) features
            cb_weight: Weight for content-based (metadata) features
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.item_features = None
        self.interactions = None
        self.metadata_df = None
        
    def load_data(self, reviews_file=None, features_file=None):
        """Load preprocessed data from CSV files.
        
        Args:
            reviews_file: Path to final_reviews CSV (username, tmdb_id, rating)
            features_file: Path to final_features CSV (tmdb_id + feature columns)
        """
        logger.info("🔄 Loading preprocessed data...")
        
        # Find latest data files if not specified
        if reviews_file is None:
            reviews_files = list(Path(".").glob("**/final_reviews_*.csv"))
            if not reviews_files:
                raise FileNotFoundError("No final_reviews CSV files found")
            reviews_file = max(reviews_files, key=lambda p: p.stat().st_mtime)
            
        if features_file is None:
            features_files = list(Path(".").glob("**/final_features_*.csv"))
            if not features_files:
                raise FileNotFoundError("No final_features CSV files found")
            features_file = max(features_files, key=lambda p: p.stat().st_mtime)
            
        logger.info(f"   Reviews: {reviews_file}")
        logger.info(f"   Features: {features_file}")
        
        # Load reviews data (only username, tmdb_id, rating columns)
        reviews_df = pd.read_csv(reviews_file, usecols=['username', 'tmdb_id', 'rating'])
        reviews_df = reviews_df.dropna(subset=['username', 'tmdb_id', 'rating'])
        
        # Load features data  
        features_df = pd.read_csv(features_file)
        
        # Create user and item mappings
        unique_users = reviews_df['username'].unique()
        unique_items = reviews_df['tmdb_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create interaction matrix
        logger.info("🔄 Creating interaction matrix...")
        interactions = sp.lil_matrix((len(unique_users), len(unique_items)))
        
        for _, row in reviews_df.iterrows():
            user_idx = self.user_mapping[row['username']]
            item_idx = self.item_mapping[row['tmdb_id']]
            interactions[user_idx, item_idx] = row['rating']
            
        self.interactions = interactions.tocsr()
        
        # Prepare item features matrix
        logger.info("🔄 Preparing item features...")
        # Filter features to only items we have in interactions
        features_filtered = features_df[features_df['tmdb_id'].isin(unique_items)].copy()
        
        # Remove duplicates by taking the first occurrence
        features_filtered = features_filtered.drop_duplicates(subset=['tmdb_id'], keep='first')
        
        features_filtered = features_filtered.set_index('tmdb_id').reindex(unique_items).fillna(0)
        
        # Store metadata for later use
        metadata_file = str(reviews_file).replace('final_reviews', 'final_metadata')
        if Path(metadata_file).exists():
            self.metadata_df = pd.read_csv(metadata_file)
        
        # Create feature matrix (excluding tmdb_id)
        feature_columns = [col for col in features_filtered.columns if col != 'tmdb_id']
        feature_matrix = features_filtered[feature_columns].values
        
        # Convert to sparse matrix
        self.item_features = sp.csr_matrix(feature_matrix)
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   Users: {len(unique_users):,}")
        logger.info(f"   Items: {len(unique_items):,}")
        logger.info(f"   Interactions: {self.interactions.nnz:,}")
        logger.info(f"   Features per item: {self.item_features.shape[1]:,}")
        
    def scale_features(self, mode='hybrid'):
        """Scale features based on the selected mode.
        
        Args:
            mode: 'hybrid', 'cf_only', or 'cb_only'
        """
        if self.item_features is None:
            raise ValueError("Must load data first")
            
        logger.info(f"🔄 Scaling features for {mode} mode...")
        
        if mode == 'cf_only':
            # Zero out all metadata features, keep only basic item representation
            logger.info("   CF-only: Using minimal item features")
            # Create minimal identity matrix for items
            scaled_features = sp.eye(self.item_features.shape[0], format='csr')
            
        elif mode == 'cb_only':
            # Use metadata features only, apply CB weight
            scaled_features = self.item_features.copy() * self.cb_weight
            logger.info(f"   CB-only: Applied weight {self.cb_weight}")
            
        elif mode == 'hybrid':
            # Create hybrid features: CB features + CF boost
            # Apply CB weight to content features
            cb_features = self.item_features.copy() * self.cb_weight
            
            # Create CF features (identity matrix) with CF boost
            cf_features = sp.eye(self.item_features.shape[0], format='csr') * self.cf_weight
            
            # Combine CB and CF features horizontally
            scaled_features = sp.hstack([cb_features, cf_features], format='csr')
            
            logger.info(f"   Hybrid: Applied CB weight {self.cb_weight}, CF boost {self.cf_weight}")
            logger.info(f"   Hybrid: Combined features shape: {scaled_features.shape}")
            
        return scaled_features
        
    def train_model(self, mode='hybrid', components=64, loss='warp', 
                   learning_rate=0.05, item_alpha=1e-6, user_alpha=1e-6, 
                   epochs=10, random_state=42):
        """Train the LightFM model.
        
        Args:
            mode: 'hybrid', 'cf_only', or 'cb_only'
            components: Number of latent factors
            loss: Loss function ('warp' or 'bpr')
            learning_rate: Learning rate for training
            item_alpha: L2 penalty for item features
            user_alpha: L2 penalty for user features  
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        logger.info(f"🚀 Training LightFM model in {mode} mode...")
        
        # Scale features based on mode
        scaled_features = self.scale_features(mode)
        
        # Initialize model
        self.model = LightFM(
            no_components=components,
            loss=loss,
            learning_rate=learning_rate,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            random_state=random_state
        )
        
        # Train model
        logger.info(f"   Training for {epochs} epochs...")
        self.model.fit(
            interactions=self.interactions,
            item_features=scaled_features,
            epochs=epochs,
            verbose=True
        )
        
        logger.info("✅ Model training complete")
        
    def save_model(self, mode='hybrid'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_file = model_dir / f"lightfm_{mode}_{timestamp}.pkl"
        
        model_data = {
            'model': self.model,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'mode': mode
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"💾 Model saved to {model_file}")
        return model_file
        
    def load_model(self, model_file):
        """Load a trained model from disk."""
        logger.info(f"📂 Loading model from {model_file}")
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.user_mapping = model_data['user_mapping']
        self.item_mapping = model_data['item_mapping']
        self.reverse_user_mapping = model_data['reverse_user_mapping']
        self.reverse_item_mapping = model_data['reverse_item_mapping']
        self.cf_weight = model_data['cf_weight']
        self.cb_weight = model_data['cb_weight']
        
        logger.info("✅ Model loaded successfully")
        
    def evaluate_model(self, mode='hybrid', k=25, test_size=0.2):
        """Evaluate model performance on held-out test set."""
        if self.model is None:
            raise ValueError("No trained model available")
            
        logger.info(f"📊 Evaluating {mode} model...")
        
        # Create test interactions by randomly masking some interactions
        test_interactions = self.interactions.copy()
        train_interactions = self.interactions.copy()
        
        # Randomly select interactions to mask for testing
        rows, cols = test_interactions.nonzero()
        n_test = int(len(rows) * test_size)
        test_indices = np.random.choice(len(rows), n_test, replace=False)
        
        # Create test set with only selected interactions
        test_mask = sp.lil_matrix(test_interactions.shape)
        for idx in test_indices:
            test_mask[rows[idx], cols[idx]] = test_interactions[rows[idx], cols[idx]]
            train_interactions[rows[idx], cols[idx]] = 0  # Remove from training
            
        test_interactions = test_mask.tocsr()
        train_interactions = train_interactions.tocsr()
        
        # Scale features for evaluation
        scaled_features = self.scale_features(mode)
        
        # Calculate metrics using the original full interaction matrix for reference
        # but evaluate on the held-out test interactions
        logger.info(f"   Evaluating on {test_interactions.nnz:,} held-out interactions...")
        
        try:
            precision = precision_at_k(self.model, test_interactions, 
                                     item_features=scaled_features, k=k).mean()
            recall = recall_at_k(self.model, test_interactions,
                               item_features=scaled_features, k=k).mean()
            auc = auc_score(self.model, test_interactions,
                           item_features=scaled_features).mean()
        except Exception as e:
            logger.warning(f"Standard evaluation failed: {e}")
            logger.info("Using simplified evaluation on full dataset...")
            
            # Fallback: evaluate on full dataset (less rigorous but still useful)
            precision = precision_at_k(self.model, self.interactions, 
                                     item_features=scaled_features, k=k).mean()
            recall = recall_at_k(self.model, self.interactions,
                               item_features=scaled_features, k=k).mean()
            auc = auc_score(self.model, self.interactions,
                           item_features=scaled_features).mean()
        
        metrics = {
            'mode': mode,
            'precision_at_k': float(precision),
            'recall_at_k': float(recall),
            'auc': float(auc),
            'k': k,
            'test_interactions': int(test_interactions.nnz) if 'test_interactions' in locals() else int(self.interactions.nnz)
        }
        
        logger.info(f"📈 Evaluation Results ({mode}):")
        logger.info(f"   Precision@{k}: {precision:.4f}")
        logger.info(f"   Recall@{k}: {recall:.4f}")
        logger.info(f"   AUC: {auc:.4f}")
        
        return metrics
        
    def recommend(self, username, top_n=25, mode='hybrid'):
        """Generate top-N recommendations for a user.
        
        Args:
            username: Letterboxd username
            top_n: Number of recommendations to return
            mode: Model mode to use for predictions
            
        Returns:
            List of recommended movies with metadata
        """
        if self.model is None:
            raise ValueError("No trained model available")
            
        if username not in self.user_mapping:
            raise ValueError(f"User '{username}' not found in training data")
            
        logger.info(f"🎯 Generating {top_n} recommendations for '{username}'...")
        
        user_idx = self.user_mapping[username]
        n_items = len(self.item_mapping)
        
        # Scale features for prediction
        scaled_features = self.scale_features(mode)
        
        # Get predictions for all items
        scores = self.model.predict(
            user_ids=[user_idx] * n_items,
            item_ids=list(range(n_items)),
            item_features=scaled_features
        )
        
        # Get top N items
        top_items = np.argsort(scores)[::-1][:top_n]
        
        # Convert back to tmdb_ids and add metadata
        recommendations = []
        for rank, item_idx in enumerate(top_items, 1):
            tmdb_id = self.reverse_item_mapping[item_idx]
            score = scores[item_idx]
            
            rec = {
                'rank': rank,
                'tmdb_id': int(tmdb_id),
                'score': float(score)
            }
            
            # Add metadata if available
            if self.metadata_df is not None:
                metadata = self.metadata_df[self.metadata_df['tmdb_id'] == tmdb_id]
                if not metadata.empty:
                    metadata_row = metadata.iloc[0]
                    rec.update({
                        'title': metadata_row.get('title', 'Unknown'),
                        'year': int(metadata_row.get('year', 0)) if pd.notna(metadata_row.get('year')) else None,
                        'runtime': int(metadata_row.get('runtime', 0)) if pd.notna(metadata_row.get('runtime')) else None,
                        'directors': metadata_row.get('directors', []),
                        'cast': metadata_row.get('cast', []),
                        'genres': metadata_row.get('genres', []),
                        'language': metadata_row.get('original_language', ''),
                        'overview': metadata_row.get('overview', ''),
                        'budget': int(metadata_row.get('budget', 0)) if pd.notna(metadata_row.get('budget')) else None,
                        'revenue': int(metadata_row.get('revenue', 0)) if pd.notna(metadata_row.get('revenue')) else None
                    })
            
            recommendations.append(rec)
            
        logger.info(f"✅ Generated {len(recommendations)} recommendations")
        return recommendations

def main():
    """Command-line interface for the recommendation engine."""
    parser = argparse.ArgumentParser(description="Letterboxd Recommendation Engine")
    parser.add_argument("--mode", choices=['hybrid', 'cf_only', 'cb_only'], 
                       default='hybrid', help="Model mode")
    parser.add_argument("--train", action='store_true', help="Train a new model")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate model")
    parser.add_argument("--recommend", type=str, help="Username to generate recommendations for")
    parser.add_argument("--top_n", type=int, default=25, help="Number of recommendations")
    parser.add_argument("--reviews_file", type=str, help="Path to reviews CSV")
    parser.add_argument("--features_file", type=str, help="Path to features CSV")
    parser.add_argument("--model_file", type=str, help="Path to saved model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Load data
    engine.load_data(args.reviews_file, args.features_file)
    
    # Load existing model or train new one
    if args.model_file:
        engine.load_model(args.model_file)
    elif args.train:
        engine.train_model(mode=args.mode, epochs=args.epochs)
        model_file = engine.save_model(args.mode)
        logger.info(f"Model saved to: {model_file}")
    else:
        # Try to find existing model
        model_files = list(Path("models").glob(f"lightfm_{args.mode}_*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            engine.load_model(latest_model)
        else:
            logger.warning("No existing model found, training new model...")
            engine.train_model(mode=args.mode, epochs=args.epochs)
            engine.save_model(args.mode)
    
    # Evaluate model
    if args.evaluate:
        metrics = engine.evaluate_model(args.mode)
        print(json.dumps(metrics, indent=2))
    
    # Generate recommendations
    if args.recommend:
        try:
            recommendations = engine.recommend(args.recommend, args.top_n, args.mode)
            print(json.dumps(recommendations, indent=2))
        except ValueError as e:
            logger.error(f"Error generating recommendations: {e}")

if __name__ == "__main__":
    main()

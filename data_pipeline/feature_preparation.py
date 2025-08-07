#!/usr/bin/env python3
"""
Feature Preparation Script

Transforms movie metadata into engineered features for machine learning models.
Creates one-hot, multi-hot, and TF-IDF features as specified in the project design.

Usage:
    python feature_preparation.py <final_metadata_csv> [output_csv]
"""

import pandas as pd
import numpy as np
import ast
import sys
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_eval_list(x):
    """Safely evaluate string representation of lists."""
    if pd.isna(x) or x == '' or x == '[]':
        return []
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x if isinstance(x, list) else []
    except (ValueError, SyntaxError):
        return []

def create_decade_buckets(years):
    """Create decade buckets for years."""
    decade_columns = {}
    decades = ['Pre-1950', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
    
    for decade in decades:
        decade_columns[f'decade_{decade}'] = np.zeros(len(years), dtype=int)
    
    for i, year in enumerate(years):
        if pd.isna(year):
            continue
        year = int(year)
        if year < 1950:
            decade_columns['decade_Pre-1950'][i] = 1
        elif year < 1960:
            decade_columns['decade_1950s'][i] = 1
        elif year < 1970:
            decade_columns['decade_1960s'][i] = 1
        elif year < 1980:
            decade_columns['decade_1970s'][i] = 1
        elif year < 1990:
            decade_columns['decade_1980s'][i] = 1
        elif year < 2000:
            decade_columns['decade_1990s'][i] = 1
        elif year < 2010:
            decade_columns['decade_2000s'][i] = 1
        elif year < 2020:
            decade_columns['decade_2010s'][i] = 1
        else:
            decade_columns['decade_2020s'][i] = 1
    
    return decade_columns

def create_runtime_buckets(runtimes):
    """Create runtime quartile buckets."""
    # Filter out zero and NaN values for quartile calculation
    valid_runtimes = runtimes[(runtimes > 0) & runtimes.notna()]
    
    if len(valid_runtimes) == 0:
        logger.warning("No valid runtime data found")
        return {}
    
    q25 = valid_runtimes.quantile(0.25)
    q50 = valid_runtimes.quantile(0.50)
    q75 = valid_runtimes.quantile(0.75)
    
    logger.info(f"Runtime quartiles: Q1={q25:.0f}, Q2={q50:.0f}, Q3={q75:.0f}")
    
    runtime_columns = {
        f'runtime_Q1_(0,{q25:.0f}]': np.zeros(len(runtimes), dtype=int),
        f'runtime_Q2_({q25:.0f},{q50:.0f}]': np.zeros(len(runtimes), dtype=int),
        f'runtime_Q3_({q50:.0f},{q75:.0f}]': np.zeros(len(runtimes), dtype=int),
        f'runtime_Q4_({q75:.0f},∞)': np.zeros(len(runtimes), dtype=int)
    }
    
    for i, runtime in enumerate(runtimes):
        if pd.isna(runtime) or runtime <= 0:
            continue
        
        if runtime <= q25:
            runtime_columns[f'runtime_Q1_(0,{q25:.0f}]'][i] = 1
        elif runtime <= q50:
            runtime_columns[f'runtime_Q2_({q25:.0f},{q50:.0f}]'][i] = 1
        elif runtime <= q75:
            runtime_columns[f'runtime_Q3_({q50:.0f},{q75:.0f}]'][i] = 1
        else:
            runtime_columns[f'runtime_Q4_({q75:.0f},∞)'][i] = 1
    
    return runtime_columns

def create_multihot_features(df, column, top_n, prefix):
    """Create multi-hot encoding for list columns."""
    logger.info(f"Creating multi-hot features for {column} (top {top_n})")
    
    # Extract all items and count frequencies
    all_items = []
    for items_list in df[column]:
        parsed_items = safe_eval_list(items_list)
        all_items.extend(parsed_items)
    
    item_counts = Counter(all_items)
    top_items = [item for item, count in item_counts.most_common(top_n)]
    
    logger.info(f"Found {len(item_counts)} unique items, using top {len(top_items)}")
    
    # Create columns
    multihot_columns = {}
    for item in top_items:
        multihot_columns[f'{prefix}_{item}'] = np.zeros(len(df), dtype=int)
    
    # Fill values
    for i, items_list in enumerate(df[column]):
        parsed_items = safe_eval_list(items_list)
        
        for item in parsed_items:
            if item in top_items:
                multihot_columns[f'{prefix}_{item}'][i] = 1
    
    return multihot_columns

def create_budget_revenue_buckets(values, prefix):
    """Create log-transformed buckets for budget/revenue."""
    logger.info(f"Creating {prefix} buckets")
    
    # Apply log1p transformation to positive values
    log_values = np.log1p(values.where(values > 0, 0))
    
    # Get quartiles of positive values only
    positive_log_values = log_values[log_values > 0]
    
    if len(positive_log_values) == 0:
        logger.warning(f"No positive {prefix} values found")
        return {f'{prefix}_Zero': (values == 0).astype(int)}
    
    q25 = positive_log_values.quantile(0.25)
    q50 = positive_log_values.quantile(0.50)  
    q75 = positive_log_values.quantile(0.75)
    
    logger.info(f"{prefix} log quartiles: Q1={q25:.2f}, Q2={q50:.2f}, Q3={q75:.2f}")
    
    buckets = {
        f'{prefix}_Zero': np.zeros(len(values), dtype=int),
        f'{prefix}_Low': np.zeros(len(values), dtype=int),
        f'{prefix}_Medium': np.zeros(len(values), dtype=int),
        f'{prefix}_High': np.zeros(len(values), dtype=int),
        f'{prefix}_VeryHigh': np.zeros(len(values), dtype=int)
    }
    
    for i, (value, log_value) in enumerate(zip(values, log_values)):
        if pd.isna(value) or value <= 0:
            buckets[f'{prefix}_Zero'][i] = 1
        elif log_value <= q25:
            buckets[f'{prefix}_Low'][i] = 1
        elif log_value <= q50:
            buckets[f'{prefix}_Medium'][i] = 1
        elif log_value <= q75:
            buckets[f'{prefix}_High'][i] = 1
        else:
            buckets[f'{prefix}_VeryHigh'][i] = 1
    
    return buckets

def create_popularity_buckets(popularity):
    """Create popularity tier buckets."""
    logger.info("Creating popularity buckets")
    
    buckets = {
        'popularity_Low': np.zeros(len(popularity), dtype=int),
        'popularity_Medium': np.zeros(len(popularity), dtype=int),
        'popularity_High': np.zeros(len(popularity), dtype=int)
    }
    
    for i, pop in enumerate(popularity):
        if pd.isna(pop):
            continue
        
        if pop <= 1:
            buckets['popularity_Low'][i] = 1
        elif pop <= 5:
            buckets['popularity_Medium'][i] = 1
        else:
            buckets['popularity_High'][i] = 1
    
    return buckets

def create_language_onehot(languages, top_n=10):
    """Create one-hot encoding for languages."""
    logger.info(f"Creating language one-hot (top {top_n})")
    
    lang_counts = Counter(languages.dropna())
    top_languages = [lang for lang, count in lang_counts.most_common(top_n)]
    
    logger.info(f"Top languages: {top_languages}")
    
    lang_columns = {}
    for lang in top_languages:
        lang_columns[f'language_{lang}'] = (languages == lang).astype(int)
    lang_columns['language_Other'] = (~languages.isin(top_languages) & languages.notna()).astype(int)
    
    return lang_columns

def create_tfidf_features(df, column, max_features=1000):
    """Create TF-IDF features for keywords."""
    logger.info(f"Creating TF-IDF features for {column}")
    
    # Convert lists to space-separated strings
    text_data = []
    for items_list in df[column]:
        parsed_items = safe_eval_list(items_list)
        text_data.append(' '.join(parsed_items) if parsed_items else '')
    
    # Apply TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=None,  # Keywords are already meaningful
        lowercase=False,   # Keywords are pre-processed
        token_pattern=r'(?u)\b\w+\b'  # Standard word tokenization
    )
    
    tfidf_matrix = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()
    
    logger.info(f"Created {len(feature_names)} TF-IDF features")
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'keyword_tfidf_{name}' for name in feature_names]
    )
    
    return tfidf_df

def main():
    """Main feature preparation function."""
    if len(sys.argv) < 2:
        print("Usage: python feature_preparation.py <final_metadata_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"final_features_{timestamp}.csv"
    
    logger.info(f"🚀 Starting feature preparation...")
    logger.info(f"   Input: {input_file}")
    logger.info(f"   Output: {output_file}")
    
    start_time = time.time()
    
    # Load data
    logger.info("📊 Loading metadata...")
    df = pd.read_csv(input_file)
    logger.info(f"   Loaded {len(df):,} movies with {len(df.columns)} columns")
    
    # Initialize feature DataFrame with tmdb_id
    features_df = pd.DataFrame({'tmdb_id': df['tmdb_id']})
    
    # 1. Year → decade buckets (one-hot)
    logger.info("🗓️  Creating decade features...")
    decade_features = create_decade_buckets(df['year'])
    for col, values in decade_features.items():
        features_df[col] = values
    
    # 2. Runtime → quartile buckets (one-hot)
    logger.info("⏱️  Creating runtime features...")
    runtime_features = create_runtime_buckets(df['runtime'])
    for col, values in runtime_features.items():
        features_df[col] = values
    
    # 3. Directors → multi-hot top 200
    logger.info("🎬 Creating director features...")
    director_features = create_multihot_features(df, 'directors', 200, 'director')
    for col, values in director_features.items():
        features_df[col] = values
    
    # 4. Cast → multi-hot top 500
    logger.info("🎭 Creating cast features...")
    cast_features = create_multihot_features(df, 'cast', 500, 'cast')
    for col, values in cast_features.items():
        features_df[col] = values
    
    # 5. Genres → multi-hot for all 19 genres
    logger.info("🎨 Creating genre features...")
    genre_features = create_multihot_features(df, 'genres', 19, 'genre')
    for col, values in genre_features.items():
        features_df[col] = values
    
    # 6. Keywords → TF-IDF
    logger.info("🏷️  Creating keyword TF-IDF features...")
    keyword_tfidf = create_tfidf_features(df, 'keywords')
    features_df = pd.concat([features_df, keyword_tfidf], axis=1)
    
    # 7. Budget → log1p + buckets
    logger.info("💰 Creating budget features...")
    budget_features = create_budget_revenue_buckets(df['budget'], 'budget')
    for col, values in budget_features.items():
        features_df[col] = values
    
    # 8. Revenue → log1p + buckets
    logger.info("💸 Creating revenue features...")
    revenue_features = create_budget_revenue_buckets(df['revenue'], 'revenue')
    for col, values in revenue_features.items():
        features_df[col] = values
    
    # 9. Original language → one-hot top 10
    logger.info("🌍 Creating language features...")
    language_features = create_language_onehot(df['original_language'])
    for col, values in language_features.items():
        features_df[col] = values
    
    # 10. Popularity → buckets (one-hot)
    logger.info("⭐ Creating popularity features...")
    popularity_features = create_popularity_buckets(df['popularity'])
    for col, values in popularity_features.items():
        features_df[col] = values
    
    # Save features
    logger.info("💾 Saving features...")
    features_df.to_csv(output_file, index=False)
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"\n✅ FEATURE PREPARATION COMPLETE")
    logger.info(f"   Processing time: {elapsed_time:.1f} seconds")
    logger.info(f"   Input movies: {len(df):,}")
    logger.info(f"   Output features: {len(features_df.columns):,} columns")
    logger.info(f"   Output file: {output_file}")
    
    # Feature summary
    feature_types = {
        'decade': sum(1 for col in features_df.columns if col.startswith('decade_')),
        'runtime': sum(1 for col in features_df.columns if col.startswith('runtime_')),
        'director': sum(1 for col in features_df.columns if col.startswith('director_')),
        'cast': sum(1 for col in features_df.columns if col.startswith('cast_')),
        'genre': sum(1 for col in features_df.columns if col.startswith('genre_')),
        'keyword_tfidf': sum(1 for col in features_df.columns if col.startswith('keyword_tfidf_')),
        'budget': sum(1 for col in features_df.columns if col.startswith('budget_')),
        'revenue': sum(1 for col in features_df.columns if col.startswith('revenue_')),
        'language': sum(1 for col in features_df.columns if col.startswith('language_')),
        'popularity': sum(1 for col in features_df.columns if col.startswith('popularity_'))
    }
    
    logger.info("📊 Feature Summary:")
    for feature_type, count in feature_types.items():
        logger.info(f"   {feature_type}: {count} features")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Metadata Analysis Script

Analyzes movie metadata to understand data distribution and guide feature engineering decisions.
Provides comprehensive statistics on years, runtime, budget, revenue, directors, cast, genres, 
languages, and keywords.

Usage:
    python metadata_analysis.py <metadata_csv_file>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import sys
import logging
from collections import Counter
from pathlib import Path

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

def analyze_numerical_distribution(df, column, title):
    """Analyze and display statistics for numerical columns."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    # Remove NaN values for analysis
    data = df[column].dropna()
    
    if len(data) == 0:
        print("   ⚠️  No data available for this column")
        return
    
    # Basic statistics
    print(f"   Total records: {len(df):,}")
    print(f"   Non-null records: {len(data):,}")
    print(f"   Null records: {len(df) - len(data):,} ({((len(df) - len(data))/len(df)*100):.1f}%)")
    print(f"\n   📈 Distribution Statistics:")
    print(f"   Min: {data.min():,.0f}")
    print(f"   Max: {data.max():,.0f}")
    print(f"   Mean: {data.mean():,.0f}")
    print(f"   Median: {data.median():,.0f}")
    print(f"   Std Dev: {data.std():,.0f}")
    
    # Percentiles
    print(f"\n   📊 Percentiles:")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(data, p)
        print(f"   {p:2d}th: {value:,.0f}")
    
    # Identify outliers (IQR method)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_low = Q1 - 1.5 * IQR
    outlier_threshold_high = Q3 + 1.5 * IQR
    
    outliers = data[(data < outlier_threshold_low) | (data > outlier_threshold_high)]
    print(f"\n   🔍 Outlier Analysis (IQR method):")
    print(f"   Outliers: {len(outliers):,} ({len(outliers)/len(data)*100:.1f}%)")
    if len(outliers) > 0:
        print(f"   Outlier range: {outliers.min():,.0f} - {outliers.max():,.0f}")
    
    return {
        'total': len(df),
        'non_null': len(data),
        'min': data.min(),
        'max': data.max(),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'outliers': len(outliers)
    }

def analyze_list_column(df, column, title, max_display=20):
    """Analyze columns containing lists (directors, cast, genres, etc.)."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    # Parse lists safely
    all_items = []
    valid_records = 0
    
    for idx, item in enumerate(df[column]):
        parsed_list = safe_eval_list(item)
        if parsed_list:
            all_items.extend(parsed_list)
            valid_records += 1
    
    print(f"   Total records: {len(df):,}")
    print(f"   Records with data: {valid_records:,} ({valid_records/len(df)*100:.1f}%)")
    print(f"   Total unique items: {len(set(all_items)):,}")
    print(f"   Total item occurrences: {len(all_items):,}")
    
    if len(all_items) > 0:
        avg_per_movie = len(all_items) / valid_records if valid_records > 0 else 0
        print(f"   Average items per movie: {avg_per_movie:.1f}")
        
        # Most common items
        item_counts = Counter(all_items)
        print(f"\n   🏆 Top {min(max_display, len(item_counts))} Most Common:")
        for i, (item, count) in enumerate(item_counts.most_common(max_display), 1):
            print(f"   {i:2d}. {item}: {count:,} movies ({count/valid_records*100:.1f}%)")
        
        return {
            'total_records': len(df),
            'records_with_data': valid_records,
            'unique_items': len(set(all_items)),
            'total_occurrences': len(all_items),
            'avg_per_movie': avg_per_movie,
            'top_items': item_counts.most_common(10)
        }
    
    return None

def analyze_categorical_column(df, column, title, max_display=20):
    """Analyze categorical columns."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    # Get value counts
    value_counts = df[column].value_counts(dropna=False)
    null_count = df[column].isnull().sum()
    
    print(f"   Total records: {len(df):,}")
    print(f"   Non-null records: {len(df) - null_count:,}")
    print(f"   Null records: {null_count:,} ({null_count/len(df)*100:.1f}%)")
    print(f"   Unique values: {df[column].nunique():,}")
    
    print(f"\n   🏆 Top {min(max_display, len(value_counts))} Most Common:")
    for i, (value, count) in enumerate(value_counts.head(max_display).items(), 1):
        percentage = count / len(df) * 100
        print(f"   {i:2d}. {value}: {count:,} movies ({percentage:.1f}%)")
    
    return {
        'total_records': len(df),
        'non_null': len(df) - null_count,
        'unique_values': df[column].nunique(),
        'top_values': value_counts.head(10).to_dict()
    }

def create_visualizations(df, output_dir):
    """Create visualization plots for key metrics."""
    logger.info("📊 Creating visualization plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create output directory
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Year Distribution
    plt.figure(figsize=(12, 6))
    year_data = df['year'].dropna()
    if len(year_data) > 0:
        plt.subplot(1, 2, 1)
        plt.hist(year_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Movie Year Distribution')
        plt.xlabel('Year')
        plt.ylabel('Number of Movies')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(year_data)
        plt.title('Movie Year Box Plot')
        plt.ylabel('Year')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "year_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Runtime Distribution
    plt.figure(figsize=(12, 6))
    runtime_data = df['runtime'].dropna()
    runtime_data = runtime_data[runtime_data > 0]  # Remove zero runtimes
    if len(runtime_data) > 0:
        plt.subplot(1, 2, 1)
        plt.hist(runtime_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Movie Runtime Distribution')
        plt.xlabel('Runtime (minutes)')
        plt.ylabel('Number of Movies')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(runtime_data)
        plt.title('Movie Runtime Box Plot')
        plt.ylabel('Runtime (minutes)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "runtime_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Budget vs Revenue (if available)
    budget_data = df['budget'].dropna()
    revenue_data = df['revenue'].dropna()
    
    if len(budget_data) > 0 and len(revenue_data) > 0:
        # Filter out zero values
        budget_nonzero = budget_data[budget_data > 0]
        revenue_nonzero = revenue_data[revenue_data > 0]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(np.log10(budget_nonzero), bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Budget Distribution (Log Scale)')
        plt.xlabel('Log10(Budget)')
        plt.ylabel('Number of Movies')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(np.log10(revenue_nonzero), bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.title('Revenue Distribution (Log Scale)')
        plt.xlabel('Log10(Revenue)')
        plt.ylabel('Number of Movies')
        plt.grid(True, alpha=0.3)
        
        # Budget vs Revenue scatter plot
        common_indices = budget_data.index.intersection(revenue_data.index)
        if len(common_indices) > 0:
            budget_common = budget_data.loc[common_indices]
            revenue_common = revenue_data.loc[common_indices]
            
            # Filter positive values
            positive_mask = (budget_common > 0) & (revenue_common > 0)
            if positive_mask.sum() > 0:
                plt.subplot(1, 3, 3)
                plt.scatter(np.log10(budget_common[positive_mask]), 
                          np.log10(revenue_common[positive_mask]), 
                          alpha=0.6, color='purple')
                plt.title('Budget vs Revenue (Log Scale)')
                plt.xlabel('Log10(Budget)')
                plt.ylabel('Log10(Revenue)')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "budget_revenue_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"📊 Visualizations saved to: {viz_dir}")

def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python metadata_analysis.py <metadata_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    logger.info(f"🔍 Loading metadata from: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"✅ Loaded {len(df):,} movies with {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"🎬 LETTERBOXD METADATA ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"Dataset: {input_file}")
    print(f"Total Movies: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    
    # Store results for summary
    results = {}
    
    # 1. Numerical Analysis
    if 'year' in df.columns:
        results['year'] = analyze_numerical_distribution(df, 'year', 'MOVIE YEARS ANALYSIS')
    
    if 'runtime' in df.columns:
        results['runtime'] = analyze_numerical_distribution(df, 'runtime', 'MOVIE RUNTIME ANALYSIS')
    
    if 'budget' in df.columns:
        results['budget'] = analyze_numerical_distribution(df, 'budget', 'MOVIE BUDGET ANALYSIS')
    
    if 'revenue' in df.columns:
        results['revenue'] = analyze_numerical_distribution(df, 'revenue', 'MOVIE REVENUE ANALYSIS')
    
    if 'popularity' in df.columns:
        results['popularity'] = analyze_numerical_distribution(df, 'popularity', 'MOVIE POPULARITY ANALYSIS')
    
    # 2. List Column Analysis
    if 'directors_main' in df.columns:
        results['directors_main'] = analyze_list_column(df, 'directors_main', 'MAIN DIRECTORS ANALYSIS')
    
    if 'cast_main' in df.columns:
        results['cast_main'] = analyze_list_column(df, 'cast_main', 'MAIN CAST ANALYSIS')
    
    if 'genres' in df.columns:
        results['genres'] = analyze_list_column(df, 'genres', 'GENRES ANALYSIS')
    
    if 'keywords' in df.columns:
        results['keywords'] = analyze_list_column(df, 'keywords', 'KEYWORDS ANALYSIS', max_display=30)
    
    # 3. Categorical Analysis
    if 'original_language' in df.columns:
        results['original_language'] = analyze_categorical_column(df, 'original_language', 'ORIGINAL LANGUAGE ANALYSIS')
    
    # 3.5. Popularity Distribution Analysis
    if 'popularity' in df.columns:
        print(f"\n{'='*60}")
        print(f"📊 POPULARITY DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")
        
        pop_data = df['popularity'].dropna()
        
        # Define popularity tiers
        if len(pop_data) > 0:
            low_pop = pop_data[pop_data <= 1].count()
            med_pop = pop_data[(pop_data > 1) & (pop_data <= 5)].count()
            high_pop = pop_data[pop_data > 5].count()
            
            print(f"   📈 Popularity Tiers:")
            print(f"   Low (≤1): {low_pop:,} movies ({low_pop/len(pop_data)*100:.1f}%)")
            print(f"   Medium (1-5): {med_pop:,} movies ({med_pop/len(pop_data)*100:.1f}%)")
            print(f"   High (>5): {high_pop:,} movies ({high_pop/len(pop_data)*100:.1f}%)")
    
    # 4. Create Summary Report
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY REPORT")
    print(f"{'='*80}")
    
    # Data completeness summary
    print(f"\n🗂️  DATA COMPLETENESS:")
    completeness_cols = ['year', 'runtime', 'budget', 'revenue', 'directors_main', 'cast_main', 'genres', 'original_language']
    for col in completeness_cols:
        if col in df.columns:
            if col in ['directors_main', 'cast_main', 'genres', 'keywords']:
                # For list columns, count non-empty lists
                non_empty = sum(1 for x in df[col] if safe_eval_list(x))
                completeness = non_empty / len(df) * 100
            else:
                completeness = (1 - df[col].isnull().sum() / len(df)) * 100
            print(f"   {col}: {completeness:.1f}%")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    if 'year' in results and results['year']:
        year_stats = results['year']
        print(f"   📅 Years span from {year_stats['min']:.0f} to {year_stats['max']:.0f}")
        print(f"      Most movies are from recent decades (median: {year_stats['median']:.0f})")
    
    if 'runtime' in results and results['runtime']:
        runtime_stats = results['runtime']
        print(f"   ⏱️  Runtime ranges from {runtime_stats['min']:.0f} to {runtime_stats['max']:.0f} minutes")
        print(f"      Average runtime: {runtime_stats['mean']:.0f} minutes")
    
    if 'genres' in results and results['genres']:
        genres_stats = results['genres']
        print(f"   🎭 {genres_stats['unique_items']} unique genres across all movies")
        print(f"      Average {genres_stats['avg_per_movie']:.1f} genres per movie")
    
    if 'directors_main' in results and results['directors_main']:
        directors_stats = results['directors_main']
        print(f"   🎬 {directors_stats['unique_items']} unique main directors")
    
    if 'cast_main' in results and results['cast_main']:
        cast_stats = results['cast_main']
        print(f"   🎭 {cast_stats['unique_items']} unique main cast members")
    
    if 'keywords' in results and results['keywords']:
        keywords_stats = results['keywords']
        print(f"   🏷️  {keywords_stats['unique_items']} unique keywords")
        print(f"      Average {keywords_stats['avg_per_movie']:.1f} keywords per movie")
    
    # Feature Engineering Recommendations
    print(f"\n🔧 FEATURE ENGINEERING RECOMMENDATIONS:")
    print(f"   📊 Numerical Features:")
    print(f"      • Year: Consider decade/era binning (1990s, 2000s, etc.)")
    print(f"      • Runtime: Consider binning (short <90, medium 90-120, long >120)")
    if 'budget' in results and results['budget']:
        print(f"      • Budget: Use log transformation due to wide range")
    if 'revenue' in results and results['revenue']:
        print(f"      • Revenue: Use log transformation, consider ROI calculation")
    
    print(f"   🎭 Categorical Features:")
    print(f"      • Genres: One-hot encoding or genre popularity scores")
    print(f"      • Directors/Cast: Consider reputation scores or collaboration features")
    print(f"      • Language: Group less common languages or use language family")
    print(f"      • Keywords: TF-IDF vectorization or keyword clustering")
    
    # Create visualizations
    output_dir = Path(input_file).parent
    create_visualizations(df, output_dir)
    
    print(f"\n✅ Analysis complete! Check visualizations in {output_dir / 'visualizations'}/")

if __name__ == "__main__":
    main()

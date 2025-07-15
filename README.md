# Letterboxd Hybrid Recommendation System

A production-ready hybrid movie recommendation system that combines collaborative filtering and content-based filtering to generate personalized movie recommendations for Letterboxd users.

## Overview

This system leverages both collaborative filtering and content-based filtering to provide curated movie recommendations. Users simply input their Letterboxd username, and the model outputs a personalized list of 25 movie recommendations with predicted ratings, weighted by both user review patterns and movie metadata features.

### Key Features

- **Hybrid Approach**: Combines collaborative filtering (user-movie interactions) with content-based filtering (movie metadata)
- **Real-time Data**: Always scrapes fresh user data to ensure recommendations reflect current preferences
- **Hyperparameter Optimization**: Uses cross-validation to optimize alpha (collaborative vs content balance) and L2 regularization
- **Production Ready**: Clean, scalable architecture with proper error handling and logging

## Architecture

### Data Sources

1. **User Reviews**: Scraped from Letterboxd user profiles
   - Explicit feedback: 1-10 star ratings
   - Implicit feedback: Movies in watchlists/favorites without ratings

2. **Movie Metadata**: Retrieved from TMDB API
   - Basic info: Title, year, runtime, genres
   - People: Directors, cast, production companies
   - Financial: Budget, revenue
   - Additional: Keywords, original language

## Data Pipeline

### 1. Data Scraping

The system pulls review data from the top 200 Letterboxd users to build a comprehensive base dataset:

- **Explicit Feedback**: Movies with 1-10 star ratings
- **Implicit Feedback**: Movies in lists/watchlists without explicit ratings
- **Movie Verification**: All unique titles are cross-referenced with TMDB API to gather:
  - TMDB ID, release date, runtime
  - Genres, directors, cast, keywords
  - Production companies, original language
  - Budget and revenue data

### 2. Data Cleaning

The data undergoes extensive cleaning to ensure quality:

#### Movie Filtering
- Remove films with fewer than 3 reviews
- Consolidate incomplete or duplicate entries

#### People Aggregation
- Directors with fewer than 3 movies → `directors_other`
- Actors with fewer than 5 appearances → removed from cast features
- Production companies with minimal presence → consolidated

#### Feature Bucketing
- **Year Buckets**: Pre-1930s, 1930-1959, 1960-1979, 1980-1999, 2000-2019, 2020-present
- **Runtime Buckets**: <90min, 90-120min, 120-150min, 150min+
- **Budget Buckets**: Small, Medium, Large (based on distribution percentiles)
- **Revenue Buckets**: Small, Medium, Large (based on distribution percentiles)

## Model Architecture

### Collaborative Filtering

Uses **Singular Value Decomposition (SVD)** on the user-movie interaction matrix:

```python
# User-movie interaction matrix
interaction_matrix = create_user_movie_matrix(ratings_data)

# SVD decomposition
svd_model = TruncatedSVD(n_components=25)
user_factors = svd_model.fit_transform(interaction_matrix)
movie_factors = svd_model.components_
```

### Content-Based Filtering

Processes movie metadata through multiple feature extraction methods:

#### Text Features (TF-IDF)
- **Genres**: Drama, Comedy, Action, etc.
- **Directors**: Christopher Nolan, Martin Scorsese, etc.
- **Cast**: Tom Hanks, Meryl Streep, etc.
- **Keywords**: Space, romance, thriller, etc.
- **Production Companies**: Warner Bros, Disney, etc.

#### Categorical Features (One-Hot Encoding)
- **Year Buckets**: Temporal movie preferences
- **Runtime Buckets**: Length preferences
- **Budget/Revenue Buckets**: Production scale preferences
- **Original Language**: Language preferences

```python
# TF-IDF for text features
tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
text_features = tfidf_vectorizer.fit_transform(movie_text_data)

# One-hot encoding for categorical features
categorical_features = one_hot_encode(categorical_data)

# Combine all features
combined_features = hstack([text_features, categorical_features])
```

### Hyperparameter Optimization

The system uses **cross-validation** to optimize key parameters:

#### Alpha Optimization
- Tests values: [0.5, 0.6, 0.7, 0.8, 0.9]
- Determines optimal balance between collaborative and content-based filtering
- Typical optimal range: 0.5-0.7

#### L2 Regularization
- Tests values: [0.1, 0.5, 1.0, 2.0, 5.0]
- Prevents overfitting in content-based features
- Uses Ridge regression for content model training

```python
# Cross-validation for hyperparameter tuning
for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for l2_reg in [0.1, 0.5, 1.0, 2.0, 5.0]:
        validation_score = validate_parameters(alpha, l2_reg)
        if validation_score < best_score:
            best_alpha, best_l2 = alpha, l2_reg
```

### Hybrid Scoring

The final recommendation score combines both approaches:

```python
# Collaborative filtering score
cf_score = dot_product(user_factors[user_id], movie_factors[movie_id])

# Content-based score (using trained Ridge model)
cb_score = content_model.predict(combined_features[movie_id])

# Hybrid score
hybrid_score = alpha * cf_score + (1 - alpha) * cb_score

# Convert to 1-10 scale
predicted_rating = clip(hybrid_score * 10, 1, 10)
```

## Usage

### Command Line Interface

```bash
# Generate recommendations for a user
python recommendation_engine.py <username>

# Example
python recommendation_engine.py schaffrillas
```

### Python API

```python
from recommendation_engine import LetterboxdRecommendationEngine

# Create engine
engine = LetterboxdRecommendationEngine()

# Generate recommendations
recommendations = engine.generate_recommendations('username')
```

### Web API

```bash
# Start Flask server
python api.py

# Get recommendations
curl http://localhost:5000/recommendations/<username>
```

## Performance

### Training Performance
- **Dataset Size**: 352,910 reviews, 7,189 movies, 198 users
- **Training Time**: ~45 seconds (including hyperparameter optimization)
- **Memory Usage**: ~200MB during training
- **Feature Dimensionality**: ~1,291 combined features

### Prediction Performance
- **Recommendation Generation**: ~5 seconds per user
- **Cross-validation**: ~30 seconds for parameter optimization
- **Scalability**: Handles 10,000+ movies and 1,000+ users efficiently

### Model Quality
- **Validation RMSE**: Typically 0.20-0.30 (on 0-1 scale)
- **Optimal Parameters**: Usually alpha=0.5-0.7, L2=0.1-0.5
- **Coverage**: Recommends from full movie catalog excluding user's rated films

## Project Structure

```
letterboxd_recommendation_model/
├── recommendation_engine.py    # Main hybrid recommendation engine
├── demo.py                     # Demo script for testing
├── api.py                      # Flask API interface
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
├── data/                      # Clean, processed data
│   ├── cleaned_reviews_v5.csv        # User reviews and ratings
│   └── cleaned_movie_metadata_v5.csv # Movie metadata features
├── scrapers/                  # Data collection scripts
│   ├── rating_scraper_pro.py  # User rating scraper
│   ├── tmdb_metadata_scraper.py # Movie metadata scraper
│   └── batch_scraper.py       # Batch processing utilities
├── data_cleaning/             # Data preprocessing pipeline
│   ├── clean_movie_metadata.py # Movie metadata cleaning
│   └── clean_reviews_data.py   # Review data cleaning
├── output/                    # Generated recommendations
│   └── recommendations_*.csv   # Timestamped recommendation files
└── models/                    # Model artifacts (empty - retrained each time)
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Internet connection (for scraping)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd letterboxd_recommendation_model

# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py
```

### Dependencies

```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning
scipy>=1.11.0          # Scientific computing
requests>=2.31.0       # HTTP requests
beautifulsoup4>=4.12.0 # Web scraping
selenium>=4.15.0       # Dynamic web scraping
flask>=2.3.0           # Web API (optional)
```

## Example Output

```
🎬 Top 10 Recommendations for jaaackd:
------------------------------------------------------------
 1. Gone Girl (2014.0)
    Rating: 3.8/10
    Genres: Mystery; Thriller; Drama

 2. Lady Bird (2017.0)
    Rating: 3.8/10
    Genres: Drama; Comedy

 3. Challengers (2024.0)
    Rating: 3.8/10
    Genres: Drama; Romance

 4. Oppenheimer (2023.0)
    Rating: 3.7/10
    Genres: Drama; History

 5. The Social Network (2010.0)
    Rating: 3.7/10
    Genres: Drama
```

## Configuration

### Model Parameters

```python
engine = LetterboxdRecommendationEngine(
    alpha=0.7,              # Default collaborative vs content balance
    num_recommendations=25,  # Number of recommendations to generate
    num_components=25,      # SVD components for collaborative filtering
    l2_reg=1.0,            # L2 regularization strength
    optimize_alpha=True     # Enable hyperparameter optimization
)
```

### Hyperparameter Ranges

- **Alpha**: [0.5, 0.6, 0.7, 0.8, 0.9] - Balance between collaborative and content-based
- **L2 Regularization**: [0.1, 0.5, 1.0, 2.0, 5.0] - Regularization strength for content model
- **SVD Components**: 25 (fixed) - Dimensionality of user/movie factors
- **Cross-validation Folds**: 5 - Number of validation folds

## Technical Details

### Collaborative Filtering
- **Algorithm**: Truncated SVD (Singular Value Decomposition)
- **Matrix**: User-movie interaction matrix with ratings scaled to 0-1
- **Implicit Feedback**: Unrated movies in user lists get score of 0.1
- **Dimensionality**: 25 latent factors for users and movies

### Content-Based Filtering
- **Text Processing**: TF-IDF vectorization with 300 max features per text field
- **Categorical Encoding**: One-hot encoding for bucketed numerical features
- **Regularization**: Ridge regression with L2 penalty to prevent overfitting
- **Feature Combination**: Horizontal stacking of all feature matrices

### Validation Strategy
- **Method**: Hold-out validation on 20% of user ratings
- **Metric**: Root Mean Square Error (RMSE)
- **Sample Size**: Up to 50 users with 5+ ratings for validation
- **Grid Search**: Exhaustive search over alpha and L2 parameter space

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [Letterboxd](https://letterboxd.com) for providing the inspiration and user data
- [TMDB](https://www.themoviedb.org) for comprehensive movie metadata
- scikit-learn for machine learning utilities
- The open-source community for foundational tools and libraries

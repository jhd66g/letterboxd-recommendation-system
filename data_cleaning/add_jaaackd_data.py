#!/usr/bin/env python3
"""
Script to add jaaackd's scraped ratings to the cleaned reviews dataset.
"""

import pandas as pd
import numpy as np

def add_jaaackd_ratings():
    """Add jaaackd's scraped ratings to the cleaned reviews dataset."""
    
    # Load the existing cleaned reviews
    print("Loading existing cleaned reviews...")
    reviews_df = pd.read_csv('/Users/jackduncan/Desktop/Letterboxd/datacleaning/cleaned_reviews.csv')
    print(f"Loaded {len(reviews_df)} existing reviews")
    
    # Load jaaackd's scraped ratings
    print("Loading jaaackd's scraped ratings...")
    jaaackd_df = pd.read_csv('/Users/jackduncan/Desktop/Letterboxd/jaaackd_ratings_new.csv')
    print(f"Loaded {len(jaaackd_df)} jaaackd ratings")
    
    # Get the next user_id for jaaackd
    max_user_id = reviews_df['user_id'].max()
    jaaackd_user_id = max_user_id + 1
    print(f"Assigning user_id {jaaackd_user_id} to jaaackd")
    
    # Get the next movie_id for new movies
    max_movie_id = reviews_df['movie_id'].max()
    next_movie_id = max_movie_id + 1
    
    # Create a mapping of existing movies (title + year) to movie_id
    existing_movies = reviews_df[['title', 'year', 'movie_id']].drop_duplicates()
    movie_mapping = {}
    for _, row in existing_movies.iterrows():
        key = (row['title'], row['year'])
        movie_mapping[key] = row['movie_id']
    
    # Process jaaackd's ratings
    jaaackd_reviews = []
    current_movie_id = next_movie_id
    
    for _, row in jaaackd_df.iterrows():
        title = row['title']
        year = row['year'] if pd.notna(row['year']) else None
        rating = row['rating']
        
        # Try to find existing movie
        movie_key = (title, year)
        if movie_key in movie_mapping:
            movie_id = movie_mapping[movie_key]
        else:
            # Try without year if year is None
            if year is None:
                # Look for the movie with the same title (any year)
                matches = existing_movies[existing_movies['title'] == title]
                if len(matches) > 0:
                    # Use the first match
                    movie_id = matches.iloc[0]['movie_id']
                    year = matches.iloc[0]['year']
                else:
                    # New movie
                    movie_id = current_movie_id
                    current_movie_id += 1
            else:
                # New movie
                movie_id = current_movie_id
                current_movie_id += 1
        
        # Create review entry
        review_entry = {
            'user_id': jaaackd_user_id,
            'movie_id': movie_id,
            'username': 'jaaackd',
            'tmdb_id': np.nan,  # We don't have TMDB IDs for scraped data
            'title': title,
            'year': year,
            'release_year': year,
            'rating': float(rating),
            'rating_type': 'explicit',
            'has_rating': True
        }
        jaaackd_reviews.append(review_entry)
    
    # Convert to DataFrame
    jaaackd_reviews_df = pd.DataFrame(jaaackd_reviews)
    print(f"Created {len(jaaackd_reviews_df)} review entries for jaaackd")
    
    # Combine with existing reviews
    print("Combining with existing reviews...")
    combined_df = pd.concat([reviews_df, jaaackd_reviews_df], ignore_index=True)
    
    # Save the updated dataset
    output_path = '/Users/jackduncan/Desktop/Letterboxd/datacleaning/cleaned_reviews_with_jaaackd.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Saved updated reviews to {output_path}")
    
    print(f"\nSummary:")
    print(f"Total reviews: {len(combined_df)}")
    print(f"Total users: {combined_df['username'].nunique()}")
    print(f"jaaackd's reviews: {len(jaaackd_reviews_df)}")
    print(f"jaaackd's average rating: {jaaackd_reviews_df['rating'].mean():.2f}")
    
    return combined_df

if __name__ == "__main__":
    add_jaaackd_ratings()

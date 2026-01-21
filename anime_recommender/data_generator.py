"""Sample data generator for testing the recommender system."""

import pandas as pd
import numpy as np


def generate_sample_anime_data(n_anime=100):
    """Generate sample anime dataset.
    
    Args:
        n_anime: Number of anime to generate
        
    Returns:
        DataFrame with anime information
    """
    genres = [
        'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy',
        'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life'
    ]
    
    anime_data = []
    for i in range(1, n_anime + 1):
        # Random genres (1-3 per anime)
        n_genres = np.random.randint(1, 4)
        anime_genres = ' '.join(np.random.choice(genres, n_genres, replace=False))
        
        anime_data.append({
            'anime_id': i,
            'title': f'Anime {i}',
            'genres': anime_genres,
            'description': f'An exciting {anime_genres.lower()} anime with great story.'
        })
    
    return pd.DataFrame(anime_data)


def generate_sample_ratings_data(n_users=50, n_anime=100, n_ratings=1000):
    """Generate sample user ratings dataset.
    
    Args:
        n_users: Number of users
        n_anime: Number of anime
        n_ratings: Total number of ratings
        
    Returns:
        DataFrame with user ratings
    """
    ratings_data = []
    
    for _ in range(n_ratings):
        user_id = np.random.randint(1, n_users + 1)
        anime_id = np.random.randint(1, n_anime + 1)
        rating = np.random.randint(1, 11)  # Ratings from 1-10
        
        ratings_data.append({
            'user_id': user_id,
            'anime_id': anime_id,
            'rating': rating
        })
    
    # Remove duplicates (same user rating same anime multiple times)
    df = pd.DataFrame(ratings_data)
    df = df.drop_duplicates(subset=['user_id', 'anime_id'])
    
    return df


def save_sample_data(anime_path='data/anime.csv', ratings_path='data/ratings.csv'):
    """Generate and save sample data to CSV files.
    
    Args:
        anime_path: Path to save anime data
        ratings_path: Path to save ratings data
    """
    anime_df = generate_sample_anime_data()
    ratings_df = generate_sample_ratings_data()
    
    anime_df.to_csv(anime_path, index=False)
    ratings_df.to_csv(ratings_path, index=False)
    
    print(f"Sample data saved to {anime_path} and {ratings_path}")
    print(f"Generated {len(anime_df)} anime and {len(ratings_df)} ratings")
    
    return anime_df, ratings_df


if __name__ == '__main__':
    save_sample_data()

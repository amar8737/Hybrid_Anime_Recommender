#!/usr/bin/env python3
"""Command-line interface for the Hybrid Anime Recommender."""

import argparse
import pandas as pd
from anime_recommender import HybridAnimeRecommender


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Anime Recommender System'
    )
    parser.add_argument(
        '--anime-data',
        default='data/anime.csv',
        help='Path to anime data CSV file'
    )
    parser.add_argument(
        '--ratings-data',
        default='data/ratings.csv',
        help='Path to ratings data CSV file'
    )
    parser.add_argument(
        '--user-id',
        type=int,
        help='User ID for collaborative filtering recommendations'
    )
    parser.add_argument(
        '--anime-id',
        type=int,
        help='Anime ID for content-based recommendations'
    )
    parser.add_argument(
        '--n-recommendations',
        type=int,
        default=10,
        help='Number of recommendations to generate'
    )
    parser.add_argument(
        '--content-weight',
        type=float,
        default=0.5,
        help='Weight for content-based filtering (0-1)'
    )
    
    args = parser.parse_args()
    
    # Validate content_weight
    if not 0 <= args.content_weight <= 1:
        parser.error("content-weight must be between 0 and 1")
    
    # Validate inputs
    if args.user_id is None and args.anime_id is None:
        parser.error("At least one of --user-id or --anime-id must be provided")
    
    # Load data
    print("Loading data...")
    try:
        anime_df = pd.read_csv(args.anime_data)
        ratings_df = pd.read_csv(args.ratings_data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python -m anime_recommender.data_generator' to generate sample data")
        return 1
    
    print(f"Loaded {len(anime_df)} anime and {len(ratings_df)} ratings")
    
    # Create and fit recommender
    print("Training hybrid recommender...")
    collab_weight = 1.0 - args.content_weight
    recommender = HybridAnimeRecommender(
        content_weight=args.content_weight,
        collab_weight=collab_weight
    )
    recommender.fit(anime_df, ratings_df)
    
    # Get recommendations
    print("\nGenerating recommendations...")
    recommendations = recommender.recommend(
        user_id=args.user_id,
        anime_id=args.anime_id,
        n_recommendations=args.n_recommendations
    )
    
    # Display results
    print(f"\nTop {len(recommendations)} Recommendations:")
    print("-" * 60)
    
    for i, (anime_id, score) in enumerate(recommendations, 1):
        anime_matches = anime_df[anime_df['anime_id'] == anime_id]
        if anime_matches.empty:
            print(f"{i}. Unknown Anime (ID: {anime_id})")
            print(f"   Score: {score:.4f}")
            print()
            continue
        anime_info = anime_matches.iloc[0]
        print(f"{i}. {anime_info['title']} (ID: {anime_id})")
        print(f"   Genres: {anime_info['genres']}")
        print(f"   Score: {score:.4f}")
        print()
    
    return 0


if __name__ == '__main__':
    exit(main())

"""Hybrid Recommender

Combines content-based and collaborative filtering approaches.
"""

import numpy as np
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender


class HybridAnimeRecommender:
    """Hybrid recommender combining content-based and collaborative filtering."""
    
    def __init__(self, content_weight=0.5, collab_weight=0.5, rating_range=(1, 10)):
        """Initialize hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations (0-1)
            collab_weight: Weight for collaborative filtering recommendations (0-1)
            rating_range: Tuple of (min_rating, max_rating) for normalization
        """
        if not np.isclose(content_weight + collab_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.rating_range = rating_range
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.is_fitted = False
        
    def fit(self, anime_df, ratings_df):
        """Fit both recommender models.
        
        Args:
            anime_df: DataFrame with anime information
            ratings_df: DataFrame with user ratings
        """
        self.content_recommender.fit(anime_df)
        self.collaborative_recommender.fit(ratings_df)
        self.is_fitted = True
        return self
        
    def recommend(self, user_id=None, anime_id=None, n_recommendations=10):
        """Get hybrid recommendations.
        
        Args:
            user_id: User ID for collaborative filtering (optional)
            anime_id: Anime ID for content-based filtering (optional)
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (anime_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        recommendations = {}
        
        # Get content-based recommendations if anime_id provided
        if anime_id is not None:
            content_recs = self.content_recommender.recommend(
                anime_id, 
                n_recommendations * 2
            )
            for aid, score in content_recs:
                recommendations[aid] = recommendations.get(aid, 0) + score * self.content_weight
                
        # Get collaborative recommendations if user_id provided
        if user_id is not None:
            collab_recs = self.collaborative_recommender.recommend(
                user_id, 
                n_recommendations * 2
            )
            min_rating, max_rating = self.rating_range
            for aid, score in collab_recs:
                # Normalize collaborative scores to 0-1 range
                normalized_score = (score - min_rating) / (max_rating - min_rating)
                recommendations[aid] = recommendations.get(aid, 0) + normalized_score * self.collab_weight
                
        # Sort and return top N
        sorted_recs = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return sorted_recs

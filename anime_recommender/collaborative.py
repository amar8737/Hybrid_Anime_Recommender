"""Collaborative Filtering Recommender

Recommends anime based on user rating patterns using matrix factorization.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


class CollaborativeRecommender:
    """Collaborative filtering recommender using matrix factorization."""
    
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.anime_factors = None
        self.user_ids = None
        self.anime_ids = None
        self.mean_rating = 0
        
    def fit(self, ratings_df):
        """Fit the collaborative filtering model.
        
        Args:
            ratings_df: DataFrame with columns ['user_id', 'anime_id', 'rating']
        """
        # Create user-anime rating matrix
        self.user_ids = ratings_df['user_id'].unique()
        self.anime_ids = ratings_df['anime_id'].unique()
        
        user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        anime_id_map = {aid: idx for idx, aid in enumerate(self.anime_ids)}
        
        # Build sparse matrix
        rows = ratings_df['user_id'].map(user_id_map).values
        cols = ratings_df['anime_id'].map(anime_id_map).values
        data = ratings_df['rating'].values
        
        self.mean_rating = np.mean(data)
        data_centered = data - self.mean_rating
        
        rating_matrix = csr_matrix(
            (data_centered, (rows, cols)),
            shape=(len(self.user_ids), len(self.anime_ids))
        )
        
        # Apply SVD
        self.user_factors = self.svd.fit_transform(rating_matrix)
        self.anime_factors = self.svd.components_.T
        
        return self
        
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """Get recommendations for a user.
        
        Args:
            user_id: ID of the user to recommend for
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude already rated anime
            
        Returns:
            List of tuples (anime_id, predicted_rating)
        """
        if self.user_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Find user index
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
        except IndexError:
            raise ValueError(f"User ID {user_id} not found")
            
        # Predict ratings for all anime
        user_vector = self.user_factors[user_idx]
        predicted_ratings = np.dot(user_vector, self.anime_factors.T) + self.mean_rating
        
        # Get top recommendations
        top_indices = predicted_ratings.argsort()[::-1][:n_recommendations]
        
        recommendations = [
            (self.anime_ids[i], predicted_ratings[i]) 
            for i in top_indices
        ]
        
        return recommendations

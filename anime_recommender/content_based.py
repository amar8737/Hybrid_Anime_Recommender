"""Content-Based Filtering Recommender

Recommends anime based on content features like genres, tags, and descriptions.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-based recommender using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_matrix = None
        self.anime_ids = None
        
    def fit(self, anime_df):
        """Fit the content-based model.
        
        Args:
            anime_df: DataFrame with columns ['anime_id', 'title', 'genres', 'description']
        """
        # Combine text features
        anime_df = anime_df.copy()
        anime_df['combined_features'] = (
            anime_df['genres'].fillna('') + ' ' + 
            anime_df.get('description', '').fillna('')
        )
        
        # Create TF-IDF matrix
        self.feature_matrix = self.vectorizer.fit_transform(
            anime_df['combined_features']
        )
        self.anime_ids = anime_df['anime_id'].values
        
        return self
        
    def recommend(self, anime_id, n_recommendations=10):
        """Get recommendations for a given anime.
        
        Args:
            anime_id: ID of the anime to base recommendations on
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (anime_id, similarity_score)
        """
        if self.feature_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Find index of anime_id
        try:
            idx = np.where(self.anime_ids == anime_id)[0][0]
        except IndexError:
            raise ValueError(f"Anime ID {anime_id} not found")
            
        # Calculate similarity scores
        similarities = cosine_similarity(
            self.feature_matrix[idx:idx+1], 
            self.feature_matrix
        ).flatten()
        
        # Get top similar anime (excluding the anime itself)
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = [
            (self.anime_ids[i], similarities[i]) 
            for i in similar_indices
        ]
        
        return recommendations

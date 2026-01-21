"""Hybrid Anime Recommender System

A recommendation system combining content-based and collaborative filtering approaches.
"""

__version__ = "0.1.0"

from .recommender import HybridAnimeRecommender
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

__all__ = [
    "HybridAnimeRecommender",
    "ContentBasedRecommender", 
    "CollaborativeRecommender",
]

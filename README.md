# Hybrid Anime Recommender

A hybrid recommendation system for anime that combines content-based filtering and collaborative filtering approaches.

## Features

- **Content-Based Filtering**: Recommends anime based on genres and descriptions using TF-IDF and cosine similarity
- **Collaborative Filtering**: Recommends anime based on user rating patterns using matrix factorization (SVD)
- **Hybrid Approach**: Combines both methods with configurable weights for optimal recommendations
- **Easy to Use**: Simple CLI interface and Python API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amar8737/Hybrid_Anime_Recommender.git
cd Hybrid_Anime_Recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### Generate Sample Data

```bash
python -m anime_recommender.data_generator
```

This will create sample anime and ratings data in the `data/` directory.

### Using the CLI

Get recommendations for a specific user:
```bash
python -m anime_recommender.cli --user-id 1 --n-recommendations 5
```

Get recommendations similar to a specific anime:
```bash
python -m anime_recommender.cli --anime-id 5 --n-recommendations 5
```

Get hybrid recommendations (combining both approaches):
```bash
python -m anime_recommender.cli --user-id 1 --anime-id 5 --n-recommendations 5 --content-weight 0.6
```

### Using the Python API

```python
import pandas as pd
from anime_recommender import HybridAnimeRecommender

# Load data
anime_df = pd.read_csv('data/anime.csv')
ratings_df = pd.read_csv('data/ratings.csv')

# Create and train recommender
recommender = HybridAnimeRecommender(content_weight=0.5, collab_weight=0.5)
recommender.fit(anime_df, ratings_df)

# Get recommendations
recommendations = recommender.recommend(user_id=1, anime_id=5, n_recommendations=10)

for anime_id, score in recommendations:
    print(f"Anime ID: {anime_id}, Score: {score:.4f}")
```

## Data Format

### Anime Data (anime.csv)
- `anime_id`: Unique identifier for each anime
- `title`: Name of the anime
- `genres`: Space-separated list of genres
- `description`: Text description of the anime

### Ratings Data (ratings.csv)
- `user_id`: Unique identifier for each user
- `anime_id`: ID of the rated anime
- `rating`: Rating score (typically 1-10)

## How It Works

The hybrid recommender combines two complementary approaches:

1. **Content-Based Filtering**: Analyzes anime features (genres, descriptions) to find similar anime. Uses TF-IDF vectorization and cosine similarity.

2. **Collaborative Filtering**: Learns from user rating patterns to predict preferences. Uses Singular Value Decomposition (SVD) for matrix factorization.

3. **Hybrid Combination**: Merges recommendations from both methods using weighted averaging, allowing you to balance between content similarity and user preferences.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
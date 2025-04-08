"""
Preprocessing utilities for the BookHaven recommendation system.
This module handles data cleaning, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def ensure_nltk_dependencies():
    """
    Ensure NLTK dependencies are downloaded.
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


def clean_text(text: str) -> str:
    """
    Clean text by removing punctuation, numbers, and extra whitespace.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_and_stem(text: str) -> List[str]:
    """
    Tokenize and stem text for NLP processing.
    
    Args:
        text (str): Text to tokenize and stem
        
    Returns:
        List[str]: List of stemmed tokens
    """
    ensure_nltk_dependencies()
    
    # Clean the text
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stem
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return stemmed_tokens


def preprocess_book_features(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess book features for modeling.
    
    Args:
        books_df (pd.DataFrame): Book data
        
    Returns:
        pd.DataFrame: Preprocessed book data
    """
    # Make a copy to avoid modifying the original
    df = books_df.copy()
    
    # Create a 'genres_str' column that joins genres into a string
    if 'genres' in df.columns:
        df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    # Create a combined text field for content-based filtering
    text_features = []
    
    if 'title' in df.columns:
        text_features.append(df['title'].fillna(''))
    
    if 'authors' in df.columns:
        # Handle different formats of authors field
        if df['authors'].apply(lambda x: isinstance(x, list)).any():
            df['authors_str'] = df['authors'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        else:
            df['authors_str'] = df['authors'].fillna('')
        text_features.append(df['authors_str'])
    
    if 'genres_str' in df.columns:
        text_features.append(df['genres_str'])
    
    if 'description' in df.columns:
        text_features.append(df['description'].fillna(''))
    
    # Combine all text features
    df['combined_features'] = pd.concat(text_features, axis=1).apply(
        lambda x: ' '.join(x.dropna().astype(str)), axis=1
    )
    
    # Clean combined features
    df['combined_features'] = df['combined_features'].apply(clean_text)
    
    return df


def normalize_ratings(ratings: pd.Series) -> pd.Series:
    """
    Normalize ratings to a scale of 0 to 1.
    
    Args:
        ratings (pd.Series): Series of ratings
        
    Returns:
        pd.Series: Normalized ratings
    """
    min_rating = ratings.min()
    max_rating = ratings.max()
    
    if min_rating == max_rating:
        return pd.Series(0.5, index=ratings.index)
    
    normalized = (ratings - min_rating) / (max_rating - min_rating)
    
    return normalized


def create_user_profile(user_interactions: pd.DataFrame, books_df: pd.DataFrame,
                       feature_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Create a user profile based on their interactions.
    
    Args:
        user_interactions (pd.DataFrame): User's interactions with books
        books_df (pd.DataFrame): Book data
        feature_weights (Dict[str, float], optional): Weights for different features
        
    Returns:
        Dict[str, float]: User profile as a dictionary of genre preferences
    """
    if feature_weights is None:
        feature_weights = {
            'rating': 1.0,
            'recency': 0.5,
        }
    
    # Initialize profile
    profile = {}
    
    # If user has no interactions, return empty profile
    if len(user_interactions) == 0:
        return profile
    
    # Merge interactions with book data
    user_books = pd.merge(user_interactions, books_df, on='book_id', how='inner')
    
    # Extract genres from the user's books
    for _, row in user_books.iterrows():
        # Get the weight based on rating
        rating_weight = 1.0
        if 'rating' in row and not pd.isna(row['rating']):
            rating_weight = row['rating'] / 5.0  # Normalize to 0-1
        
        # Process genres
        if 'genres' in row and isinstance(row['genres'], list):
            for genre in row['genres']:
                if genre in profile:
                    profile[genre] += rating_weight * feature_weights['rating']
                else:
                    profile[genre] = rating_weight * feature_weights['rating']
    
    # Normalize the profile
    if profile:
        max_value = max(profile.values())
        for genre in profile:
            profile[genre] /= max_value
    
    return profile


def create_book_features_dict(books_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Create a dictionary mapping book IDs to their features.
    
    Args:
        books_df (pd.DataFrame): Book data
        
    Returns:
        Dict[int, Dict[str, Any]]: Dictionary mapping book IDs to features
    """
    book_features = {}
    
    for _, book in books_df.iterrows():
        book_id = book['book_id']
        features = {
            'title': book.get('title', ''),
            'authors': book.get('authors', ''),
            'average_rating': book.get('average_rating', 0.0),
            'genres': book.get('genres', []),
            'combined_features': book.get('combined_features', '')
        }
        book_features[book_id] = features
    
    return book_features


def calculate_interaction_matrix(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a user-item interaction matrix for collaborative filtering.
    
    Args:
        interactions_df (pd.DataFrame): Interactions data
        
    Returns:
        pd.DataFrame: User-item interaction matrix with users as rows, items as columns,
                     and ratings as values
    """
    # Check if required columns exist
    if 'user_id' not in interactions_df.columns or 'book_id' not in interactions_df.columns:
        return pd.DataFrame()
    
    # Create the matrix
    if 'rating' in interactions_df.columns:
        # Use ratings if available
        matrix = interactions_df.pivot(index='user_id', columns='book_id', values='rating')
    else:
        # Use binary interactions (1 for interaction, 0 for no interaction)
        interactions_df['interaction'] = 1
        matrix = interactions_df.pivot(index='user_id', columns='book_id', values='interaction')
    
    # Fill NaN values with 0
    matrix = matrix.fillna(0)
    
    return matrix
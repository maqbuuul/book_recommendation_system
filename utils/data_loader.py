"""
Data loader utility functions for the BookHaven recommendation system.
This module handles loading and processing the Goodreads dataset.
"""

import os
import pandas as pd
import numpy as np
import gzip
import json
import streamlit as st
from typing import Tuple, List, Dict, Any


@st.cache_data
def load_book_data() -> pd.DataFrame:
    """
    Load book data from goodreads_books.json.gz.
    
    Returns:
        pd.DataFrame: DataFrame containing book information
    """
    try:
        # Path to the compressed JSON file
        file_path = os.path.join("data", "goodreads_books.json.gz")
        
        # Read compressed JSON file
        books = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                books.append(json.loads(line))
        
        # Convert to DataFrame
        books_df = pd.DataFrame(books)
        
        # Process the DataFrame
        books_df = process_book_data(books_df)
        
        return books_df
    
    except Exception as e:
        st.error(f"Error loading book data: {str(e)}")
        # Return empty DataFrame with expected columns if loading fails
        return pd.DataFrame(columns=['book_id', 'title', 'authors', 'average_rating', 'image_url', 'genres'])


@st.cache_data
def load_interactions() -> pd.DataFrame:
    """
    Load user-book interactions from goodreads_interactions.
    
    Returns:
        pd.DataFrame: DataFrame containing user-book interactions
    """
    try:
        # Path to the interactions file
        file_path = os.path.join("data", "goodreads_interactions")
        
        # Read Excel file
        interactions_df = pd.read_excel(file_path)
        
        # Process interactions
        interactions_df = process_interactions(interactions_df)
        
        return interactions_df
    
    except Exception as e:
        st.error(f"Error loading interactions data: {str(e)}")
        # Return empty DataFrame with expected columns if loading fails
        return pd.DataFrame(columns=['user_id', 'book_id', 'rating', 'is_read'])


@st.cache_data
def load_book_id_mapping() -> pd.DataFrame:
    """
    Load book ID mapping data.
    
    Returns:
        pd.DataFrame: DataFrame mapping between internal and external book IDs
    """
    try:
        # Path to the book ID mapping file
        file_path = os.path.join("data", "book_id_map")
        
        # Read Excel file
        book_id_map_df = pd.read_excel(file_path)
        
        return book_id_map_df
    
    except Exception as e:
        st.error(f"Error loading book ID mapping: {str(e)}")
        # Return empty DataFrame with expected columns if loading fails
        return pd.DataFrame(columns=['goodreads_book_id', 'book_id'])


@st.cache_data
def load_user_id_mapping() -> pd.DataFrame:
    """
    Load user ID mapping data.
    
    Returns:
        pd.DataFrame: DataFrame mapping between internal and external user IDs
    """
    try:
        # Path to the user ID mapping file
        file_path = os.path.join("data", "user_id_map")
        
        # Read Excel file
        user_id_map_df = pd.read_excel(file_path)
        
        return user_id_map_df
    
    except Exception as e:
        st.error(f"Error loading user ID mapping: {str(e)}")
        # Return empty DataFrame with expected columns if loading fails
        return pd.DataFrame(columns=['goodreads_user_id', 'user_id'])


def process_book_data(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean the book data.
    
    Args:
        books_df (pd.DataFrame): Raw book data
        
    Returns:
        pd.DataFrame: Processed book data
    """
    # Make a copy to avoid modifying the original
    df = books_df.copy()
    
    # Extract main genres from the categories/shelves
    if 'popular_shelves' in df.columns:
        df['genres'] = df['popular_shelves'].apply(extract_genres)
    
    # Make sure required columns exist
    required_columns = ['book_id', 'title', 'authors', 'average_rating', 'image_url']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Convert average_rating to float
    if 'average_rating' in df.columns:
        df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
        df['average_rating'].fillna(0.0, inplace=True)
    
    # Make sure we have a genre column
    if 'genres' not in df.columns:
        df['genres'] = None
    
    # Fill missing values
    df['title'].fillna('Unknown Title', inplace=True)
    df['authors'].fillna('Unknown Author', inplace=True)
    df['image_url'].fillna('', inplace=True)
    
    return df


def process_interactions(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean the interactions data.
    
    Args:
        interactions_df (pd.DataFrame): Raw interactions data
        
    Returns:
        pd.DataFrame: Processed interactions data
    """
    # Make a copy to avoid modifying the original
    df = interactions_df.copy()
    
    # Ensure required columns exist
    required_columns = ['user_id', 'book_id', 'rating']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Create is_read column if it doesn't exist
    if 'is_read' not in df.columns:
        # If rating exists, assume the book was read
        df['is_read'] = df['rating'].notna()
    
    # Convert rating to float
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Remove rows where both user_id and book_id are missing
    df = df.dropna(subset=['user_id', 'book_id'], how='all')
    
    return df


def extract_genres(shelves: List[Dict[str, Any]], max_genres: int = 3) -> List[str]:
    """
    Extract the main genres from popular shelves.
    
    Args:
        shelves (List[Dict]): List of shelf dictionaries
        max_genres (int): Maximum number of genres to return
        
    Returns:
        List[str]: List of main genres
    """
    if not shelves or not isinstance(shelves, list):
        return []
    
    # Common genres to prioritize
    common_genres = {
        'fiction', 'non-fiction', 'mystery', 'thriller', 'romance', 'fantasy', 
        'science-fiction', 'sci-fi', 'horror', 'biography', 'history', 
        'young-adult', 'classics', 'contemporary', 'memoir', 'poetry',
        'philosophy', 'crime', 'adventure', 'children', 'drama'
    }
    
    # Extract shelf names and count
    genre_counts = {}
    for shelf in shelves:
        if isinstance(shelf, dict) and 'name' in shelf and 'count' in shelf:
            name = shelf['name'].lower().strip()
            count = shelf['count']
            
            # Skip to-read, read, etc.
            if name in {'to-read', 'read', 'currently-reading', 'books', 'default'}:
                continue
                
            genre_counts[name] = count
    
    # Sort by count (popularity)
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Prioritize common genres and take top max_genres
    genres = []
    
    # First, add common genres by popularity
    for genre, _ in sorted_genres:
        if genre in common_genres and len(genres) < max_genres:
            genres.append(genre)
    
    # Then, add any remaining genres by popularity
    for genre, _ in sorted_genres:
        if genre not in genres and len(genres) < max_genres:
            genres.append(genre)
            
    return genres


def get_popular_books(books_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Get popular books based on ratings.
    
    Args:
        books_df (pd.DataFrame): Book data
        n (int): Number of popular books to return
        
    Returns:
        pd.DataFrame: DataFrame with n popular books
    """
    # Make a copy to avoid modifying the original
    df = books_df.copy()
    
    # Check if the required columns exist
    if 'average_rating' not in df.columns or 'ratings_count' not in df.columns:
        # If we don't have ratings_count, just sort by average_rating
        if 'average_rating' in df.columns:
            sorted_df = df.sort_values(by='average_rating', ascending=False)
        else:
            sorted_df = df.sample(min(n, len(df)))
    else:
        # Calculate a popularity score that considers both rating and number of ratings
        # Only consider books with a minimum number of ratings
        min_ratings = 100
        popular_df = df[df['ratings_count'] >= min_ratings].copy()
        
        # If we don't have enough books, lower the threshold
        if len(popular_df) < n:
            popular_df = df.copy()
        
        # Calculate popularity score (weighted rating)
        popular_df['popularity_score'] = popular_df['average_rating'] * np.log1p(popular_df['ratings_count'])
        
        # Sort by popularity score
        sorted_df = popular_df.sort_values(by='popularity_score', ascending=False)
    
    # Return top n books
    return sorted_df.head(n)


def get_user_reading_history(user_id: int, interactions_df: pd.DataFrame, 
                           books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a user's reading history.
    
    Args:
        user_id (int): User ID
        interactions_df (pd.DataFrame): Interactions data
        books_df (pd.DataFrame): Book data
        
    Returns:
        pd.DataFrame: Books the user has read or rated
    """
    # Filter interactions for the user
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    # Filter books that the user has interacted with
    if len(user_interactions) > 0:
        user_books = pd.merge(user_interactions, books_df, on='book_id', how='inner')
        return user_books
    else:
        return pd.DataFrame(columns=books_df.columns)


def get_genres_distribution(books_df: pd.DataFrame) -> Dict[str, int]:
    """
    Get distribution of genres across all books.
    
    Args:
        books_df (pd.DataFrame): Book data
        
    Returns:
        Dict[str, int]: Dictionary mapping genre to count
    """
    genre_counts = {}
    
    # Iterate through all books
    for _, book in books_df.iterrows():
        if 'genres' in book and book['genres']:
            for genre in book['genres']:
                if genre in genre_counts:
                    genre_counts[genre] += 1
                else:
                    genre_counts[genre] = 1
    
    # Sort by count (descending)
    sorted_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_genres
"""
Recommendation algorithms for the BookHaven recommendation system.
This module provides various recommendation methods for suggesting books to users.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import logging


class ContentBasedRecommender:
    """Content-based recommendation system using book features."""
    
    def __init__(self, books_df: pd.DataFrame, model_path: str = "models/content_based.pkl"):
        """
        Initialize the content-based recommender.
        
        Args:
            books_df (pd.DataFrame): Book data
            model_path (str): Path to save/load the model
        """
        self.books_df = books_df
        self.model_path = model_path
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.similarity_matrix = None
        
        # Try to load the model if it exists
        self.load_model()
    
    def train(self, force: bool = False) -> bool:
        """
        Train the content-based model.
        
        Args:
            force (bool): Whether to force retraining even if model exists
            
        Returns:
            bool: Whether training was successful
        """
        # Check if we already have a trained model
        if not force and self.tfidf_matrix is not None and self.similarity_matrix is not None:
            return True
        
        try:
            # Check if the books_df has the required columns
            if 'combined_features' not in self.books_df.columns or len(self.books_df) == 0:
                return False
            
            # Create TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )
            
            # Fit and transform the combined features
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books_df['combined_features'])
            
            # Calculate cosine similarity matrix
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            
            # Save the model
            self.save_model()
            
            return True
        
        except Exception as e:
            logging.error(f"Error training content-based model: {str(e)}")
            return False
    
    def save_model(self) -> bool:
        """
        Save the trained model to disk.
        
        Returns:
            bool: Whether saving was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save the model components
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'similarity_matrix': self.similarity_matrix
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        
        except Exception as e:
            logging.error(f"Error saving content-based model: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            bool: Whether loading was successful
        """
        if not os.path.exists(self.model_path):
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            
            return True
        
        except Exception as e:
            logging.error(f"Error loading content-based model: {str(e)}")
            return False
    
    def recommend_similar_to_book(self, book_id: int, n: int = 10, 
                                exclude_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Recommend books similar to a given book.
        
        Args:
            book_id (int): ID of the book to find similar ones to
            n (int): Number of recommendations to return
            exclude_ids (List[int], optional): Book IDs to exclude from recommendations
            
        Returns:
            pd.DataFrame: DataFrame of recommended books
        """
        if self.similarity_matrix is None:
            success = self.train()
            if not success:
                return pd.DataFrame()
        
        # Find the index of the book in the DataFrame
        book_indices = self.books_df.index[self.books_df['book_id'] == book_id].tolist()
        
        if not book_indices:
            return pd.DataFrame()
        
        book_index = book_indices[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[book_index]))
        
        # Sort by similarity scores
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n similar books
        similar_books_indices = [i[0] for i in similarity_scores[1:n+len(exclude_ids or [])+1]]
        
        # Create a DataFrame of similar books
        similar_books = self.books_df.iloc[similar_books_indices].copy()
        
        # Add similarity score column
        similar_books['similarity_score'] = [similarity_scores[i][1] for i in range(1, len(similar_books) + 1)]
        
        # Exclude specified book IDs
        if exclude_ids:
            similar_books = similar_books[~similar_books['book_id'].isin(exclude_ids)]
        
        # Return top n
        return similar_books.head(n)
    
    def recommend_for_user_profile(self, user_profile: Dict[str, float], 
                                  n: int = 10, exclude_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Recommend books based on a user profile.
        
        Args:
            user_profile (Dict[str, float]): User's genre preferences
            n (int): Number of recommendations to return
            exclude_ids (List[int], optional): Book IDs to exclude from recommendations
            
        Returns:
            pd.DataFrame: DataFrame of recommended books
        """
        if not user_profile:
            return pd.DataFrame()
        
        # Copy the books DataFrame
        books = self.books_df.copy()
        
        # Calculate a score for each book based on genre match
        books['score'] = books.apply(
            lambda row: self._calculate_profile_match(row, user_profile),
            axis=1
        )
        
        # Sort by score
        sorted_books = books.sort_values('score', ascending=False)
        
        # Exclude specified book IDs
        if exclude_ids:
            sorted_books = sorted_books[~sorted_books['book_id'].isin(exclude_ids)]
        
        # Return top n
        return sorted_books.head(n)
    
    def _calculate_profile_match(self, book: pd.Series, user_profile: Dict[str, float]) -> float:
        """
        Calculate how well a book matches a user profile.
        
        Args:
            book (pd.Series): Book data
            user_profile (Dict[str, float]): User's genre preferences
            
        Returns:
            float: Match score
        """
        score = 0.0
        
        # Check for genre match
        if 'genres' in book and isinstance(book['genres'], list):
            for genre in book['genres']:
                if genre in user_profile:
                    score += user_profile[genre]
        
        # Consider book rating as a factor
        if 'average_rating' in book and not pd.isna(book['average_rating']):
            score *= (book['average_rating'] / 5.0)
        
        return score


class CollaborativeRecommender:
    """Collaborative filtering recommendation system using user interactions."""
    
    def __init__(self, interactions_df: pd.DataFrame, books_df: pd.DataFrame):
        """
        Initialize the collaborative recommender.
        
        Args:
            interactions_df (pd.DataFrame): User-book interactions
            books_df (pd.DataFrame): Book data
        """
        self.interactions_df = interactions_df
        self.books_df = books_df
        
        # Create user-item matrix
        self.user_item_matrix = self._create_user_item_matrix()
    
    def _create_user_item_matrix(self) -> pd.DataFrame:
        """
        Create a user-item matrix from interactions.
        
        Returns:
            pd.DataFrame: User-item matrix with ratings
        """
        if 'rating' in self.interactions_df.columns:
            # Use explicit ratings
            matrix = self.interactions_df.pivot(
                index='user_id', 
                columns='book_id', 
                values='rating'
            ).fillna(0)
        else:
            # Use implicit feedback (1 for interaction)
            self.interactions_df['rating'] = 1
            matrix = self.interactions_df.pivot(
                index='user_id', 
                columns='book_id', 
                values='rating'
            ).fillna(0)
        
        return matrix
    
    def recommend_for_user(self, user_id: int, n: int = 10, 
                        exclude_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Recommend books for a user using collaborative filtering.
        
        Args:
            user_id (int): ID of the user
            n (int): Number of recommendations to return
            exclude_ids (List[int], optional): Book IDs to exclude from recommendations
            
        Returns:
            pd.DataFrame: DataFrame of recommended books
        """
        # Check if user exists in the matrix
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Find similar users
        similar_users = self._find_similar_users(user_id)
        
        # Calculate predicted ratings for unrated books
        predictions = []
        
        for book_id in self.user_item_matrix.columns:
            # Skip books the user has already rated
            if user_ratings[book_id] > 0:
                continue
                
            # Skip excluded books
            if exclude_ids and book_id in exclude_ids:
                continue
                
            # Calculate predicted rating
            predicted_rating = self._predict_rating(user_id, book_id, similar_users)
            
            if predicted_rating > 0:
                predictions.append((book_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top n book IDs
        top_book_ids = [p[0] for p in predictions[:n]]
        
        # Get book details
        if top_book_ids:
            recommendations = self.books_df[self.books_df['book_id'].isin(top_book_ids)].copy()
            
            # Add predicted ratings
            predicted_ratings = {p[0]: p[1] for p in predictions[:n]}
            recommendations['predicted_rating'] = recommendations['book_id'].map(predicted_ratings)
            
            # Sort by predicted rating
            recommendations = recommendations.sort_values('predicted_rating', ascending=False)
            
            return recommendations
        else:
            return pd.DataFrame()
    
    def _find_similar_users(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find users similar to a given user.
        
        Args:
            user_id (int): ID of the user
            n (int): Number of similar users to return
            
        Returns:
            List[Tuple[int, float]]: List of (user_id, similarity) tuples
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's ratings vector
        user_vector = self.user_item_matrix.loc[user_id].values
        
        # Calculate similarity with all other users
        similarities = []
        
        for other_id in self.user_item_matrix.index:
            if other_id == user_id:
                continue
                
            other_vector = self.user_item_matrix.loc[other_id].values
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(user_vector, other_vector)
            
            if similarity > 0:
                similarities.append((other_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n similar users
        return similarities[:n]
    
    def _cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1 (np.ndarray): First vector
            vector2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity
        """
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def _predict_rating(self, user_id: int, book_id: int, 
                      similar_users: List[Tuple[int, float]]) -> float:
        """
        Predict rating for a user-book pair.
        
        Args:
            user_id (int): ID of the user
            book_id (int): ID of the book
            similar_users (List[Tuple[int, float]]): List of similar users
            
        Returns:
            float: Predicted rating
        """
        # Get users who have rated this book
        numerator = 0
        denominator = 0
        
        for other_id, similarity in similar_users:
            # Check if this user has rated the book
            if book_id in self.user_item_matrix.columns:
                other_rating = self.user_item_matrix.loc[other_id, book_id]
                
                if other_rating > 0:
                    numerator += similarity * other_rating
                    denominator += similarity
        
        # Calculate predicted rating
        if denominator > 0:
            return numerator / denominator
        else:
            return 0


class HybridRecommender:
    """Hybrid recommendation system combining content-based and collaborative filtering."""
    
    def __init__(self, content_recommender: ContentBasedRecommender, 
                collaborative_recommender: CollaborativeRecommender,
                books_df: pd.DataFrame):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_recommender (ContentBasedRecommender): Content-based recommender
            collaborative_recommender (CollaborativeRecommender): Collaborative recommender
            books_df (pd.DataFrame): Book data
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.books_df = books_df
    
    def recommend_for_user(self, user_id: int, user_profile: Dict[str, float],
                          n: int = 10, method: str = 'weighted',
                          weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate recommendations for a user using multiple methods.
        
        Args:
            user_id (int): ID of the user
            user_profile (Dict[str, float]): User's genre preferences
            n (int): Number of recommendations to return
            method (str): Method to combine recommendations ('weighted', 'cascade', 'mixed')
            weights (Dict[str, float], optional): Weights for different recommendation methods
            
        Returns:
            pd.DataFrame: DataFrame of recommended books
        """
        if weights is None:
            weights = {
                'collaborative': 0.7,
                'content': 0.3
            }
        
        # Get user's read books to exclude
        exclude_ids = self._get_user_read_books(user_id)
        
        if method == 'weighted':
            return self._weighted_recommendations(user_id, user_profile, n, exclude_ids, weights)
        elif method == 'cascade':
            return self._cascade_recommendations(user_id, user_profile, n, exclude_ids)
        else:  # 'mixed'
            return self._mixed_recommendations(user_id, user_profile, n, exclude_ids)
    
    def _get_user_read_books(self, user_id: int) -> List[int]:
        """
        Get books already read by the user.
        
        Args:
            user_id (int): ID of the user
            
        Returns:
            List[int]: List of book IDs read by the user
        """
        user_interactions = self.collaborative_recommender.interactions_df
        user_books = user_interactions[user_interactions['user_id'] == user_id]['book_id'].tolist()
        return user_books
    
    def _weighted_recommendations(self, user_id: int, user_profile: Dict[str, float],
                                n: int, exclude_ids: List[int],
                                weights: Dict[str, float]) -> pd.DataFrame:
        """
        Generate weighted recommendations from multiple sources.
        
        Args:
            user_id (int): ID of the user
            user_profile (Dict[str, float]): User's genre preferences
            n (int): Number of recommendations to return
            exclude_ids (List[int]): Book IDs to exclude
            weights (Dict[str, float]): Weights for different recommendation methods
            
        Returns:
            pd.DataFrame: DataFrame of recommended books
        """
        # Get more recommendations than needed to have enough after merging
        n_expanded = n * 3
        
        # Get collaborative recommendations
        collab_recs = self.collaborative_recommender.recommend_for_user(
            user_id, n_expanded, exclude_ids
        )
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend_for_user_profile(
            user_profile, n_expanded, exclude_ids
        )
        
        # Initialize final recommendations
        all_recs = pd.DataFrame()
        
        # Merge collaborative recommendations
        if not collab_recs.empty:
            collab_recs = collab_recs.copy()
            collab_recs['collab_score'] = collab_recs['predicted_rating'] / 5.0  # Normalize to 0-1
            collab_recs.drop('predicted_rating', axis=1, inplace=True, errors='ignore')
            
            all_recs = collab_recs
        
        # Merge content-based recommendations
        if not content_recs.empty:
            content_recs = content_recs.copy()
            content_recs['content_score'] = content_recs['score']  # Already should be 0-1
            content_recs.drop('score', axis=1, inplace=True, errors='ignore')
            
            if all_recs.empty:
                all_recs = content_recs
            else:
                # Merge on book_id
                all_recs = pd.merge(all_recs, content_recs[['book_id', 'content_score']], 
                                   on='book_id', how='outer')
        
        # Fill NaN scores with 0
        if 'collab_score' not in all_recs.columns:
            all_recs['collab_score'] = 0
        else:
            all_recs['collab_score'].fillna(0, inplace=True)
            
        if 'content_score' not in all_recs.columns:
            all_recs['content_score'] = 0
        else:
            all_recs['content_score'].fillna(0, inplace=True)
        
        # Calculate weighted score
        all_recs['final_score'] = (
            weights['collaborative'] * all_recs['collab_score'] +
            weights['content'] * all_recs['content_score']
        )
        
        # Sort by final score
        all_recs = all_recs.sort_values('final_score', ascending=False)
        
        # Get top n recommendations
        return all_recs.head(n)
    
    def _cascade_recommendations(self, user_id: int, user_profile: Dict[str, float],
                               n: int, exclude_ids: List[int]) -> pd.DataFrame:
        """
        Generate recommendations using a cascade approach.
        First get collaborative recommendations, then re-rank using content-based.
        
        Args:
            user_id (int): ID of the user
            user_profile (Dict[str, float]): User's genre preferences
            n (int): Number of recommendations to return
            exclude_ids (List[int]): Book IDs to exclude
            
        Returns:
            pd.DataFrame: DataFrame of recommended books
        """
        # Get more recommendations than needed
        n_expanded = n * 2
        
        # Step 1: Get collaborative recommendations
        collab_recs = self.collaborative_recommender.recommend_for_user(
            user_id, n_expanded, exclude_ids
        )
        
        if collab_recs.empty:
            # Fall back to content-based if no collaborative recommendations
            return self.content_recommender.recommend_for_user_profile(
                user_profile, n, exclude_ids
            )
        
        # Step 2: Re-rank using content-based approach
        # Calculate content-based score for each book
        collab_recs['content_score'] = collab_recs.apply(
            lambda row: self.content_recommender._calculate_profile_match(row, user_profile),
            axis=1
        )
        
        # Combine scores (give more weight to collaborative)
        collab_recs['final_score'] = (
            0.7 * (collab_recs['predicted_rating'] / 5.0) +  # Normalize to 0-1
            0.3 * collab_recs['content_score']
        )
        
        # Sort by final score
        result = collab_recs.sort_values('final_score', ascending=False)
        
        # Return top n
        return result.head(n)
    
    def _mixed_recommendations(self, user_id: int, user_profile: Dict[str, float],
                             n: int, exclude_ids: List[int]) -> pd.DataFrame:
        """
        Generate mixed recommendations - some from each method.
        
        Args:
            user_id (int): ID of the user
            user_profile (Dict[str, float]): User's genre preferences
            n (int): Number of recommendations to return
            exclude_ids (List[int]): Book IDs to exclude
            
        Returns:
            pd.DataFrame: DataFrame of recommended books
        """
        # Determine how many recommendations to get from each source
        n_collab = int(n * 0.7)  # 70% collaborative
        n_content = n - n_collab  # 30% content-based
        
        # Get collaborative recommendations
        collab_recs = self.collaborative_recommender.recommend_for_user(
            user_id, n_collab, exclude_ids
        )
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend_for_user_profile(
            user_profile, n_content, exclude_ids
        )
        
        # Combine recommendations
        collab_recs = collab_recs.copy() if not collab_recs.empty else pd.DataFrame()
        content_recs = content_recs.copy() if not content_recs.empty else pd.DataFrame()
        
        if not collab_recs.empty:
            collab_recs['source'] = 'collaborative'
        
        if not content_recs.empty:
            content_recs['source'] = 'content'
            
            # Rename score column to match with collab_recs
            if 'score' in content_recs.columns:
                content_recs['predicted_rating'] = content_recs['score'] * 5
                content_recs.drop('score', axis=1, inplace=True)
        
        # Combine recommendations
        all_recs = pd.concat([collab_recs, content_recs], ignore_index=True)
        
        # If we don't have enough recommendations, fill with popular books
        if len(all_recs) < n:
            remaining = n - len(all_recs)
            
            # Get popular books not already recommended or read
            exclude_all = exclude_ids + all_recs['book_id'].tolist()
            popular_books = self._get_popular_books(remaining, exclude_all)
            
            if not popular_books.empty:
                popular_books['source'] = 'popular'
                all_recs = pd.concat([all_recs, popular_books], ignore_index=True)
        
        return all_recs
    
    def _get_popular_books(self, n: int, exclude_ids: List[int]) -> pd.DataFrame:
        """
        Get popular books not in exclude_ids.
        
        Args:
            n (int): Number of books to return
            exclude_ids (List[int]): Book IDs to exclude
            
        Returns:
            pd.DataFrame: DataFrame of popular books
        """
        # Copy books DataFrame
        books = self.books_df.copy()
        
        # Exclude specified books
        if exclude_ids:
            books = books[~books['book_id'].isin(exclude_ids)]
        
        # Sort by average rating and popularity
        if 'ratings_count' in books.columns and 'average_rating' in books.columns:
            # Create popularity score that considers both rating and number of ratings
            books['popularity_score'] = books['average_rating'] * np.log1p(books['ratings_count'])
            sorted_books = books.sort_values('popularity_score', ascending=False)
        elif 'average_rating' in books.columns:
            sorted_books = books.sort_values('average_rating', ascending=False)
        else:
            sorted_books = books
        
        # Return top n
        return sorted_books.head(n)
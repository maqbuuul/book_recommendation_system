"""
BookHaven - Your Personal Reading Companion
Main Streamlit application file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Add the project directory to the path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.data_loader import (
    load_book_data, load_interactions, load_book_id_mapping, 
    load_user_id_mapping, get_popular_books, get_user_reading_history
)
from utils.preprocessing import (
    preprocess_book_features, create_user_profile, normalize_ratings
)
from utils.recommenders import (
    ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
)
from utils.ui_components import (
    render_header, render_section_header, render_book_row, render_profile_section,
    render_search_box, render_recommendation_methods, render_book_detail_modal,
    render_empty_state, render_pagination, render_loading, filter_books, paginate_books,
    reset_session_state
)


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Custom CSS to style the app
def local_css():
    st.markdown(
        """
    <style>
        /* Main color scheme */
        :root {
            --primary-color: #1A365D;
            --secondary-color: #F2B705;
            --background-color: #F8F9FA;
            --text-color: #333333;
            --success-color: #28A745;
            --warning-color: #FFC107;
            --danger-color: #DC3545;
        }
        
        /* Overall styling */
        .main {
            background-color: var(--background-color);
            padding: 0 !important;
        }
        
        /* Header styling */
        .header {
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #1A365D 0%, #2A4A7F 100%);
        }
        
        .header h1 {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            opacity: 0.9;
            margin-bottom: 1rem;
        }
        
        /* Card styling */
        .book-card {
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .book-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        }
        
        .book-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            background-color: #f0f0f0;
        }
        
        .book-card-content {
            padding: 1rem;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }
        
        .book-card h3 {
            font-weight: bold;
            font-size: 1rem;
            margin-bottom: 0.3rem;
            color: var(--text-color);
        }
        
        .book-card p {
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        .book-genre {
            display: inline-block;
            background-color: #E6F0FF;
            color: #1A365D;
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 1rem;
            margin-top: 0.5rem;
            align-self: flex-start;
        }
        
        .book-card-footer {
            border-top: 1px solid #eee;
            padding-top: 0.5rem;
            margin-top: 0.5rem;
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
        }
        
        .book-card-footer a {
            color: #1A365D;
            text-decoration: none;
        }
        
        .book-card-footer a:hover {
            text-decoration: underline;
        }
        
        /* Star rating */
        .rating {
            color: #F2B705;
            font-size: 0.85rem;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1A365D !important;
            color: white;
        }
        
        /* Search box styling */
        .search-box {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Section headers */
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .section-header h2 {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--text-color);
        }
        
        .section-header a {
            color: #1A365D;
            font-size: 0.9rem;
            text-decoration: none;
        }
        
        .section-header a:hover {
            text-decoration: underline;
        }
        
        /* Recommendation method cards */
        .method-card {
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 1rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .method-card:hover {
            background-color: #f5f5f5;
        }
        
        .method-card.active {
            background-color: #E6F0FF;
            border-color: #1A365D;
        }
        
        .method-card h3 {
            font-weight: 600;
            font-size: 1rem;
            color: #1A365D;
            margin-bottom: 0.5rem;
        }
        
        .method-card p {
            font-size: 0.85rem;
            color: #666;
        }
        
        /* Profile section */
        .profile-section {
            border-top: 1px solid #eee;
            padding-top: 1rem;
            margin-top: 1rem;
        }
        
        .tag {
            display: inline-block;
            background-color: #E6F0FF;
            color: #1A365D;
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 1rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .author-tag {
            background-color: #FFF5E6;
            color: #F2B705;
        }
        
        /* Button styling */
        .primary-button {
            background-color: #1A365D;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .primary-button:hover {
            background-color: #0F2A4A;
        }
        
        .secondary-button {
            background-color: #F2B705;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .secondary-button:hover {
            background-color: #E5A800;
        }
        
        /* Fix Streamlit elements */
        div.row-widget.stRadio > div {
            flex-direction: row;
            align-items: center;
        }
        
        div.row-widget.stRadio > div > label {
            margin: 0 0.5rem;
        }
        
        /* Remove padding from Streamlit containers */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Streamlit sidebar title */
        .sidebar-title {
            margin-left: 20px;
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu, footer, header {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Load and preprocess data
@st.cache_resource
def load_data():
    """
    Load and preprocess all data needed for the app.
    
    Returns:
        Dict: Dictionary containing all data
    """
    logging.info("Loading data...")
    
    # Load raw data
    books_df = load_book_data()
    interactions_df = load_interactions()
    book_id_map_df = load_book_id_mapping()
    user_id_map_df = load_user_id_mapping()
    
    # Preprocess book features
    logging.info("Preprocessing book features...")
    books_df = preprocess_book_features(books_df)
    
    # Create recommenders
    logging.info("Creating recommendation models...")
    content_recommender = ContentBasedRecommender(books_df)
    collaborative_recommender = CollaborativeRecommender(interactions_df, books_df)
    hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender, books_df)
    
    # Extract available genres
    if 'genres' in books_df.columns and books_df['genres'].apply(lambda x: isinstance(x, list)).any():
        all_genres = set()
        for genres in books_df['genres'].dropna():
            if isinstance(genres, list):
                all_genres.update(genres)
        genres_list = sorted(list(all_genres))
    else:
        genres_list = ["Fiction", "Non-Fiction", "Mystery", "Science Fiction", "Fantasy", 
                      "Romance", "Thriller", "Horror", "Biography", "History"]
    
    # Get popular books
    popular_books = get_popular_books(books_df, n=12)
    
    data_dict = {
        'books_df': books_df,
        'interactions_df': interactions_df,
        'book_id_map_df': book_id_map_df,
        'user_id_map_df': user_id_map_df,
        'content_recommender': content_recommender,
        'collaborative_recommender': collaborative_recommender,
        'hybrid_recommender': hybrid_recommender,
        'genres_list': genres_list,
        'popular_books': popular_books
    }
    
    logging.info("Data loading complete.")
    return data_dict


# Home page
def home_page(data):
    """
    Render the home page.
    
    Args:
        data (Dict): Data dictionary
    """
    # Header
    render_header()
    
    # Get data
    books_df = data['books_df']
    popular_books = data['popular_books']
    hybrid_recommender = data['hybrid_recommender']
    
    # Popular Now section
    render_section_header("Popular Now", "See all →")
    
    # Convert popular books to list of dictionaries
    popular_books_list = popular_books.to_dict('records')
    render_book_row(popular_books_list)
    
    # Recommendations section
    if "user_id" in st.session_state and st.session_state.user_id:
        render_section_header("Your Recommendations", "Refresh →")
        
        # Get user profile
        interactions_df = data['interactions_df']
        user_id = st.session_state.user_id
        
        # Get user's reading history
        user_history = get_user_reading_history(user_id, interactions_df, books_df)
        
        # Create user profile
        user_profile = create_user_profile(user_history, books_df)
        
        # Get book IDs the user has already interacted with
        user_book_ids = set(user_history['book_id'].tolist())
        
        # Get recommendations
        recommendations = hybrid_recommender.recommend_for_user(
            user_id=user_id,
            user_profile=user_profile,
            n=8,
            method='weighted'
        )
        
        if not recommendations.empty:
            recommendations_list = recommendations.to_dict('records')
            render_book_row(recommendations_list)
        else:
            render_empty_state("We don't have enough data to make personalized recommendations yet. Start by rating some books!")
    else:
        # Show suggestion to sign in or create an account
        st.markdown(
            """
            <div style="background-color: #e9f7fe; padding: 1rem; border-radius: 0.5rem; margin: 2rem 0;">
                <h3 style="margin-top: 0;">Get Personalized Recommendations</h3>
                <p>Sign in or create an account to get personalized book recommendations.</p>
                <button class="primary-button">Sign In</button>
                <button style="margin-left: 0.5rem;" class="secondary-button">Create Account</button>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Show trending books instead
        render_section_header("Trending This Week", "See all →")
        
        # Get a different set of popular books for trending
        # (Here we're just using the same popular books but would be different in a real app)
        trending_books = popular_books.sample(min(8, len(popular_books))).to_dict('records')
        render_book_row(trending_books)


# Search page
def search_page(data):
    """
    Render the search page.
    
    Args:
        data (Dict): Data dictionary
    """
    st.markdown("<h1>Search & Discover</h1>", unsafe_allow_html=True)
    
    # Get data
    books_df = data['books_df']
    genres_list = data['genres_list']
    
    # Initialize session state for pagination
    if 'search_page' not in st.session_state:
        st.session_state.search_page = 1
    
    # Render search box
    search_params = render_search_box(genres_list)
    
    # Filter books if search button is clicked or there's a query in session state
    if search_params['search_clicked'] or 'search_query' in st.session_state:
        # Save search query to session state
        if search_params['query']:
            st.session_state.search_query = search_params['query']
        
        # Get search query from session state if available
        query = st.session_state.get('search_query', search_params['query'])
        
        # Filter books
        filtered_books = filter_books(
            books_df,
            query=query,
            genre=search_params['genre'],
            year_range=search_params['year_range'],
            min_rating=search_params['min_rating']
        )
        
        # Results header
        num_results = len(filtered_books)
        render_section_header(f"Search Results ({num_results} books)")
        
        # Show sort dropdown
        sort_by = st.selectbox(
            "Sort by", 
            ["Relevance", "Rating", "Title"],
            label_visibility="collapsed"
        )
        
        # Sort results
        if sort_by == "Rating" and 'average_rating' in filtered_books.columns:
            filtered_books = filtered_books.sort_values('average_rating', ascending=False)
        elif sort_by == "Title" and 'title' in filtered_books.columns:
            filtered_books = filtered_books.sort_values('title')
        
        # Paginate results
        per_page = 12
        total_pages = max(1, (num_results + per_page - 1) // per_page)
        
        # Display books
        if not filtered_books.empty:
            # Get books for current page
            page_books = paginate_books(filtered_books, st.session_state.search_page, per_page)
            
            # Convert to list of dictionaries
            books_list = page_books.to_dict('records')
            
            # Render books
            render_book_row(books_list)
            
            # Pagination
            st.session_state.search_page = render_pagination(
                st.session_state.search_page, 
                total_pages,
                key_prefix="search"
            )
        else:
            render_empty_state()
    else:
        # Show popular categories
        render_section_header("Explore by Category")
        
        # Display genre buttons
        cols = st.columns(4)
        for i, genre in enumerate(genres_list[:8]):
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; 
                               margin-bottom: 1rem; text-align: center; cursor: pointer;"
                         onclick="this.style.backgroundColor='#e9f7fe';">
                        <h3 style="margin: 0; font-size: 1.1rem;">{genre}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        # Show popular books
        render_section_header("Popular Books", "See all →")
        
        popular_books = data['popular_books'].to_dict('records')
        render_book_row(popular_books)


# Recommendations page
def recommendations_page(data):
    """
    Render the recommendations page.
    
    Args:
        data (Dict): Data dictionary
    """
    st.markdown("<h1>My Recommendations</h1>", unsafe_allow_html=True)
    
    # Get data
    books_df = data['books_df']
    interactions_df = data['interactions_df']
    content_recommender = data['content_recommender']
    collaborative_recommender = data['collaborative_recommender']
    hybrid_recommender = data['hybrid_recommender']
    
    # Recommendation methods
    recommendation_method = render_recommendation_methods()
    
    # Create a dummy user profile if no user is logged in
    # In a real app, you would use the logged-in user's profile
    user_id = st.session_state.get('user_id', 1)  # Default to user 1 if not set
    
    # Get user's reading history
    user_history = get_user_reading_history(user_id, interactions_df, books_df)
    
    # Create user profile
    user_profile = create_user_profile(user_history, books_df)
    
    # Extract top genres and authors from user profile
    if user_profile:
        top_genres = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[:3]
        top_genres = [genre.capitalize() for genre, _ in top_genres]
    else:
        top_genres = ["Fiction", "Mystery", "Thriller"]
    
    # Get favorite authors (from user history)
    if not user_history.empty and 'authors' in user_history.columns:
        author_counts = {}
        for _, row in user_history.iterrows():
            if 'authors' in row and row['authors']:
                if isinstance(row['authors'], list):
                    for author in row['authors']:
                        author_counts[author] = author_counts.get(author, 0) + 1
                else:
"""
UI components for the BookHaven recommendation system.
This module provides reusable UI components for the Streamlit interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import base64
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional


def render_header():
    """Render the header component."""
    st.markdown(
        """
        <div class="header">
            <h1>Welcome to BookHaven</h1>
            <p>Your personal reading companion</p>
            <p style="max-width: 600px; margin-bottom: 1.5rem;">
                Discover your next favorite book with our personalized recommendation system.
                Based on your reading history and preferences, we'll help you find books you'll love.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, action_text: str = None, action_link: str = "#"):
    """
    Render a section header.
    
    Args:
        title (str): Section title
        action_text (str, optional): Text for the action link
        action_link (str, optional): URL for the action link
    """
    if action_text:
        st.markdown(
            f"""
            <div class="section-header">
                <h2>{title}</h2>
                <a href="{action_link}">{action_text}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)


def get_placeholder_cover(book_id: int, title: str = "", author: str = "") -> str:
    """
    Generate a placeholder book cover.
    
    Args:
        book_id (int): Book ID for deterministic color generation
        title (str, optional): Book title
        author (str, optional): Book author
        
    Returns:
        str: Base64-encoded image data URL
    """
    # Use book_id as a seed for deterministic color generation
    random.seed(book_id)
    
    # Choose a color based on the book_id
    hue = random.randint(0, 360)
    saturation = random.randint(60, 100)
    lightness = random.randint(30, 70)
    
    # Convert HSL to RGB (simplified conversion)
    c = (1 - abs(2 * lightness / 100 - 1)) * saturation / 100
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = lightness / 100 - c / 2
    
    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    elif 240 <= hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    # Convert to 8-bit color values
    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)
    
    # Create a colored image
    img = Image.new("RGB", (150, 225), color=(r, g, b))
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


def render_stars(rating: float) -> str:
    """
    Render star ratings as HTML.
    
    Args:
        rating (float): Rating value (0-5)
        
    Returns:
        str: HTML for star rating
    """
    if pd.isna(rating) or rating is None:
        rating = 0.0
    
    # Ensure rating is within 0-5 range
    rating = max(0, min(5, float(rating)))
    
    # Calculate full, half, and empty stars
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    # Generate HTML
    stars_html = ""
    for _ in range(full_stars):
        stars_html += "★"
    if half_star:
        stars_html += "½"
    for _ in range(empty_stars):
        stars_html += "☆"
    
    return f"""<span class="rating">{stars_html} {rating:.1f}</span>"""


def render_book_card(book: Dict[str, Any]) -> str:
    """
    Render a book card as HTML.
    
    Args:
        book (Dict[str, Any]): Book data
        
    Returns:
        str: HTML for book card
    """
    # Get book details
    book_id = book.get('book_id', 0)
    title = book.get('title', 'Unknown Title')
    author = book.get('authors', 'Unknown Author')
    rating = book.get('average_rating', 0.0)
    
    # Handle different author formats
    if isinstance(author, list):
        author = ", ".join(author)
    
    # Get genre (first one if list)
    genre = "Unknown"
    if 'genres' in book and book['genres']:
        if isinstance(book['genres'], list) and len(book['genres']) > 0:
            genre = book['genres'][0].capitalize()
        else:
            genre = str(book['genres']).capitalize()
    
    # Get book cover image
    if 'image_url' in book and book['image_url']:
        cover_url = book['image_url']
    else:
        cover_url = get_placeholder_cover(book_id, title, author)
    
    # Generate HTML for book card
    return f"""
    <div class="book-card">
        <img src="{cover_url}" alt="{title}">
        <div class="book-card-content">
            <h3>{title}</h3>
            <p>by {author}</p>
            {render_stars(rating)}
            <span class="book-genre">{genre}</span>
            <div class="book-card-footer">
                <a href="#" onclick="window.book_detail_{book_id}()">Details →</a>
                <a href="#" onclick="window.save_book_{book_id}()">Save</a>
            </div>
        </div>
    </div>
    """


def render_book_row(books: List[Dict[str, Any]], cols: int = 4) -> None:
    """
    Render a row of book cards.
    
    Args:
        books (List[Dict]): List of book data
        cols (int, optional): Number of columns
    """
    # Check if we have books to display
    if not books:
        st.info("No books to display.")
        return
    
    # Process books in chunks of cols
    for i in range(0, len(books), cols):
        # Get current chunk of books
        chunk_books = books[i:i+cols]
        
        # Generate HTML for each book card
        cols_html = ""
        for book in chunk_books:
            cols_html += f"""
            <div class="column" style="width: {100 / cols}%; padding: 0 0.5rem;">
                {render_book_card(book)}
            </div>
            """
        
        # Render the row
        st.markdown(
            f"""
            <div style="display: flex; margin: 0 -0.5rem;">
                {cols_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_profile_section(
    top_genres: List[str] = None,
    favorite_authors: List[str] = None
) -> None:
    """
    Render the user profile section.
    
    Args:
        top_genres (List[str], optional): User's top genres
        favorite_authors (List[str], optional): User's favorite authors
    """
    if top_genres is None:
        top_genres = ["Fiction", "Mystery", "Thriller"]
    
    if favorite_authors is None:
        favorite_authors = ["Stephen King", "Agatha Christie"]
    
    # Generate HTML for genre tags
    genre_tags = ""
    for genre in top_genres:
        genre_tags += f'<span class="tag">{genre}</span> '
    
    # Generate HTML for author tags
    author_tags = ""
    for author in favorite_authors:
        author_tags += f'<span class="tag author-tag">{author}</span> '
    
    # Render the profile section
    st.markdown(
        f"""
        <div class="profile-section">
            <h3>Your reading profile</h3>
            <div style="display: flex; margin-top: 0.8rem;">
                <div style="flex: 1;">
                    <h4 style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Top Genres</h4>
                    <div>
                        {genre_tags}
                    </div>
                </div>
                <div style="flex: 1;">
                    <h4 style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Favorite Authors</h4>
                    <div>
                        {author_tags}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search_box(genres: List[str], min_year: int = 1900, max_year: int = 2025) -> Dict[str, Any]:
    """
    Render the search and filter box.
    
    Args:
        genres (List[str]): Available genres
        min_year (int, optional): Minimum publication year
        max_year (int, optional): Maximum publication year
        
    Returns:
        Dict[str, Any]: Dictionary of search parameters
    """
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    
    # Search input
    search_query = st.text_input("", placeholder="Search by title, author, or ISBN...")
    
    # Filters in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        genre = st.selectbox("Genre", ["All Genres"] + genres)
    
    with col2:
        year_range = st.select_slider(
            "Publication Year", 
            options=list(range(min_year, max_year + 1, 5)),
            value=(min_year, max_year)
        )
    
    with col3:
        min_rating = st.select_slider(
            "Minimum Rating",
            options=[0.0, 3.0, 4.0, 4.5],
            value=0.0,
            format_func=lambda x: f"{x}+ Stars" if x > 0 else "Any Rating",
        )
    
    # Search button
    search_button = st.button("Search Books", type="primary")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Return search parameters
    return {
        "query": search_query,
        "genre": genre,
        "year_range": year_range,
        "min_rating": min_rating,
        "search_clicked": search_button
    }


def render_recommendation_methods() -> str:
    """
    Render recommendation method selection.
    
    Returns:
        str: Selected recommendation method
    """
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    
    st.markdown("<h2>How would you like recommendations?</h2>", unsafe_allow_html=True)
    
    recommendation_method = st.radio(
        "Select a recommendation method",
        ["Based on my history", "Similar to a book", "Community favorites"],
        label_visibility="collapsed",
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return recommendation_method


def render_book_detail_modal(book: Dict[str, Any]) -> None:
    """
    Render a modal with book details.
    
    Args:
        book (Dict[str, Any]): Book data
    """
    # Get book details
    title = book.get('title', 'Unknown Title')
    author = book.get('authors', 'Unknown Author')
    rating = book.get('average_rating', 0.0)
    description = book.get('description', 'No description available.')
    
    # Handle different author formats
    if isinstance(author, list):
        author = ", ".join(author)
    
    # Get genres as string
    genres_str = ""
    if 'genres' in book and book['genres']:
        if isinstance(book['genres'], list):
            genres_str = ", ".join([g.capitalize() for g in book['genres']])
        else:
            genres_str = str(book['genres']).capitalize()
    
    # Book cover
    if 'image_url' in book and book['image_url']:
        cover_url = book['image_url']
    else:
        cover_url = get_placeholder_cover(book.get('book_id', 0), title, author)
    
    # Render modal content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(cover_url, width=200)
    
    with col2:
        st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3>by {author}</h3>", unsafe_allow_html=True)
        st.markdown(render_stars(rating), unsafe_allow_html=True)
        
        if genres_str:
            st.markdown(f"<p><strong>Genres:</strong> {genres_str}</p>", unsafe_allow_html=True)
        
        st.markdown("<h3>Description</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{description}</p>", unsafe_allow_html=True)
        
        # Actions
        col_a, col_b = st.columns(2)
        with col_a:
            st.button("Add to Reading List", key=f"add_{book.get('book_id', 0)}")
        with col_b:
            st.button("Mark as Read", key=f"read_{book.get('book_id', 0)}")


def render_empty_state(message: str = "No books found matching your criteria.") -> None:
    """
    Render an empty state when no books are available.
    
    Args:
        message (str): Message to display
    """
    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; 
                    border-radius: 0.5rem; margin: 1rem 0;">
            <img src="https://via.placeholder.com/150x150?text=📚" alt="Empty state" style="opacity: 0.5;">
            <h3 style="margin-top: 1rem; color: #6c757d;">{message}</h3>
            <p style="color: #6c757d; max-width: 500px; margin: 0 auto;">
                Try adjusting your filters or exploring our popular books instead.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pagination(page: int, total_pages: int, key_prefix: str = "pagination") -> int:
    """
    Render pagination controls.
    
    Args:
        page (int): Current page number (1-based)
        total_pages (int): Total number of pages
        key_prefix (str): Prefix for Streamlit component keys
        
    Returns:
        int: Updated page number based on user interaction
    """
    if total_pages <= 1:
        return 1
    
    # Container for pagination
    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-top: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
    
    with col1:
        if st.button("<<", key=f"{key_prefix}_first", disabled=(page == 1)):
            page = 1
    
    with col2:
        if st.button("<", key=f"{key_prefix}_prev", disabled=(page == 1)):
            page = max(1, page - 1)
    
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <span style="font-size: 0.9rem; color: #666;">
                    Page {page} of {total_pages}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col4:
        if st.button(">", key=f"{key_prefix}_next", disabled=(page == total_pages)):
            page = min(total_pages, page + 1)
    
    with col5:
        if st.button(">>", key=f"{key_prefix}_last", disabled=(page == total_pages)):
            page = total_pages
    
    st.markdown(
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    return page


def render_loading() -> None:
    """Render a loading indicator."""
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; padding: 2rem;">
            <div class="lds-dual-ring"></div>
        </div>
        <style>
        .lds-dual-ring {
            display: inline-block;
            width: 80px;
            height: 80px;
        }
        .lds-dual-ring:after {
            content: " ";
            display: block;
            width: 64px;
            height: 64px;
            margin: 8px;
            border-radius: 50%;
            border: 6px solid #1A365D;
            border-color: #1A365D transparent #1A365D transparent;
            animation: lds-dual-ring 1.2s linear infinite;
        }
        @keyframes lds-dual-ring {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(360deg);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def filter_books(
    books_df: pd.DataFrame,
    query: str = "",
    genre: str = "All Genres",
    year_range: tuple = (1900, 2025),
    min_rating: float = 0.0
) -> pd.DataFrame:
    """
    Filter books based on search criteria.
    
    Args:
        books_df (pd.DataFrame): Book data
        query (str): Search query
        genre (str): Genre filter
        year_range (tuple): Publication year range (min, max)
        min_rating (float): Minimum rating
        
    Returns:
        pd.DataFrame: Filtered books
    """
    # Make a copy of the dataframe
    filtered_df = books_df.copy()
    
    # Apply text search
    if query:
        query = query.lower()
        
        # Search in title
        title_match = filtered_df['title'].str.lower().str.contains(query, na=False)
        
        # Search in authors
        if 'authors' in filtered_df.columns:
            if filtered_df['authors'].apply(lambda x: isinstance(x, list)).any():
                # Handle list of authors
                author_match = filtered_df['authors'].apply(
                    lambda authors: any(query in author.lower() for author in authors) 
                    if isinstance(authors, list) else False
                )
            else:
                # Handle string authors
                author_match = filtered_df['authors'].str.lower().str.contains(query, na=False)
        else:
            author_match = pd.Series(False, index=filtered_df.index)
        
        # Search in ISBN if available
        if 'isbn' in filtered_df.columns:
            isbn_match = filtered_df['isbn'].str.contains(query, na=False)
        elif 'isbn13' in filtered_df.columns:
            isbn_match = filtered_df['isbn13'].str.contains(query, na=False)
        else:
            isbn_match = pd.Series(False, index=filtered_df.index)
        
        # Apply text search filter
        filtered_df = filtered_df[title_match | author_match | isbn_match]
    
    # Apply genre filter
    if genre != "All Genres" and 'genres' in filtered_df.columns:
        if filtered_df['genres'].apply(lambda x: isinstance(x, list)).any():
            # Handle list of genres
            genre_match = filtered_df['genres'].apply(
                lambda genres: genre.lower() in [g.lower() for g in genres]
                if isinstance(genres, list) else False
            )
        else:
            # Handle string genres
            genre_match = filtered_df['genres'].str.lower() == genre.lower()
        
        filtered_df = filtered_df[genre_match]
    
    # Apply year filter if publication_year exists
    if 'publication_year' in filtered_df.columns:
        year_match = (
            (filtered_df['publication_year'] >= year_range[0]) &
            (filtered_df['publication_year'] <= year_range[1])
        )
        filtered_df = filtered_df[year_match]
    
    # Apply rating filter
    if 'average_rating' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['average_rating'] >= min_rating]
    
    return filtered_df


def paginate_books(books_df: pd.DataFrame, page: int, per_page: int = 12) -> pd.DataFrame:
    """
    Paginate books.
    
    Args:
        books_df (pd.DataFrame): Book data
        page (int): Page number (1-based)
        per_page (int): Number of books per page
        
    Returns:
        pd.DataFrame: Books for the current page
    """
    # Ensure page is valid
    if page < 1:
        page = 1
    
    # Calculate start and end indices
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get books for the current page
    if start_idx < len(books_df):
        return books_df.iloc[start_idx:min(end_idx, len(books_df))]
    else:
        return pd.DataFrame(columns=books_df.columns)


def reset_session_state(*keys: str) -> None:
    """
    Reset specified session state keys.
    
    Args:
        *keys: Variable number of keys to reset
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
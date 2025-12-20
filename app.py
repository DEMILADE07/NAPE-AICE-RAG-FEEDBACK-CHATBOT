"""Streamlit web interface for NAPE RAG Chatbot"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from storage import StorageManager
from rag_engine import RAGEngine
import pandas as pd
from datetime import datetime
import time
import os
from PIL import Image
from io import BytesIO

# Page config
st.set_page_config(
    page_title="NAPE 43rd AICE Feedback Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'storage' not in st.session_state:
    st.session_state.storage = StorageManager()
    st.session_state.rag_engine = RAGEngine(st.session_state.storage)

# Enhanced Custom CSS with animations
st.markdown("""
    <style>
    /* Import Google Fonts for professional look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Image Container */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    .header-image {
        max-width: 100%;
        height: auto;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #006400 0%, #228B22 50%, #32CD32 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1rem 0;
        padding: 1rem 0;
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Sidebar Styling - Oil & Gas Theme */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 2px solid #006400;
    }
    
    /* Sidebar sections */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Sidebar headings */
    .sidebar .element-container h3 {
        color: #006400;
        font-weight: 700;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    /* Button Styling - Professional and accessible */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #006400 0%, #228B22 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,100,0,0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #228B22 0%, #006400 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,100,0,0.3);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Example Query Buttons - Fixed equal sizes - More aggressive */
    div[data-testid="column"] {
        width: 20% !important;
        flex: 0 0 20% !important;
    }
    
    button[key*="example"] {
        width: 100% !important;
        height: 100px !important;
        min-height: 100px !important;
        max-height: 100px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        padding: 0.75rem 0.5rem !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
        box-sizing: border-box !important;
    }
    
    /* Metric Cards - Larger for readability, prevent truncation */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #006400;
        white-space: nowrap;
        overflow: visible;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Sidebar metric containers - prevent truncation - More aggressive */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        word-break: keep-all !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        max-width: none !important;
        width: auto !important;
    }
    
    /* Sidebar metric container */
    section[data-testid="stSidebar"] [data-testid="stMetricContainer"] {
        overflow: visible !important;
        width: 100% !important;
    }
    
    /* Ensure sidebar has enough width */
    section[data-testid="stSidebar"] {
        min-width: 280px !important;
    }
    
    /* Column containers in sidebar */
    section[data-testid="stSidebar"] div[data-testid="column"] {
        min-width: 0 !important;
        flex: 1 1 0% !important;
    }
    
    /* Info Boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #006400;
        font-size: 1rem;
    }
    
    /* Answer Box - Professional gradient */
    .answer-box {
        background: linear-gradient(135deg, #006400 0%, #228B22 50%, #32CD32 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,100,0,0.2);
        margin: 1.5rem 0;
        font-size: 1.1rem;
        line-height: 1.8;
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
        font-size: 0.95rem;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #006400 !important;
    }
    
    /* Card-like containers */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 1rem 1.5rem;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        animation: slideIn 0.6s ease-out;
    }
    
    /* Force plotly charts to use full width */
    .js-plotly-plot {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Ensure columns use full available width */
    div[data-testid="column"] {
        min-width: 0 !important;
        flex: 1 1 0% !important;
    }
    
    /* Plotly chart container - force full width */
    .plotly {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Plotly graph div */
    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
    }
    
    div[data-testid="stPlotlyChart"] > div {
        width: 100% !important;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header Image - Multiple approaches to ensure it displays (70% size)
header_displayed = False

# Method 1: Direct file path with size constraint
try:
    header_path = "header image.jpg"
    if os.path.exists(header_path):
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        # Use width parameter to control size (70% of container)
        st.image(header_path, use_container_width=False, width=700)
        st.markdown('</div>', unsafe_allow_html=True)
        header_displayed = True
except Exception:
    pass

# Method 2: PIL Image if direct path fails
if not header_displayed:
    try:
        header_img = Image.open("header image.jpg")
        # Resize to 70% of original
        original_width, original_height = header_img.size
        new_width = int(original_width * 0.7)
        new_height = int(original_height * 0.7)
        header_img_resized = header_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        st.image(header_img_resized, use_container_width=False, width=new_width)
        st.markdown('</div>', unsafe_allow_html=True)
        header_displayed = True
    except Exception:
        pass

# Method 3: Base64 encoding as last resort
if not header_displayed:
    try:
        import base64
        header_img = Image.open("header image.jpg")
        # Resize to 70%
        original_width, original_height = header_img.size
        new_width = int(original_width * 0.7)
        new_height = int(original_height * 0.7)
        header_img_resized = header_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64
        from io import BytesIO
        buffer = BytesIO()
        header_img_resized.save(buffer, format='JPEG')
        img_data = base64.b64encode(buffer.getvalue()).decode()
        
        st.markdown(
            f'<div class="header-container" style="text-align: center;">'
            f'<img src="data:image/jpeg;base64,{img_data}" style="width: 70%; max-width: 1000px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.15);" />'
            f'</div>',
            unsafe_allow_html=True
        )
        header_displayed = True
    except Exception:
        pass

if not header_displayed:
    st.markdown('<h1 class="main-header">📊 NAPE 43rd AICE Feedback Analysis</h1>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Oil & Gas themed
with st.sidebar:
    # Oil & Gas themed header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 3px solid #006400; margin-bottom: 1.5rem;">
        <h2 style="color: #006400; margin: 0; font-size: 1.5rem; font-weight: 700;">🛢️ NAPE 43rd AICE</h2>
        <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Feedback Analysis System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats with better styling - Oil & Gas theme (removed green bar/div)
    st.markdown("#### 📈 Quick Statistics")
    events = st.session_state.storage.get_event_list()
    total_responses = sum(e['response_count'] for e in events)
    
    # Use markdown instead of metrics to prevent truncation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📅 Events**")
        st.markdown(f'<div style="font-size: 2rem; font-weight: 700; color: #006400;">{len(events)}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**💬 Responses**")
        st.markdown(f'<div style="font-size: 2rem; font-weight: 700; color: #006400;">{total_responses:,}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Event filter - Functional
    st.markdown("#### 🔍 Filter by Event")
    event_names = ['All'] + [e['event_name'] for e in events]
    selected_event = st.selectbox(
        "Select Event", 
        event_names, 
        label_visibility="visible",
        help="Filter analytics dashboard by specific event"
    )
    
    st.markdown("---")
    
    # Additional info section
    st.markdown("#### ℹ️ About")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; font-size: 0.85rem; color: #666;">
        <p><strong>System:</strong> AI-Powered Feedback Analysis</p>
        <p><strong>Technology:</strong> RAG (Retrieval-Augmented Generation)</p>
        <p><strong>Purpose:</strong> Conference Feedback Insights</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["💬 Query Interface", "📊 Analytics Dashboard", "📋 Event List"])

# Tab 1: Query Interface
with tab1:
    st.markdown("### 🎯 Ask Questions About Conference Feedback")
    st.markdown("Ask natural language questions about attendee feedback. Examples:")
    
    # Example queries - FIXED: Use session state properly
    example_queries = [
        "What did attendees say about WiGE?",
        "What was the average rating for accommodation?",
        "What were common complaints about transportation?",
        "How many hotels were used in total?",
        "What did people say about Posh Hotel?"
    ]
    
    # Initialize query in session state if not exists
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ''
    
    # Display example buttons - Equal sized cards
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            # Truncate to fit in button while keeping same size
            display_text = example[:25] + "..." if len(example) > 25 else example
            if st.button(f"💡 {display_text}", key=f"example_{i}", use_container_width=True):
                st.session_state.current_query = example
                st.rerun()  # Rerun to update the input field
    
    # Query input - use session state
    query = st.text_input(
        "Enter your question:", 
        value=st.session_state.current_query,
        placeholder="e.g., What did attendees think about the technical sessions?",
        label_visibility="visible",
        key="query_input"
    )
    
    # Update session state when user types
    if query != st.session_state.current_query:
        st.session_state.current_query = query
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    if ask_button and query:
        # Clear the query from session state after processing
        query_to_process = st.session_state.current_query
        st.session_state.current_query = ''  # Clear after use
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        try:
            status_text.text("🔍 Analyzing your query...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            status_text.text("📊 Retrieving relevant feedback...")
            progress_bar.progress(40)
            time.sleep(0.3)
            
            status_text.text("🤖 Generating response...")
            progress_bar.progress(60)
            time.sleep(0.3)
            
            result = st.session_state.rag_engine.query(query_to_process)
            
            progress_bar.progress(100)
            elapsed_time = time.time() - start_time
            
            status_text.empty()
            progress_bar.empty()
            
            # Display answer with better styling
            st.markdown("### 💡 Answer")
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True
            )
            
            # Show response time
            st.caption(f"⏱️ Response generated in {elapsed_time:.2f} seconds")
            
            # Show sources if available
            if result.get('sources'):
                num_sources = len(result['sources'])
                with st.expander(f"📎 Source Feedback (Top {num_sources})", expanded=False):
                    for i, source in enumerate(result['sources'], 1):
                        event_name = source['metadata'].get('event_name', 'Unknown')
                        st.markdown(f"**{i}. [{event_name}]**")
                        st.markdown(f"   *{source['comment']}*")
                        if i < len(result['sources']):
                            st.markdown("---")
            
            # Show statistics if available with better formatting
            if result.get('stats') and result['stats'].get('ratings'):
                st.markdown("### 📊 Related Statistics")
                stats_df = pd.DataFrame(result['stats']['ratings'])
                
                # Format the dataframe
                display_df = stats_df[['event_name', 'rating_question', 'average_rating', 'total_count']].copy()
                display_df['average_rating'] = display_df['average_rating'].round(2)
                display_df['rating_question'] = display_df['rating_question'].apply(
                    lambda x: x[:60] + "..." if len(str(x)) > 60 else x
                )
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Quick visualization
                if len(display_df) > 0:
                    st.markdown("#### 📈 Rating Distribution")
                    fig = px.bar(
                        display_df.head(10),
                        x='average_rating',
                        y='rating_question',
                        orientation='h',
                        color='average_rating',
                        color_continuous_scale='RdYlGn',
                        labels={'average_rating': 'Average Rating', 'rating_question': 'Question'},
                        height=400
                    )
                    fig.update_layout(showlegend=False, font=dict(size=14))
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error processing query: {str(e)}")

# Tab 2: Analytics Dashboard - ENHANCED
with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    
    events = st.session_state.storage.get_event_list()
    
    if not events:
        st.warning("⚠️ No data available. Please refresh data from Google Sheets first.")
    else:
        # Overall statistics with better cards
        st.markdown("#### 📈 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_events = len(events)
        total_responses = sum(e['response_count'] for e in events)
        categories = len(set(e['event_category'] for e in events))
        
        with col1:
            st.metric("📅 Total Events", total_events, delta=None)
        with col2:
            st.metric("💬 Total Responses", f"{total_responses:,}", delta=None)
        with col3:
            st.metric("🏷️ Categories", categories, delta=None)
        with col4:
            avg_responses = total_responses / total_events if total_events > 0 else 0
            st.metric("📊 Avg/Event", f"{avg_responses:.1f}", delta=None)
        
        st.markdown("---")
        
        # NEW: Pie Chart - Responses by Category
        st.markdown("#### 📊 Response Distribution by Category")
        events_df = pd.DataFrame(events)
        category_counts = events_df.groupby('event_category')['response_count'].sum().reset_index()
        category_counts = category_counts.sort_values('response_count', ascending=False)
        
        # Use equal columns - ensure they use full width
        col1, col2 = st.columns(2, gap="small")
        
        with col1:
            fig_pie = px.pie(
                category_counts,
                values='response_count',
                names='event_category',
                title="Responses by Category",
                color_discrete_sequence=px.colors.sequential.Greens
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=16)
            fig_pie.update_layout(
                font=dict(size=16), 
                height=600,
                width=None,  # Let it use full container width
                margin=dict(l=5, r=5, t=80, b=5),
                autosize=True,
                showlegend=True,
                legend=dict(font=dict(size=14), x=1.02, y=0.5)
            )
            # Force chart to use full column width
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
        
        with col2:
            # Bar chart for category comparison
            fig_cat_bar = px.bar(
                category_counts,
                x='event_category',
                y='response_count',
                title="Total Responses by Category",
                labels={'response_count': 'Number of Responses', 'event_category': 'Category'},
                color='response_count',
                color_continuous_scale='Greens'
            )
            fig_cat_bar.update_layout(
                font=dict(size=16), 
                height=600,
                width=None,  # Let it use full container width
                showlegend=False,
                margin=dict(l=40, r=10, t=80, b=40),
                autosize=True
            )
            fig_cat_bar.update_xaxes(tickangle=-45, tickfont=dict(size=14), title_font=dict(size=16))
            fig_cat_bar.update_yaxes(tickfont=dict(size=14), title_font=dict(size=16))
            # Force chart to use full column width
            st.plotly_chart(fig_cat_bar, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
        
        st.markdown("---")
        
        # Event response counts - Top 20
        st.markdown("#### 📊 Top Events by Response Count")
        events_df_filtered = events_df.copy()
        
        if selected_event and selected_event != 'All':
            events_df_filtered = events_df_filtered[events_df_filtered['event_name'] == selected_event]
        
        top_events = events_df_filtered.sort_values('response_count', ascending=True).tail(20)
        
        fig = px.bar(
            top_events,
            x='response_count',
            y='event_name',
            orientation='h',
            title="Top 20 Events by Response Count",
            labels={'response_count': 'Number of Responses', 'event_name': 'Event'},
            color='response_count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=600, showlegend=False, font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Ratings visualization - Enhanced
        st.markdown("#### ⭐ Rating Analysis")
        stats = st.session_state.storage.get_structured_stats()
        
        if stats.get('ratings'):
            ratings_df = pd.DataFrame(stats['ratings'])
            
            # Group by event and show average
            event_ratings = ratings_df.groupby('event_name')['average_rating'].mean().reset_index()
            event_ratings = event_ratings.sort_values('average_rating', ascending=False).head(15)
            
            # Use equal columns - ensure they use full width
            col1, col2 = st.columns(2, gap="small")
            
            with col1:
                # Bar chart for top rated events
                fig2 = px.bar(
                    event_ratings,
                    x='average_rating',
                    y='event_name',
                    orientation='h',
                    title="Top 15 Events by Average Rating",
                    labels={'average_rating': 'Average Rating (out of 5)', 'event_name': 'Event'},
                    color='average_rating',
                    color_continuous_scale='RdYlGn'
                )
                fig2.update_layout(
                    height=600,
                    width=None,  # Let it use full container width
                    showlegend=False, 
                    font=dict(size=16), 
                    margin=dict(l=90, r=10, t=80, b=40),
                    autosize=True
                )
                fig2.update_xaxes(tickfont=dict(size=14), title_font=dict(size=16))
                fig2.update_yaxes(tickfont=dict(size=12), title_font=dict(size=16))
                # Force chart to use full column width
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
            
            with col2:
                # Rating distribution histogram
                fig_hist = px.histogram(
                    ratings_df,
                    x='average_rating',
                    nbins=20,
                    title="Distribution of Ratings",
                    labels={'average_rating': 'Average Rating', 'count': 'Number of Questions'},
                    color_discrete_sequence=['#228B22']
                )
                fig_hist.update_layout(
                    height=600,
                    width=None,  # Let it use full container width
                    font=dict(size=16), 
                    showlegend=False, 
                    margin=dict(l=40, r=10, t=80, b=40),
                    autosize=True
                )
                fig_hist.update_xaxes(tickfont=dict(size=14), title_font=dict(size=16))
                fig_hist.update_yaxes(tickfont=dict(size=14), title_font=dict(size=16))
                # Force chart to use full column width
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
            
            st.markdown("---")
            
            # NEW: Ratings by Category
            st.markdown("#### 📈 Average Ratings by Category")
            ratings_by_category = ratings_df.groupby('event_category')['average_rating'].mean().reset_index()
            ratings_by_category = ratings_by_category.sort_values('average_rating', ascending=False)
            
            fig_cat_ratings = px.bar(
                ratings_by_category,
                x='event_category',
                y='average_rating',
                title="Average Rating by Event Category",
                labels={'average_rating': 'Average Rating (out of 5)', 'event_category': 'Category'},
                color='average_rating',
                color_continuous_scale='RdYlGn'
            )
            fig_cat_ratings.update_layout(height=400, showlegend=False, font=dict(size=14))
            fig_cat_ratings.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_cat_ratings, use_container_width=True)
        else:
            st.info("ℹ️ No rating data available yet.")

# Tab 3: Event List
with tab3:
    st.markdown("### 📋 All Events")
    
    events = st.session_state.storage.get_event_list()
    
    if events:
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values('event_name')
        
        # Add category filter
        categories = ['All'] + sorted(events_df['event_category'].unique().tolist())
        selected_category = st.selectbox("Filter by Category", categories, key="category_filter")
        
        if selected_category != 'All':
            events_df = events_df[events_df['event_category'] == selected_category]
        
        st.dataframe(
            events_df[['event_name', 'event_category', 'response_count']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("⚠️ No events found. Please refresh data from Google Sheets.")

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '<p><strong>NAPE 43rd AICE Monitoring & Evaluation Committee</strong></p>'
    '<p>Built with ❤️ using Streamlit & RAG</p>'
    '</div>',
    unsafe_allow_html=True
)

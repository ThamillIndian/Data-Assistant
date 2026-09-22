import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from utils.data_loader import load_csv_data, validate_sales_schema, get_data_info
from utils.summarizer import compute_kpis, create_sales_trend_chart, create_family_distribution_chart, create_store_performance_chart, generate_executive_summary, format_kpis_display
from utils.rag_agent import RetailRAGAgent

# Load environment variables from .env file
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Retail Insights Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'kpis' not in st.session_state:
    st.session_state.kpis = None
if 'executive_summary' not in st.session_state:
    st.session_state.executive_summary = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main header
st.markdown('<h1 class="main-header">📊 Retail Insights Assistant</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load API key from environment variable
    env_api_key = os.getenv("GEMINI_API_KEY")
    
    # Allow manual override
    api_key = st.text_input(
        "Google API Key", 
        type="password", 
        placeholder="Enter your Gemini API key (loaded from .env if available)",
        value=env_api_key if env_api_key else ""
    )
    
    if api_key:
        st.session_state.api_key = api_key
        if env_api_key and api_key == env_api_key:
            st.success("API Key loaded from .env file!")
        else:
            st.success("API Key set!")
    else:
        st.warning("Please enter your Google API Key to use AI features")
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload your sales CSV file
    2. Click 'Make Summary' to generate insights
    3. Ask questions in the chat panel
    4. Explore visualizations
    """)

# Main 3-column layout
col1, col2, col3 = st.columns([1, 2, 1.5])

# LEFT COLUMN: File Management & Controls
with col1:
    st.header("📁 File Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Sales Data (CSV)",
        type=['csv'],
        help="Upload a CSV file with columns: date, store, family, unit_sales, transactions"
    )
    
    if uploaded_file:
        # Load data
        df = load_csv_data(uploaded_file)
        
        if df is not None:
            # Validate schema
            is_valid, error_msg = validate_sales_schema(df)
            
            if is_valid:
                st.session_state.df = df
                st.success(f"✅ Data loaded successfully! {len(df)} rows")
                
                # Display data info
                info = get_data_info(df)
                st.info(f"""
                **Dataset Info:**
                - Rows: {info['rows']:,}
                - Columns: {info['columns']}
                - Memory: {info['memory_usage']:.2f} MB
                - Unique Stores: {info['unique_stores']}
                - Unique Families: {info['unique_families']}
                """)
                
                # Data preview toggle
                show_preview = st.checkbox("Show Data Preview", value=False)
                if show_preview:
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Make Summary button
                st.markdown("---")
                st.markdown("### 🚀 Generate Insights")
                if st.button("Make Summary", type="primary", use_container_width=True):
                    if 'api_key' in st.session_state:
                        print(f"DEBUG: API key present: {st.session_state.api_key[:10]}...")
                        with st.spinner("Generating insights..."):
                            # Compute KPIs
                            print("DEBUG: Computing KPIs...")
                            st.session_state.kpis = compute_kpis(df)
                            print(f"DEBUG: KPIs computed: {list(st.session_state.kpis.keys())}")
                            
                            # Generate executive summary
                            print("DEBUG: Generating executive summary...")
                            st.session_state.executive_summary = generate_executive_summary(
                                st.session_state.kpis, 
                                st.session_state.api_key
                            )
                            print(f"DEBUG: Executive summary generated: {len(st.session_state.executive_summary) if st.session_state.executive_summary else 0} chars")
                            
                            # Initialize agent
                            print("DEBUG: Initializing agent...")
                            st.session_state.agent = RetailRAGAgent(
                                st.session_state.api_key, 
                                df
                            )
                            print("DEBUG: Agent initialized")
                            
                            st.success("Summary generated successfully!")
                            st.rerun()
                    else:
                        st.error("Please enter your API key first!")
            else:
                st.error(f"❌ Invalid data schema: {error_msg}")
    
    # Clear data button
    if st.session_state.df is not None:
        if st.button("Clear Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.kpis = None
            st.session_state.executive_summary = None
            st.session_state.agent = None
            st.session_state.chat_history = []
            st.rerun()

# MIDDLE COLUMN: Summary Section
with col2:
    st.header("📈 Executive Summary")
    
    if st.session_state.kpis is None:
        st.info("👆 Upload data and click 'Make Summary' to generate insights")
    else:
        # Display KPIs
        st.markdown("### Key Performance Indicators")
        kpis_display = format_kpis_display(st.session_state.kpis)
        st.markdown(kpis_display)
        
        st.markdown("---")
        
        # Display executive summary
        st.markdown("### 📝 AI-Generated Executive Summary")
        if st.session_state.executive_summary:
            st.markdown(st.session_state.executive_summary)
        
        st.markdown("---")
        
        # Display charts
        st.markdown("### 📊 Visualizations")
        
        # Sales trend chart
        sales_chart = create_sales_trend_chart(st.session_state.df)
        if sales_chart:
            st.plotly_chart(sales_chart, use_container_width=True)
        
        # Family distribution chart
        family_chart = create_family_distribution_chart(st.session_state.df)
        if family_chart:
            st.plotly_chart(family_chart, use_container_width=True)
        
        # Store performance chart
        store_chart = create_store_performance_chart(st.session_state.df)
        if store_chart:
            st.plotly_chart(store_chart, use_container_width=True)

# RIGHT COLUMN: Chat Section
with col3:
    st.header("💬 Ask Questions")
    
    print(f"DEBUG: Chat section - Agent is None: {st.session_state.agent is None}")
    
    if st.session_state.agent is None:
        st.info("👆 Upload data and generate summary to start chatting")
    else:
        print("DEBUG: Agent exists, showing chat interface")
        # Chat interface
        st.markdown("### Conversation")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)
        
        # Chat input
        st.markdown("---")
        print("DEBUG: About to create chat input")
        user_input = st.chat_input("Ask a question about your data...")
        print(f"DEBUG: Chat input value: {user_input}")
        
        if user_input:
            print(f"DEBUG: User input received: {user_input}")
            # Display user message
            st.chat_message("user").write(user_input)
            
            # Get agent response
            with st.spinner("Thinking..."):
                print("DEBUG: Calling agent.query...")
                response = st.session_state.agent.query(user_input)
                print(f"DEBUG: Agent response: {response[:100] if response else 'None'}...")
            
            # Display assistant response
            st.chat_message("assistant").write(response)
            
            # Update chat history in session state
            st.session_state.chat_history = st.session_state.agent.get_conversation_history()
            
            # Rerun to update the display
            st.rerun()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.agent.clear_history()
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Retail Insights Assistant | Powered by Streamlit, LangChain & Gemini</p>
</div>
""", unsafe_allow_html=True)

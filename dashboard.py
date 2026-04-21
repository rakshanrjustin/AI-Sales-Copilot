import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# DEMO MODE FLAG - Set to True for Streamlit Cloud deployment
DEMO = True

# Set page configuration
st.set_page_config(
    page_title="AI Marketing Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    try:
        if DEMO:
            # Demo mode: Load small sample dataset
            print("Loading demo data...")
            try:
                df = pd.read_parquet("sample_data.parquet")
                # For demo mode, use same data for both main and priority datasets
                # Sample a smaller subset for priority users to simulate prioritization
                priority_df = df.sample(n=min(1000, len(df)), random_state=42).copy()
                
                # Create priority column based on abandonment scores for demo mode
                def assign_priority(score):
                    if score > 0.8:
                        return "CRITICAL"
                    elif score > 0.6:
                        return "HIGH"
                    elif score > 0.4:
                        return "MEDIUM"
                    else:
                        return "LOW"
                
                priority_df['priority'] = priority_df['abandonment_score'].apply(assign_priority)
                print(f"Demo data loaded successfully! Main: {len(df)}, Priority: {len(priority_df)}")
                return df, priority_df
            except FileNotFoundError:
                st.error("Demo mode: sample_data.parquet not found. Please run create_sample_data.py first.")
                return pd.DataFrame(), pd.DataFrame()
        else:
            # Production mode: Load full datasets
            print("Loading production data...")
            df = pd.read_parquet("final_marketing_output.parquet")
            priority_df = pd.read_parquet("priority_users.parquet")
            print(f"Production data loaded successfully! Main: {len(df)}, Priority: {len(priority_df)}")
            return df, priority_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def get_api_data(endpoint):
    """Get data from API if available"""
    try:
        response = requests.get(f"http://127.0.0.1:8000{endpoint}")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Main title
st.title("AI-Powered Marketing Decision Dashboard")
st.markdown("---")

# Load data
df, priority_df = load_data()

if df.empty or priority_df.empty:
    if DEMO:
        st.error("Demo data not found. Please run:")
        st.code("python create_sample_data.py")
    else:
        st.error("Data files not found. Please run the pipeline first:")
        st.code("""
    1. python data_analysis.py
    2. python model_training.py  
    3. python prioritization.py
    """)
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose Page:", [
    "Overview Dashboard", 
    "User Analysis", 
    "Segmentation Insights",
    "Priority Management",
    "User Search"
])

# Function to create KPI cards
def create_kpi_card(title, value, delta=None, delta_color="normal"):
    return st.metric(
        label=title,
        value=f"{value:,}" if isinstance(value, (int, float)) else str(value),
        delta=delta,
        delta_color=delta_color
    )

# Page 1: Overview Dashboard
if page == "Overview Dashboard":
    st.header("System Overview")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_kpi_card("Total Users", len(df))
    
    with col2:
        create_kpi_card("Priority Users", len(priority_df))
    
    with col3:
        hot_leads = (df['segment'] == "HOT LEAD").sum()
        create_kpi_card("Hot Leads", hot_leads)
    
    with col4:
        avg_score = df['abandonment_score'].mean()
        create_kpi_card("Avg Score", f"{avg_score:.3f}")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Distribution")
        segment_counts = df['segment'].value_counts()
        fig_segment = px.bar(
            x=segment_counts.index, 
            y=segment_counts.values,
            title="Users by Segment",
            color=segment_counts.index,
            color_discrete_map={
                "HOT LEAD": "#FF6B6B",
                "AT RISK": "#FFA500", 
                "NORMAL": "#4ECDC4"
            }
        )
        fig_segment.update_layout(showlegend=False)
        st.plotly_chart(fig_segment, width='stretch')
    
    with col2:
        st.subheader("Action Distribution")
        action_counts = df['action'].value_counts()
        fig_action = px.bar(
            x=action_counts.index,
            y=action_counts.values,
            title="Recommended Actions",
            color=action_counts.index,
            color_discrete_map={
                "Offer 20% discount": "#FF4B4B",
                "Send reminder email": "#FFA500",
                "Show urgency message": "#FFD700",
                "No action": "#95A5A6"
            }
        )
        fig_action.update_layout(showlegend=False)
        st.plotly_chart(fig_action, width='stretch')
    
    # Priority Distribution
    st.subheader("Priority Distribution (Top Users)")
    priority_counts = priority_df['priority'].value_counts()
    fig_priority = px.bar(
        x=priority_counts.index,
        y=priority_counts.values,
        title="Priority Levels",
        color=priority_counts.index,
        color_discrete_map={
            "CRITICAL": "#E74C3C",
            "HIGH": "#F39C12",
            "MEDIUM": "#F1C40F",
            "LOW": "#95A5A6"
        }
    )
    fig_priority.update_layout(showlegend=False)
    st.plotly_chart(fig_priority, width='stretch')

# Page 2: User Analysis
elif page == "User Analysis":
    st.header("User Behavior Analysis")
    
    # Score Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        fig_hist = px.histogram(
            df, 
            x='abandonment_score', 
            nbins=50,
            title="Abandonment Score Distribution",
            color_discrete_sequence=["#3498DB"]
        )
        fig_hist.add_vline(x=df['abandonment_score'].mean(), line_dash="dash", 
                         line_color="red", annotation_text=f"Mean: {df['abandonment_score'].mean():.3f}")
        st.plotly_chart(fig_hist, width='stretch')
    
    with col2:
        st.subheader("Score Statistics")
        stats_data = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
            'Value': [
                f"{df['abandonment_score'].mean():.3f}",
                f"{df['abandonment_score'].median():.3f}",
                f"{df['abandonment_score'].std():.3f}",
                f"{df['abandonment_score'].min():.3f}",
                f"{df['abandonment_score'].max():.3f}",
                f"{df['abandonment_score'].quantile(0.25):.3f}",
                f"{df['abandonment_score'].quantile(0.75):.3f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), width='stretch')
    
    # Segment vs Score Box Plot
    st.subheader("Score Distribution by Segment")
    fig_box = px.box(
        df, 
        x='segment', 
        y='abandonment_score',
        title="Score Distribution Across Segments",
        color='segment',
        color_discrete_map={
            "HOT LEAD": "#FF6B6B",
            "AT RISK": "#FFA500", 
            "NORMAL": "#4ECDC4"
        }
    )
    st.plotly_chart(fig_box, width='stretch')

# Page 3: Segmentation Insights
elif page == "Segmentation Insights":
    st.header("Segmentation Insights")
    
    # Segment Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Breakdown")
        segment_analysis = df.groupby('segment').agg({
            'abandonment_score': ['count', 'mean', 'std']
        }).round(3)
        segment_analysis.columns = ['Count', 'Avg Score', 'Std Dev']
        st.dataframe(segment_analysis, width='stretch')
    
    with col2:
        st.subheader("Action Effectiveness")
        action_analysis = df.groupby(['segment', 'action']).size().unstack(fill_value=0)
        st.dataframe(action_analysis, width='stretch')
    
    # Pie Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Pie Chart")
        fig_pie1 = px.pie(
            values=df['segment'].value_counts().values,
            names=df['segment'].value_counts().index,
            title="User Segments"
        )
        st.plotly_chart(fig_pie1, width='stretch')
    
    with col2:
        st.subheader("Action Pie Chart")
        fig_pie2 = px.pie(
            values=df['action'].value_counts().values,
            names=df['action'].value_counts().index,
            title="Recommended Actions"
        )
        st.plotly_chart(fig_pie2, width='stretch')

# Page 4: Priority Management
elif page == "Priority Management":
    st.header("Priority User Management")
    
    # Priority Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        critical_users = (priority_df['priority'] == 'CRITICAL').sum()
        st.metric("Critical Users", critical_users)
    
    with col2:
        high_users = (priority_df['priority'] == 'HIGH').sum()
        st.metric("High Priority Users", high_users)
    
    with col3:
        avg_priority_score = priority_df['abandonment_score'].mean()
        st.metric("Avg Priority Score", f"{avg_priority_score:.3f}")
    
    st.markdown("---")
    
    # Priority Users Table
    st.subheader("Top Priority Users")
    
    top_n = st.slider("Select number of users to display", 5, 100, 20)
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.selectbox("Filter by Priority", ["All"] + list(priority_df['priority'].unique()))
    
    with col2:
        segment_filter = st.selectbox("Filter by Segment", ["All"] + list(priority_df['segment'].unique()))
    
    with col3:
        action_filter = st.selectbox("Filter by Action", ["All"] + list(priority_df['action'].unique()))
    
    # Apply filters
    filtered_priority = priority_df.copy()
    
    if priority_filter != "All":
        filtered_priority = filtered_priority[filtered_priority['priority'] == priority_filter]
    
    if segment_filter != "All":
        filtered_priority = filtered_priority[filtered_priority['segment'] == segment_filter]
    
    if action_filter != "All":
        filtered_priority = filtered_priority[filtered_priority['action'] == action_filter]
    
    # Display filtered results
    st.dataframe(
        filtered_priority.head(top_n).style.background_gradient(
            subset=['abandonment_score'], 
            cmap='Reds'
        ),
        width='stretch'
    )
    
    # Export functionality
    if st.button("Export Filtered Data"):
        csv = filtered_priority.head(top_n).to_csv(index=True)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="priority_users.csv",
            mime="text/csv"
        )

# Page 5: User Search
elif page == "User Search":
    st.header("User Search & Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input("Enter User ID", step=1, value=0)
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("Search User", type="primary")
    
    if search_button or user_id:
        if user_id in df.index:
            user = df.loc[user_id]
            
            # User Details Card
            st.success(f"User {user_id} Found!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Abandonment Score", f"{user['abandonment_score']:.3f}")
            
            with col2:
                st.metric("Segment", user['segment'])
            
            with col3:
                st.metric("Recommended Action", user['action'])
            
            # Detailed Information
            st.subheader("Detailed Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "User ID": int(user_id),
                    "Score": float(user['abandonment_score']),
                    "Segment": user['segment'],
                    "Action": user['action']
                })
            
            with col2:
                # Priority check
                if user_id in priority_df.index:
                    priority_info = priority_df.loc[user_id]
                    st.success("This user is in the Priority List!")
                    st.json({
                        "Priority": priority_info['priority'],
                        "Priority Score": float(priority_info['abandonment_score'])
                    })
                else:
                    st.info("This user is not in the Priority List")
            
            # API Integration
            st.subheader("API Integration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Refresh from API"):
                    try:
                        api_data = get_api_data(f"/user/{user_id}")
                        if api_data:
                            st.success("Data retrieved from API!")
                            st.json(api_data)
                        else:
                            st.error("API not running or user not found")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col2:
                if st.button("Get Personalized Message"):
                    try:
                        # This would need to be implemented in the API
                        st.info("Personalized messaging feature coming soon!")
                    except:
                        st.error("API not running")
        
        else:
            st.warning(f"User {user_id} not found in the dataset")
            st.info("Try searching for users in the priority list or check the user ID")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built for AI Marketing Solution - built by Rakshan Justin | <a href='https://rakshanjustin.vercel.app' target='_blank'>rakshanjustin.vercel.app</a></p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")
st.sidebar.write(f"**Total Users:** {len(df):,}")
st.sidebar.write(f"**Priority Users:** {len(priority_df):,}")
st.sidebar.write(f"**Hot Leads:** {(df['segment'] == 'HOT LEAD').sum():,}")

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
api_status = get_api_data("/")
if api_status:
    st.sidebar.success("API Running")
else:
    st.sidebar.error("API Offline")

st.sidebar.markdown("---")
st.sidebar.caption("Built by Rakshan")

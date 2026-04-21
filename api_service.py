from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
from typing import Optional, Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Marketing Engine API",
    description="Intelligent Abandoned Cart Prediction and User Prioritization System",
    version="1.0.0"
)

# Global variable to store data
df = None

@app.on_event("startup")
async def startup_event():
    """Load data when the API starts"""
    global df
    try:
        logger.info("Loading prioritized user data...")
        df = pd.read_parquet("priority_users.parquet")
        logger.info(f"Data loaded successfully! Shape: {df.shape}")
        logger.info(f"Available columns: {list(df.columns)}")
    except FileNotFoundError:
        logger.error("Error: priority_users.parquet not found. Please run prioritization.py first.")
        df = pd.DataFrame()  # Empty dataframe as fallback
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        df = pd.DataFrame()

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "AI Marketing Engine Running",
        "status": "healthy",
        "version": "1.0.0",
        "data_loaded": len(df) > 0 if df is not None else False
    }

@app.get("/user/{user_id}")
def get_user(user_id: int):
    """Get specific user details and recommendations"""
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        user = df.loc[user_id] if user_id in df.index else None
        
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user_id,
            "abandonment_score": float(user['abandonment_score']),
            "segment": user['segment'],
            "action": user['action'],
            "priority": user['priority']
        }
    except Exception as e:
        logger.error(f"Error retrieving user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/top-users")
def top_users(limit: int = 10):
    """Get top priority users"""
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    try:
        top_users_data = df.head(limit)
        return top_users_data.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error retrieving top users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/summary")
def summary():
    """Get summary statistics of prioritized users"""
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        # Segment distribution
        segment_counts = df['segment'].value_counts().to_dict()
        
        # Priority distribution
        priority_counts = df['priority'].value_counts().to_dict()
        
        # Action distribution
        action_counts = df['action'].value_counts().to_dict()
        
        # Score statistics
        score_stats = {
            "mean_score": float(df['abandonment_score'].mean()),
            "max_score": float(df['abandonment_score'].max()),
            "min_score": float(df['abandonment_score'].min()),
            "median_score": float(df['abandonment_score'].median())
        }
        
        return {
            "total_users": len(df),
            "segment_distribution": segment_counts,
            "priority_distribution": priority_counts,
            "action_distribution": action_counts,
            "score_statistics": score_stats
        }
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/users/by-priority/{priority}")
def users_by_priority(priority: str, limit: int = 20):
    """Get users filtered by priority level"""
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    valid_priorities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    if priority not in valid_priorities:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid priority. Must be one of: {valid_priorities}"
        )
    
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    try:
        filtered_users = df[df['priority'] == priority].head(limit)
        return filtered_users.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error retrieving users by priority {priority}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/users/by-segment/{segment}")
def users_by_segment(segment: str, limit: int = 20):
    """Get users filtered by segment"""
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    valid_segments = ['HOT LEAD', 'AT RISK', 'NORMAL']
    if segment not in valid_segments:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid segment. Must be one of: {valid_segments}"
        )
    
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    try:
        filtered_users = df[df['segment'] == segment].head(limit)
        return filtered_users.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error retrieving users by segment {segment}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats/score-distribution")
def score_distribution():
    """Get detailed score distribution statistics"""
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        # Create score bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        df['score_bin'] = pd.cut(df['abandonment_score'], bins=bins, labels=labels, include_lowest=True)
        score_dist = df['score_bin'].value_counts().sort_index().to_dict()
        
        # Remove the temporary column
        df.drop('score_bin', axis=1, inplace=True)
        
        return {
            "score_distribution": score_dist,
            "total_users": len(df)
        }
    except Exception as e:
        logger.error(f"Error calculating score distribution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "data_loaded": len(df) > 0 if df is not None else False,
        "total_users": len(df) if df is not None else 0,
        "api_version": "1.0.0",
        "endpoints": [
            "/",
            "/health",
            "/user/{user_id}",
            "/top-users",
            "/summary",
            "/users/by-priority/{priority}",
            "/users/by-segment/{segment}",
            "/stats/score-distribution"
        ]
    }

if __name__ == "__main__":
    print("Starting AI Marketing Engine API...")
    print("Available endpoints:")
    print("  GET /                    - Health check")
    print("  GET /health              - Detailed health check")
    print("  GET /user/{user_id}      - Get specific user")
    print("  GET /top-users           - Get top users")
    print("  GET /summary             - Get summary statistics")
    print("  GET /users/by-priority/{priority} - Filter by priority")
    print("  GET /users/by-segment/{segment}   - Filter by segment")
    print("  GET /stats/score-distribution     - Score distribution")
    print("\nStarting server on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

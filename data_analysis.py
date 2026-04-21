
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Loading datasets...")

# File paths
file_path_nov = '/Users/rakshanjustin/SALES COPILOT/Raw Dataset/archive/2019-Nov.csv'
file_path_oct = '/Users/rakshanjustin/SALES COPILOT/Raw Dataset/archive/2019-Oct.csv'

try:
    # ---------------- LOAD DATA (OPTIMIZED) ----------------
    use_cols = ['event_time', 'event_type', 'product_id', 'category_code', 'brand', 'price', 'user_id']

    df_nov = pd.read_csv(file_path_nov, usecols=use_cols)
    df_oct = pd.read_csv(file_path_oct, usecols=use_cols)

    print(f"November Shape: {df_nov.shape}")
    print(f"October Shape: {df_oct.shape}")

    print("\n" + "="*50 + "\n")

    # ---------------- COMBINE DATA ----------------
    print("Combining datasets...")
    df = pd.concat([df_oct, df_nov], ignore_index=True)
    del df_oct, df_nov  # free memory

    print(f"Combined Shape: {df.shape}")

    print("\n" + "="*50 + "\n")

    # ---------------- BASIC INFO ----------------
    print("Basic Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # ---------------- OPTIMIZE MEMORY ----------------
    print("\nOptimizing memory...")

    df['event_type'] = df['event_type'].astype('category')
    df['category_code'] = df['category_code'].astype('category')
    df['brand'] = df['brand'].astype('category')

    df['product_id'] = pd.to_numeric(df['product_id'], downcast='integer')
    df['user_id'] = pd.to_numeric(df['user_id'], downcast='integer')
    df['price'] = pd.to_numeric(df['price'], downcast='float')

    print("Memory optimized.")

    # ---------------- TIME PROCESSING ----------------
    print("\nProcessing timestamps...")
    df['event_time'] = pd.to_datetime(df['event_time'])

    df = df.sort_values(by=['user_id', 'event_time'])

    # ---------------- CORE ANALYSIS ----------------

    print("\nEvent Distribution:")
    event_counts = df['event_type'].value_counts()
    print(event_counts)

    print("\nUnique Counts:")
    print("Users:", df['user_id'].nunique())
    print("Products:", df['product_id'].nunique())

    # ---------------- CART vs PURCHASE ----------------
    cart_users = df[df['event_type'] == 'cart']['user_id'].nunique()
    purchase_users = df[df['event_type'] == 'purchase']['user_id'].nunique()

    print("\nCart Users:", cart_users)
    print("Purchase Users:", purchase_users)

    # ---------------- VIEW ANALYSIS ----------------
    print("\nView Behavior Analysis...")

    views = df[df['event_type'] == 'view']

    views_per_user = views.groupby('user_id').size()

    print("\nViews per user stats:")
    print(views_per_user.describe())

    # ---------------- SAMPLE FOR VISUALIZATION (SAFE) ----------------
    print("\nCreating sample for visualization...")

    sample_df = df.sample(frac=0.01, random_state=42)  # 1% sample

    # ---------------- VISUALS ----------------
    print("\nGenerating plots...")

    plt.figure(figsize=(8,5))
    sample_df['event_type'].value_counts().plot(kind='bar')
    plt.title("Event Type Distribution (Sample)")
    plt.savefig("event_distribution.png")
    plt.close()

    plt.figure(figsize=(8,5))
    sns.histplot(sample_df['price'], bins=50)
    plt.title("Price Distribution (Sample)")
    plt.savefig("price_distribution.png")
    plt.close()

    # ---------------- TIME GAP ANALYSIS ----------------
    print("\nCalculating time gaps...")

    df['time_diff'] = df.groupby('user_id')['event_time'].diff().dt.seconds

    print(df['time_diff'].describe())

    print("\nEDA Completed Successfully!")

    # ---------------- USER-LEVEL FEATURE ENGINEERING ----------------
    print("\n" + "="*60)
    print("USER-LEVEL FEATURE ENGINEERING")
    print("="*60)

    import os

    if os.path.exists("user_level_data.parquet"):
        print("Loading existing user-level dataset...")
        user_df = pd.read_parquet("user_level_data.parquet")
    else:
        print("Building user-level dataset...")
        
        # Single efficient aggregation
        print("Aggregating user behaviors...")
        
        # Create event type indicators for efficient aggregation
        df['is_view'] = (df['event_type'] == 'view').astype('int8')
        df['is_cart'] = (df['event_type'] == 'cart').astype('int8')
        df['is_purchase'] = (df['event_type'] == 'purchase').astype('int8')
        
        # Single groupby aggregation
        user_agg = df.groupby('user_id').agg({
            'is_view': 'sum',
            'is_cart': 'sum', 
            'is_purchase': 'sum',
            'event_type': 'count',  # total_events
            'time_diff': 'mean'      # avg_time_diff
        }).rename(columns={
            'is_view': 'views_count',
            'is_cart': 'cart_count',
            'is_purchase': 'purchase_count',
            'event_type': 'total_events',
            'time_diff': 'avg_time_diff'
        })
        
        # Clean up temporary columns
        df.drop(['is_view', 'is_cart', 'is_purchase'], axis=1, inplace=True)
        
        print("Creating targets...")
        
        # Create target variables
        user_df = user_agg.copy()
        user_df['abandoned_cart'] = ((user_df['cart_count'] > 0) & (user_df['purchase_count'] == 0)).astype('int8')
        user_df['high_intent'] = ((user_df['views_count'] > 5) & (user_df['purchase_count'] == 0)).astype('int8')
        
        print("Creating behavioral features...")
        
        # Create behavioral features with safe division
        user_df['conversion_rate'] = user_df['purchase_count'] / (user_df['views_count'] + 1)
        user_df['cart_to_purchase'] = user_df['purchase_count'] / (user_df['cart_count'] + 1)
        
        # Clean up aggregation dataframe
        del user_agg
        
        print("Saving dataset...")
        user_df.to_parquet("user_level_data.parquet")
        print("User-level dataset saved!")

    # Output results
    print(f"\nUser-level dataset shape: {user_df.shape}")
    print("\nFirst 5 rows:")
    print(user_df.head())
    
    print("\nTarget distributions:")
    print("Abandoned Cart:")
    print(user_df['abandoned_cart'].value_counts())
    print("\nHigh Intent:")
    print(user_df['high_intent'].value_counts())
    
    print("\nUser-Level Feature Engineering Completed!")

    # ---------------- MODEL TRAINING & PREDICTION - ABANDONED CART ----------------
    print("\n" + "="*60)
    print("MODEL TRAINING & PREDICTION - ABANDONED CART")
    print("="*60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    print("Preparing data for model...")

    # Define leakage-free behavioral features
    feature_cols = [
        'views_count',
        'total_events', 
        'avg_time_diff'
    ]

    print("Using leakage-free features for realistic modeling...")

    # Handle missing values
    user_df_clean = user_df[feature_cols + ['abandoned_cart']].copy()
    user_df_clean = user_df_clean.fillna(0)

    # Prepare features and target
    X = user_df_clean[feature_cols]
    y = user_df_clean['abandoned_cart']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    print("Training model...")

    # Train RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    # Predict on test data
    y_pred = model.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Generating predictions...")

    # Generate prediction scores for all users
    user_df['abandonment_score'] = model.predict_proba(X)[:, 1]

    print("Identifying high-risk users...")

    # Identify high-risk users (score > 0.7)
    high_risk_users = user_df[user_df['abandonment_score'] > 0.7]

    print(f"\nHigh-risk users count: {len(high_risk_users):,}")
    print(f"Percentage of total users: {len(high_risk_users)/len(user_df)*100:.2f}%")

    # Display sample of high-risk users
    print("\nSample of high-risk users:")
    print(high_risk_users[['abandonment_score', 'views_count', 'cart_count', 'purchase_count']].head(10))

    # Feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance)

    print("\nModel Training & Prediction Completed!")

except Exception as e:
    print(f"Error: {e}")
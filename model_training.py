import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("ABANDONED CART PREDICTION MODEL")
    print("=" * 60)
    
    # ---------------- LOAD USER DATASET ----------------
    print("Loading user dataset...")
    
    try:
        user_df = pd.read_parquet("user_level_data.parquet")
        print(f"Dataset loaded successfully! Shape: {user_df.shape}")
    except FileNotFoundError:
        print("Error: user_level_data.parquet not found. Please run data_analysis.py first.")
        return
    
    # ---------------- PREPARE DATA ----------------
    print("Preparing data...")
    
    # Define leakage-free behavioral features
    features = [
        'views_count',
        'total_events', 
        'avg_time_diff'
    ]
    
    target = 'abandoned_cart'
    
    # Prepare features and target
    X = user_df[features].fillna(0)
    y = user_df[target]
    
    print(f"Features: {features}")
    print(f"Target: {target}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    # ---------------- TRAIN-TEST SPLIT ----------------
    print("\nSplitting dataset...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # ---------------- TRAIN MODEL ----------------
    print("Training model...")
    
    # Train improved model with class balancing
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # ---------------- EVALUATE MODEL ----------------
    print("Evaluating model...")
    
    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # ---------------- GENERATE PREDICTIONS ----------------
    print("Generating predictions...")
    
    # Generate prediction scores for all users
    user_df['abandonment_score'] = model.predict_proba(X)[:, 1]
    
    # ---------------- IDENTIFY HIGH-RISK USERS ----------------
    print("Identifying high-risk users...")
    
    # Use lower threshold for better business recall
    threshold = 0.4
    high_risk_users = user_df[user_df['abandonment_score'] > threshold]
    
    print(f"\nHigh-Risk Users Analysis:")
    print(f"Threshold: {threshold}")
    print(f"High-risk users count: {len(high_risk_users):,}")
    print(f"Percentage of total users: {len(high_risk_users)/len(user_df)*100:.2f}%")
    
    # Display sample of high-risk users
    print("\nTop 10 High-Risk Users:")
    high_risk_sorted = high_risk_users.sort_values('abandonment_score', ascending=False)
    print(high_risk_sorted[['abandonment_score', 'views_count', 'total_events', 'avg_time_diff']].head(10))
    
    # ---------------- FEATURE IMPORTANCE ----------------
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)
    
    # ---------------- SAVE PREDICTIONS ----------------
    print("Saving predictions...")
    
    user_df.to_parquet("user_predictions.parquet")
    print("Predictions saved as 'user_predictions.parquet'")
    
    # ---------------- BUSINESS INSIGHTS ----------------
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)
    
    # Calculate potential impact
    total_abandoned = len(user_df[user_df['abandoned_cart'] == 1])
    identified_at_risk = len(high_risk_users[high_risk_users['abandoned_cart'] == 1])
    capture_rate = identified_at_risk / total_abandoned if total_abandoned > 0 else 0
    
    print(f"Total abandoned cart users: {total_abandoned:,}")
    print(f"Identified at-risk users: {identified_at_risk:,}")
    print(f"Capture rate: {capture_rate:.2%}")
    
    # Revenue impact estimation (if we had price data)
    print(f"\nModel Performance Summary:")
    print(f"- Overall accuracy: {(y_pred == y_test).mean():.3f}")
    print(f"- High-risk identification: {len(high_risk_users):,} users")
    print(f"- Model recall for abandoned carts: {recall:.3f}")
    
    print("\nModel Training & Prediction Completed Successfully!")
    print("="*60)

    # ---------------- DECISION ENGINE & BUSINESS OUTPUT ----------------
    print("\n" + "="*60)
    print("DECISION ENGINE & BUSINESS OUTPUT")
    print("="*60)

    # ---------------- IMPROVED RECALL WITH THRESHOLD TUNING ----------------
    print("\nApplying improved threshold for better recall...")

    threshold = 0.4
    high_risk_users = user_df[user_df['abandonment_score'] > threshold]

    print(f"High-risk users count: {len(high_risk_users):,}")
    print(f"Percentage of total users: {len(high_risk_users)/len(user_df)*100:.2f}%")

    # ---------------- BUILD DECISION ENGINE ----------------
    print("\nBuilding decision engine...")

    def assign_action(score):
        if score > 0.7:
            return "Offer 20% discount"
        elif score > 0.5:
            return "Send reminder email"
        elif score > 0.3:
            return "Show urgency message"
        else:
            return "No action"

    print("Assigning actions...")
    user_df['action'] = user_df['abandonment_score'].apply(assign_action)

    # ---------------- CREATE USER SEGMENTS ----------------
    print("Creating segments...")

    import numpy as np

    user_df['segment'] = np.where(
        (user_df['high_intent'] == 1) & (user_df['abandonment_score'] > 0.5),
        "HOT LEAD",
        np.where(
            user_df['abandonment_score'] > 0.5,
            "AT RISK",
            "NORMAL"
        )
    )

    # ---------------- PRINT SEGMENT DISTRIBUTION ----------------
    print("\nSegment Distribution:")
    segment_counts = user_df['segment'].value_counts()
    for segment, count in segment_counts.items():
        print(f"{segment}: {count:,} ({count/len(user_df)*100:.1f}%)")

    # ---------------- CREATE FINAL OUTPUT DATASET ----------------
    print("\nCreating final output dataset...")

    final_output = user_df[[
        'abandonment_score',
        'segment',
        'action'
    ]].copy()

    # ---------------- SHOW SAMPLE OUTPUT ----------------
    print("\nSample Final Output:")
    print(final_output.head(10))

    # ---------------- SAVE FINAL DATASET ----------------
    print("Saving final output...")
    
    final_output.to_parquet("final_marketing_output.parquet")
    print("Final output saved successfully!")

    # ---------------- BUSINESS SUMMARY ----------------
    print("\n" + "="*60)
    print("BUSINESS SUMMARY")
    print("="*60)
    
    # Action distribution
    action_counts = user_df['action'].value_counts()
    print("\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"{action}: {count:,} users")

    # High-value leads (HOT LEAD)
    hot_leads = user_df[user_df['segment'] == 'HOT LEAD']
    print(f"\nHigh-Value Hot Leads: {len(hot_leads):,} users")
    if len(hot_leads) > 0:
        print(f"Average abandonment score: {hot_leads['abandonment_score'].mean():.3f}")
        print(f"Top actions for hot leads:")
        hot_actions = hot_leads['action'].value_counts()
        for action, count in hot_actions.head(3).items():
            print(f"  {action}: {count:,}")

    print("\nDecision Engine & Business Output Completed!")
    print("="*60)

if __name__ == "__main__":
    main()

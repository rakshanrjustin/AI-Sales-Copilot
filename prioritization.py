import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("PRIORITIZATION LAYER")
    print("=" * 60)
    
    # ---------------- LOAD PREDICTIONS ----------------
    print("Loading marketing predictions...")
    
    try:
        df = pd.read_parquet("final_marketing_output.parquet")
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: final_marketing_output.parquet not found. Please run model_training.py first.")
        return
    
    # ---------------- RANKING SYSTEM ----------------
    print("\nRanking users by abandonment score...")
    
    # Sort users by abandonment_score (descending)
    df = df.sort_values(by='abandonment_score', ascending=False)
    
    print(f"Users ranked successfully!")
    print(f"Highest score: {df['abandonment_score'].max():.3f}")
    print(f"Lowest score: {df['abandonment_score'].min():.3f}")
    
    # ---------------- PRIORITIZATION LOGIC ----------------
    print("\nApplying prioritization logic...")
    
    # Configurable top selection
    TOP_PERCENT = 0.1  # top 10%
    
    top_k = int(len(df) * TOP_PERCENT)
    priority_users = df.head(top_k)
    
    print(f"Top {TOP_PERCENT*100:.1f}% users selected: {len(priority_users):,}")
    
    # ---------------- PRIORITY LABELS ----------------
    print("Assigning priority labels...")
    
    def assign_priority(score):
        if score > 0.8:
            return "CRITICAL"
        elif score > 0.6:
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    priority_users['priority'] = priority_users['abandonment_score'].apply(assign_priority)
    
    # ---------------- PRIORITY DISTRIBUTION ----------------
    print("\nPriority Distribution:")
    priority_counts = priority_users['priority'].value_counts()
    for priority, count in priority_counts.items():
        percentage = count / len(priority_users) * 100
        print(f"{priority}: {count:,} ({percentage:.1f}%)")
    
    # ---------------- SEGMENT BREAKDOWN ----------------
    print("\nSegment Breakdown (Priority Users):")
    segment_counts = priority_users['segment'].value_counts()
    for segment, count in segment_counts.items():
        percentage = count / len(priority_users) * 100
        print(f"{segment}: {count:,} ({percentage:.1f}%)")
    
    # ---------------- ACTION BREAKDOWN ----------------
    print("\nRecommended Actions (Priority Users):")
    action_counts = priority_users['action'].value_counts()
    for action, count in action_counts.items():
        percentage = count / len(priority_users) * 100
        print(f"{action}: {count:,} ({percentage:.1f}%)")
    
    # ---------------- SAMPLE PRIORITY USERS ----------------
    print("\nSample Priority Users (Top 10):")
    sample_users = priority_users.head(10)
    for idx, user in sample_users.iterrows():
        print(f"User {idx}: Score={user['abandonment_score']:.3f}, Priority={user['priority']}, Action={user['action']}")
    
    # ---------------- SAVE PRIORITIZED OUTPUT ----------------
    print("\nSaving prioritized users...")
    
    priority_users.to_parquet("priority_users.parquet")
    print("Prioritized users saved as 'priority_users.parquet'")
    
    # ---------------- BUSINESS METRICS ----------------
    print("\n" + "="*60)
    print("BUSINESS METRICS")
    print("="*60)
    
    # Calculate potential impact
    total_users = len(df)
    selected_users = len(priority_users)
    selection_rate = selected_users / total_users
    
    # Critical users analysis
    critical_users = priority_users[priority_users['priority'] == 'CRITICAL']
    hot_leads_priority = priority_users[priority_users['segment'] == 'HOT LEAD']
    
    print(f"Total Users: {total_users:,}")
    print(f"Priority Users: {selected_users:,}")
    print(f"Selection Rate: {selection_rate:.1%}")
    print(f"Critical Users: {len(critical_users):,}")
    print(f"Hot Leads in Priority: {len(hot_leads_priority):,}")
    
    # Efficiency metrics
    if len(critical_users) > 0:
        print(f"Average Critical Score: {critical_users['abandonment_score'].mean():.3f}")
    
    # Cost efficiency estimation
    print(f"\nCost Efficiency Analysis:")
    print(f"- Targeting {selection_rate:.1%} of users instead of 100%")
    print(f"- Potential cost reduction: {(1-selection_rate)*100:.1f}%")
    print(f"- High-value focus: {len(hot_leads_priority)/selected_users*100:.1f}% of priority users are HOT LEADS")
    
    print("\nPrioritization Layer Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

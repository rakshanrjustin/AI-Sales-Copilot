import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("CREATING SAMPLE DATA FOR DEMO MODE")
    print("=" * 60)
    
    try:
        # Load the full marketing output dataset
        print("Loading full marketing output dataset...")
        df = pd.read_parquet("final_marketing_output.parquet")
        print(f"Full dataset loaded: {len(df):,} rows")
        
        # Sample 5000 rows for demo mode
        sample_size = 5000
        if len(df) >= sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            # If dataset is smaller than 5000, use all data
            sample_df = df.copy()
            print(f"Dataset smaller than {sample_size}, using all {len(df)} rows")
        
        print(f"Sample created: {len(sample_df):,} rows")
        
        # Display sample statistics
        print("\nSample Dataset Statistics:")
        print(f"Shape: {sample_df.shape}")
        print(f"Columns: {list(sample_df.columns)}")
        
        print("\nSegment Distribution:")
        segment_counts = sample_df['segment'].value_counts()
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count} ({count/len(sample_df)*100:.1f}%)")
        
        print("\nAction Distribution:")
        action_counts = sample_df['action'].value_counts()
        for action, count in action_counts.items():
            print(f"  {action}: {count} ({count/len(sample_df)*100:.1f}%)")
        
        print("\nScore Statistics:")
        print(f"  Mean: {sample_df['abandonment_score'].mean():.3f}")
        print(f"  Median: {sample_df['abandonment_score'].median():.3f}")
        print(f"  Min: {sample_df['abandonment_score'].min():.3f}")
        print(f"  Max: {sample_df['abandonment_score'].max():.3f}")
        
        # Save sample data
        print("\nSaving sample data...")
        sample_df.to_parquet("sample_data.parquet")
        print("Sample data saved as 'sample_data.parquet'")
        
        # Verify the saved file
        print("\nVerifying saved file...")
        verification_df = pd.read_parquet("sample_data.parquet")
        print(f"Verification successful: {len(verification_df):,} rows loaded")
        
        print("\n" + "="*60)
        print("SAMPLE DATA CREATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Ready for Streamlit Cloud deployment:")
        print("1. Set DEMO = True in dashboard.py")
        print("2. Run: streamlit run dashboard.py")
        print("3. Deploy to Streamlit Cloud")
        
    except FileNotFoundError:
        print("Error: final_marketing_output.parquet not found!")
        print("Please run the full pipeline first:")
        print("  1. python data_analysis.py")
        print("  2. python model_training.py")
        print("  3. python prioritization.py")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")

if __name__ == "__main__":
    main()

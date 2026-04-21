import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options for better visibility
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("DATA CLEANING AND PREPROCESSING")
print("=" * 60)

# Load the CSV files
file_path_nov = '/Users/rakshanjustin/SALES COPILOT/Raw Dataset/archive/2019-Nov.csv'
file_path_oct = '/Users/rakshanjustin/SALES COPILOT/Raw Dataset/archive/2019-Oct.csv'

# Load data with optimized parameters for large datasets
print("Loading datasets with optimized parameters...")
df_nov = pd.read_csv(file_path_nov, dtype={
    'event_type': 'category',
    'category_code': 'object',
    'brand': 'object',
    'user_session': 'object'
})
df_oct = pd.read_csv(file_path_oct, dtype={
    'event_type': 'category',
    'category_code': 'object',
    'brand': 'object',
    'user_session': 'object'
})

print(f"November 2019: {df_nov.shape[0]:,} rows")
print(f"October 2019: {df_oct.shape[0]:,} rows")

# Combine datasets for analysis
df_nov['month'] = 'November'
df_oct['month'] = 'October'
df_combined = pd.concat([df_nov, df_oct], ignore_index=True)
print(f"Combined dataset: {df_combined.shape[0]:,} rows")

# Convert event_time to datetime
print("\nConverting event_time to datetime...")
df_combined['event_time'] = pd.to_datetime(df_combined['event_time'], format='mixed')
df_combined['date'] = df_combined['event_time'].dt.date
df_combined['hour'] = df_combined['event_time'].dt.hour
df_combined['day_of_week'] = df_combined['event_time'].dt.day_name()

# Handle missing values
print("\nHandling missing values...")
print("Before cleaning:")
print(f"Missing category_code: {df_combined['category_code'].isnull().sum():,} ({df_combined['category_code'].isnull().sum()/len(df_combined)*100:.1f}%)")
print(f"Missing brand: {df_combined['brand'].isnull().sum():,} ({df_combined['brand'].isnull().sum()/len(df_combined)*100:.1f}%)")

# Fill missing categorical values with 'Unknown'
df_combined['category_code'] = df_combined['category_code'].fillna('Unknown')
df_combined['brand'] = df_combined['brand'].fillna('Unknown')

# Drop rows with missing user_session (very few)
df_combined = df_combined.dropna(subset=['user_session'])

print("After cleaning:")
print(f"Missing category_code: {df_combined['category_code'].isnull().sum():,}")
print(f"Missing brand: {df_combined['brand'].isnull().sum():,}")
print(f"Final dataset shape: {df_combined.shape[0]:,} rows")

# Extract main category from category_code
print("\nExtracting main categories...")
df_combined['main_category'] = df_combined['category_code'].apply(
    lambda x: x.split('.')[0] if x != 'Unknown' else 'Unknown'
)

# Basic statistics
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print("\nEvent types distribution:")
event_counts = df_combined['event_type'].value_counts()
for event_type, count in event_counts.items():
    print(f"{event_type}: {count:,} ({count/len(df_combined)*100:.1f}%)")

print(f"\nPrice statistics:")
print(f"Mean price: ${df_combined['price'].mean():.2f}")
print(f"Median price: ${df_combined['price'].median():.2f}")
print(f"Min price: ${df_combined['price'].min():.2f}")
print(f"Max price: ${df_combined['price'].max():.2f}")
print(f"Std price: ${df_combined['price'].std():.2f}")

print(f"\nUnique values:")
print(f"Unique users: {df_combined['user_id'].nunique():,}")
print(f"Unique products: {df_combined['product_id'].nunique():,}")
print(f"Unique sessions: {df_combined['user_session'].nunique():,}")
print(f"Unique brands: {df_combined['brand'].nunique():,}")
print(f"Unique main categories: {df_combined['main_category'].nunique()}")

# Save cleaned data for further analysis
print("\nSaving cleaned dataset...")
df_combined.to_csv('/Users/rakshanjustin/SALES COPILOT/cleaned_ecommerce_data.csv', index=False)
print("Cleaned dataset saved as 'cleaned_ecommerce_data.csv'")

print("\nData cleaning and preprocessing completed!")
print("Ready for detailed analysis.")

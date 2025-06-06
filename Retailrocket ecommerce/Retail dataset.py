import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('mercari_recommendation_sample.csv')
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()
print(df.duplicated().sum())

df.to_csv ("cleaned_mercari_recommendation_sample.csv", index=False)
print(df)

# EDA analysis
# Univariate Analysis
#matrix of user-product ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=df, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('rating')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.legend()
plt.show()

#Recommended top 5 products
top_products = df.groupby('item_id')['rating'].mean().sort_values(ascending=False).head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.index, y=top_products.values, palette='viridis')
plt.title('Top 5 Recommended Products')
plt.xlabel('item_id')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.tight_layout()
plt.legend()
plt.show()

#Heatmap of user-product ratings
plt.figure(figsize=(12, 8))
sns.heatmap(df.pivot_table(index='user_id', columns='item_id', values='rating'), cmap='viridis', annot=False, cbar=True)
plt.title('Heatmap of User-Product Ratings')
plt.xlabel('item_id')
plt.ylabel('user_id')
plt.tight_layout()
plt.show()

#similarity  scores between users/products
user_similarity = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

user_similarity = user_similarity.corr()
# Plotting user similarity heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(user_similarity, cmap='viridis', annot=False, cbar=True)
plt.title('User Similarity Heatmap')
plt.xlabel('user_id')
plt.ylabel('product_id')
plt.tight_layout()
plt.show()

#summarize
#Number of unique users and products
num_users = df['user_id'].nunique()
num_products = df['item_id'].nunique()
print(f'Number of unique users: {num_users}')
print(f'Number of unique products: {num_products}')

#conversion rate=purchases/views
conversion_rate = df['views'].mean()
print(f'Average conversion rate (views): {conversion_rate:.2f}')

#Average number of views before a purchase
avg_views_before_purchase = df[df['rating'] > 0]['views'].mean()
print(f'Average number of views before a purchase: {avg_views_before_purchase:.2f}')









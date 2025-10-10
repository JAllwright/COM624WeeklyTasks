# This code sets up your Python environment by importing the necessary libraries.
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting graphs
import seaborn as sns # For advanced visualisations
# Set a clean visual style for plots
sns.set(style="whitegrid")

# Load the dataset into a DataFrame
df = pd.read_csv('retail_sales_final.csv')
# Display the first few rows to understand the structure
print(df.head())
# Check data types and missing values
print(df.info())
# Get summary statistics for all columns
print(df.describe(include='all'))

# Bar chart to show count of missing values per column
missing_counts = df.isnull().sum()
missing_counts.plot(kind='bar', color='orange')
plt.title("Missing Values per Column")
plt.ylabel("Count")
plt.show()
# Histogram to show distribution of a numerical column
df['Sales'].plot(kind='hist', bins=20, color='skyblue')
plt.title("Sales Distribution (Messy)")
plt.xlabel("Sales")
plt.show()
# Pie chart to show proportion of missing vs non-missing
missing_total = df.isnull().sum().sum()
non_missing_total = df.size - missing_total
plt.pie([missing_total, non_missing_total], labels=['Missing', 'Non-Missing'],
autopct='%1.1f%%', colors=['red', 'green'])
plt.title("Overall Missing Data Proportion")
plt.show()

# Heatmap to show missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
# Boxplot to detect outliers in 'Sales'
sns.boxplot(x=df['Sales'])
plt.title("Sales Boxplot (Messy)")
plt.show()

# Check for duplicate rows
print("Number of duplicates:", df.duplicated().sum())
# Drop duplicate rows
df = df.drop_duplicates()
# Handle Missing Values
# Drop rows with any missing values
df_cleaned = df.dropna()
# Alternatively, fill missing values with the mean
# df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
#Standardise Column Names
# Clean column names for consistency
df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_')

# Calculate mean, median, and mode for 'sales'
print("Mean Sales:", df_cleaned['sales'].mean())
print("Median Sales:", df_cleaned['sales'].median())
print("Mode Sales:", df_cleaned['sales'].mode()[0])

# Histogram of 'sales'
df_cleaned['sales'].plot(kind='hist', bins=20, color='green')
plt.title("Sales Distribution (Clean)")
plt.xlabel("Sales")
plt.show()
# Boxplot of 'sales'
sns.boxplot(x=df_cleaned['sales'])
plt.title("Sales Boxplot (Clean)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# Scatter plot of 'sales' vs 'profit'
sns.scatterplot(x='sales', y='profit', data=df_cleaned)
plt.title("Sales vs Profit")
plt.show()

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('retail_sales_clean.csv', index=False)


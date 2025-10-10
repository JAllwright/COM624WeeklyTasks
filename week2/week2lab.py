# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy import stats
# Step 2: Load Dataset
df = pd.read_csv("mobile_device_usage.csv")
df.dropna(inplace=True)
df = df[df['Screen_On_Time'] > 0]
# Step 3: Descriptive Statistics
mean_screen = df['Screen_On_Time'].mean()
median_screen = df['Screen_On_Time'].median()
mode_screen = df['Screen_On_Time'].mode()[0]
std_screen = df['Screen_On_Time'].std()
range_screen = df['Screen_On_Time'].max() - df['Screen_On_Time'].min()
print(f"Mean: {mean_screen:.2f}, Median: {median_screen}, Mode: {mode_screen}")
print(f"Standard Deviation: {std_screen:.2f}, Range: {range_screen}")

# Step 4: Visualisations
plt.figure(figsize=(8, 5))
sns.histplot(df['Screen_On_Time'], bins=20, kde=True)
plt.title("Distribution of Screen-On Time")
plt.xlabel("Hours per Day")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['App_Usage_Time'])
plt.title("Boxplot of App Usage Time")
plt.xlabel("Minutes per Day")
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Variables")
plt.show()
# Step 5: Grouped Analysis
os_group = df.groupby('Operating_System')['Screen_On_Time'].mean()
os_group.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Average Screen-On Time by Operating System")
plt.ylabel("Hours per Day")
plt.show()
# Step 6: Summary Table
print(df.describe())

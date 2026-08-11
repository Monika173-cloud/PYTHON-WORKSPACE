import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your dataset (Replace 'your_dataset.csv' with your actual file path)
# df = pd.read_csv('your_dataset.csv')

# Creating a mock dataframe for demonstration purposes
data = {
    'Age': [25, 30, 35, np.nan, 28, 200, 35], # 200 is an outlier, nan is missing
    'Salary': [50000, 60000, np.nan, 45000, 52000, 58000, 60000],
    'Department': ['IT', 'HR', 'IT', 'Marketing', 'HR', 'IT', 'HR']
}
df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

# ==========================================
# PHASE 1: DATA CLEANING
# ==========================================

# 1. Handle Duplicates
df = df.drop_duplicates()

# 2. Handle Missing Values
# Fill numeric missing values with the median of the column
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# 3. Handle Outliers (Example: Cap Age if it's unrealistic)
# Let's assume any age over 100 is an outlier and replace it with the median
median_age = df['Age'].median()
df.loc[df['Age'] > 100, 'Age'] = median_age

print("\n--- Cleaned Data ---")
print(df)

# ==========================================
# PHASE 2: VISUALIZATION
# ==========================================
# Setting up the visualization style
sns.set_theme(style="whitegrid")

# Create a dashboard layout (1 row, 2 columns of plots)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution of Age (Histogram)
sns.histplot(df['Age'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Age')
axes[0].set_xlabel('Age')

# Plot 2: Average Salary by Department (Barplot)
sns.barplot(x='Department', y='Salary', data=df, ax=axes[1], errorbar=None, palette='pastel')
axes[1].set_title('Average Salary by Department')
axes[1].set_ylabel('Salary')

# Adjust layout and display
plt.tight_layout()
plt.show()
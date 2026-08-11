# ============================================
# TASK 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inside notebook
%matplotlib inline

# Set plot style
sns.set_theme(style="whitegrid")


# ============================================
# 2. Load Dataset
# ============================================

# Replace with your CSV file name
df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print()


# ============================================
# 3. Display Basic Information
# ============================================

print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()


# ============================================
# 4. Check Missing Values
# ============================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Value Percentage:")
print((df.isnull().sum() / len(df)) * 100)


# ============================================
# 5. Check Duplicate Values
# ============================================

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================
# 6. Statistical Summary
# ============================================

print("\nStatistical Summary:")
print(df.describe())


# ============================================
# 7. Categorical Data Summary
# ============================================

print("\nCategorical Columns:")

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:
    print("\n", column)
    print(df[column].value_counts())


# ============================================
# 8. Numerical Columns
# ============================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns

print("\nNumerical Columns:")
print(numerical_columns.tolist())


# ============================================
# 9. Distribution of Numerical Variables
# ============================================

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column].dropna(),
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.show()


# ============================================
# 10. Boxplots - Detect Outliers
# ============================================

for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[column]
    )

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

    plt.show()


# ============================================
# 11. Correlation Analysis
# ============================================

correlation = df[numerical_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)


# ============================================
# 12. Correlation Heatmap
# ============================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# ============================================
# 13. Pairplot
# ============================================

if len(numerical_columns) > 1:

    sns.pairplot(
        df[numerical_columns].dropna()
    )

    plt.show()


# ============================================
# 14. Categorical Data Visualization
# ============================================

for column in categorical_columns:

    # Avoid plotting columns with too many unique values
    if df[column].nunique() <= 10:

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=df,
            x=column
        )

        plt.title(f"Distribution of {column}")
        plt.xticks(rotation=45)

        plt.show()


# ============================================
# 15. Find Strong Correlations
# ============================================

print("\nStrong Correlations:")

corr_matrix = df[numerical_columns].corr()

for i in range(len(corr_matrix.columns)):

    for j in range(i + 1, len(corr_matrix.columns)):

        value = corr_matrix.iloc[i, j]

        if abs(value) >= 0.5:

            print(
                f"{corr_matrix.columns[i]} <-> "
                f"{corr_matrix.columns[j]} : {value:.2f}"
            )


# ============================================
# 16. Generate EDA Report
# ============================================

print("\n================================")
print("        EDA REPORT")
print("================================")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Number of Numerical Columns:")
print(len(numerical_columns))

print("\n3. Number of Categorical Columns:")
print(len(categorical_columns))

print("\n4. Missing Values:")
print(df.isnull().sum().sum())

print("\n5. Duplicate Rows:")
print(df.duplicated().sum())

print("\n6. Strong Correlations:")

for i in range(len(corr_matrix.columns)):

    for j in range(i + 1, len(corr_matrix.columns)):

        value = corr_matrix.iloc[i, j]

        if abs(value) >= 0.5:

            print(
                f"- {corr_matrix.columns[i]} and "
                f"{corr_matrix.columns[j]} "
                f"(Correlation = {value:.2f})"
            )

print("\nEDA Completed Successfully!")
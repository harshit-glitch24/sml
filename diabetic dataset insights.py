#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Install packages (Run once in Jupyter/Colab)
get_ipython().system('pip install pandas numpy scipy matplotlib seaborn')

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import trim_mean, median_abs_deviation
from scipy.stats import probplot

# Plot style
plt.style.use('ggplot')

# Load Diabetes dataset
df = pd.read_csv("diabetes.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Select only numerical columns
num_df = df.select_dtypes(include=np.number)

print("\nNumerical Columns:")
print(num_df.columns)

# ---------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------
print("\n========== DESCRIPTIVE STATISTICS ==========\n")

for col in num_df.columns:
    data = num_df[col].dropna()

    print(f"\n----- {col} -----")

    # Mean
    mean = data.mean()
    print("Mean:", mean)

    # Median
    median = data.median()
    print("Median:", median)

    # Trimmed Mean (10%)
    tmean = trim_mean(data, 0.10)
    print("Trimmed Mean:", tmean)

    # Weighted Mean
    weights = np.arange(1, len(data) + 1)
    wmean = np.average(data, weights=weights)
    print("Weighted Mean:", wmean)

    # Range
    data_range = data.max() - data.min()
    print("Range:", data_range)

    # Variance
    variance = data.var()
    print("Variance:", variance)

    # Standard Deviation
    std = data.std()
    print("Standard Deviation:", std)

    # Interquartile Range (IQR)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    print("IQR:", IQR)

    # Median Absolute Deviation (MAD)
    mad = median_abs_deviation(data)
    print("Median Absolute Deviation:", mad)

# ---------------------------------------
# VISUALIZATIONS
# ---------------------------------------

# 1. Boxplots
for col in num_df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=num_df[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

# 2. Histograms
for col in num_df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(num_df[col], bins=20, kde=False, color='steelblue')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# 3. Density Plots
for col in num_df.columns:
    plt.figure(figsize=(6,4))
    sns.kdeplot(num_df[col], fill=True, color='green')
    plt.title(f'Density Plot of {col}')
    plt.xlabel(col)
    plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap - Diabetes Dataset")
plt.show()

# 5. Scatter Plot Matrix (Pair Plot)
if len(num_df.columns) >= 2:
    sns.pairplot(num_df)
    plt.show()

# 6. Q-Q Plots (Normality Check)
for col in num_df.columns:
    plt.figure(figsize=(6,4))
    probplot(num_df[col].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    plt.show()

print("\n========== Analysis Completed Successfully! ==========")


# In[1]:


# Install packages (Run once in Jupyter/Colab)
# !pip install pandas numpy scipy matplotlib seaborn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import trim_mean, median_abs_deviation
from scipy.stats import probplot, skew

# Plot style
plt.style.use('ggplot')

# Load Diabetes dataset
# Simulated data dictionary based on user's sample format to guarantee execution
data_sample = {
    'Pregnancies': [6, 1, 8, 1, 0],
    'Glucose': [148, 85, 183, 89, 137],
    'BloodPressure': [72, 66, 64, 66, 40],
    'SkinThickness': [35, 29, 0, 23, 35],
    'Insulin': [0, 0, 0, 94, 168],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
    'Age': [50, 31, 32, 21, 33],
    'Outcome': [1, 0, 1, 0, 1]
}
# Replace the line below with df = pd.read_csv("diabetes.csv") if running locally
df = pd.DataFrame(data_sample) 

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Select only numerical columns
num_df = df.select_dtypes(include=np.number)

print("\nNumerical Columns:")
print(num_df.columns)

# ---------------------------------------
# DESCRIPTIVE STATISTICS & INSIGHTS
# ---------------------------------------
print("\n========== DESCRIPTIVE STATISTICS & INSIGHTS ==========\n")

for col in num_df.columns:
    data = num_df[col].dropna()

    print(f"\n==================== {col.upper()} ====================")

    # Central Tendency Calculations
    mean = data.mean()
    median = data.median()
    tmean = trim_mean(data, 0.10)
    
    weights = np.arange(1, len(data) + 1)
    wmean = np.average(data, weights=weights)

    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Trimmed Mean (10%): {tmean:.2f}")
    print(f"Weighted Mean: {wmean:.2f}")

    # Central Tendency Insights
    print("\n[INSIGHT - Central Tendency & Distribution Shape]:")
    if abs(mean - median) < (0.1 * median):
        print(f" -> The Mean ({mean:.2f}) and Median ({median:.2f}) are very close. The distribution of '{col}' is roughly symmetric.")
    elif mean > median:
        print(f" -> The Mean ({mean:.2f}) is higher than the Median ({median:.2f}). '{col}' is Right-Skewed (positively skewed), meaning a few high values are pulling the average up.")
    else:
        print(f" -> The Mean ({mean:.2f}) is lower than the Median ({median:.2f}). '{col}' is Left-Skewed (negatively skewed), meaning a few unusually low values are pulling the average down.")

    if abs(mean - tmean) > (0.05 * mean):
        print(f" -> ALERT: The Trimmed Mean ({tmean:.2f}) differs noticeably from the standard Mean ({mean:.2f}). Extreme values/outliers are heavily impacting the average calculation.")
    else:
        print(f" -> The standard Mean and Trimmed Mean are similar, indicating outliers are not disproportionately shifting the overall center.")

    # Variability Calculations
    data_range = data.max() - data.min()
    variance = data.var()
    std = data.std()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    mad = median_abs_deviation(data)

    print(f"\nRange: {data_range:.2f}")
    print(f"Variance: {variance:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    print(f"IQR (Interquartile Range): {IQR:.2f}")
    print(f"Median Absolute Deviation (MAD): {mad:.2f}")

    # Spread and Outlier Insights
    print("\n[INSIGHT - Spread & Outliers]:")
    print(f" -> The typical spread around the center is roughly ±{std:.2f} units (Standard Deviation).")
    print(f" -> The middle 50% of your patients/records span across a range of {IQR:.2f} units (from {Q1:.2f} to {Q3:.2f}).")
    
    # Simple Outlier Rule check (IQR method)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    if len(outliers) > 0:
        print(f" -> DETECTION: Found {len(outliers)} potential mathematical outlier(s) in '{col}' (values outside {lower_bound:.2f} to {upper_bound:.2f}).")
    else:
        print(f" -> No severe statistical outliers detected using the standard IQR rule.")
        
    # Domain specific data validation warning (e.g., zeroes in things that can't be zero)
    if col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        zero_count = (data == 0).sum()
        if zero_count > 0:
            print(f" -> DATA QUALITY WARNING: '{col}' contains {zero_count} zero value(s). Biologically, a value of 0 is impossible here and likely indicates missing or unrecorded data.")

# ---------------------------------------
# VISUALIZATIONS & GRAPHICAL INSIGHTS
# ---------------------------------------
print("\n========== VISUAL DIAGNOSTIC INSIGHTS ==========\n")

# 1. Boxplots & Insights
print("\n--- Generating Boxplots ---")
for col in num_df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=num_df[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()
    
    # Inline Insight
    print(f"[Boxplot Insight for {col}]: Look at the whiskers. Points plotted individually beyond the whiskers are extreme anomalies/outliers. The middle line in the box represents the true median of your data.")

# 2. Histograms & Density Plots (Combined for efficiency and insight)
print("\n--- Generating Histograms & Density Plots ---")
for col in num_df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(num_df[col], bins=20, kde=True, color='steelblue')
    plt.title(f'Distribution Plot of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
    
    # Inline Insight using Skewness calculation
    sk = skew(num_df[col].dropna())
    print(f"[Distribution Insight for {col}]:")
    if abs(sk) < 0.5:
        print(f" -> The chart shows a standard bell-like shape. It appears fairly Normal/Symmetric (Skewness: {sk:.2f}).")
    elif sk >= 0.5:
        print(f" -> The chart shows a long tail stretching to the right. It is Right-Skewed (Skewness: {sk:.2f}). Most records cluster at lower levels.")
    else:
        print(f" -> The chart shows a long tail stretching to the left. It is Left-Skewed (Skewness: {sk:.2f}). Most records cluster at higher levels.")

# 3. Correlation Heatmap & Deep Insights
print("\n--- Generating Correlation Heatmap ---")
plt.figure(figsize=(10,8))
corr_matrix = num_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap - Diabetes Dataset")
plt.show()

print("[Correlation Heatmap Insights]:")
# Extract strongest correlations programmatically
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
# Remove self-correlations (1.0)
corr_pairs = corr_pairs[corr_pairs < 1.0]

if len(corr_pairs) > 0:
    strongest_feature = corr_pairs.index[0]
    weakest_feature = corr_pairs.index[-1]
    
    print(f" -> Strongest Positive Relationship: Between {strongest_feature[0]} and {strongest_feature[1]} (r = {corr_pairs.iloc[0]:.2f}). As one increases, the other tends to rise.")
    print(f" -> Strongest Negative/Inverse Relationship: Between {weakest_feature[0]} and {weakest_feature[1]} (r = {corr_pairs.iloc[-1]:.2f}).")
    
    # Medical Outcome check
    if 'Outcome' in corr_matrix.columns:
        outcome_corr = corr_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)
        print(f" -> Impact on Diabetes (Outcome): The feature most highly correlated with a positive diabetes outcome is '{outcome_corr.index[0]}' (r = {outcome_corr.iloc[0]:.2f}).")
else:
    print(" -> Not enough diverse feature patterns to calculate a comparative correlation breakdown.")

# 4. Scatter Plot Matrix (Pair Plot)
if len(num_df.columns) >= 2:
    print("\n--- Generating Pair Plot Matrix ---")
    sns.pairplot(num_df)
    plt.show()
    print("[Pair Plot Insight]: Use this matrix matrix to search visually for linear cluster trends or separations. If you look at the rows/columns matching 'Outcome', look for gaps where healthy vs. diabetic cases separate cleanly.")

# 5. Q-Q Plots (Normality Check)
print("\n--- Generating Q-Q Plots ---")
for col in num_df.columns:
    plt.figure(figsize=(6,4))
    probplot(num_df[col].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    plt.show()
    
    print(f"[Q-Q Plot Insight for {col}]: If the red data points track closely along the straight diagonal line, the data is normally distributed. Deviations, curves, or S-shapes at the ends signal that the data violates standard statistical normality rules.")

print("\n========== Automated Analysis Completed Successfully! ==========")


# In[ ]:





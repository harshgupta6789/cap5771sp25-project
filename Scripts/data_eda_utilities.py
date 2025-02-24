import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_line_plot(df, title, xlabel, ylabel, xticks, xtickslabels, figsize=(10,6)):
    plt.figure(figsize=figsize)
    df.plot(kind='line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks, xtickslabels)
    plt.grid(True)
    plt.show()

def create_bar_chart(df, title, xlabel, ylabel, color='skyblue', figsize=(10,6)):
    plt.figure(figsize=figsize)
    df.plot(kind='bar', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

def create_histogram(df, title, columns, bins=30, figsize=(12, 6)):
    df[columns].hist(figsize=figsize, bins=bins)
    plt.suptitle(title, fontsize=16)
    plt.show()

def create_boxplot(df, title, columns, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    sns.boxplot(data=df[columns])
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.show()

def draw_correlation_heatmap(df):
    corr_matrix = df.corr(numeric_only=True)

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

# Skewed Distributions
def draw_skewness_histogram(df, selected_cols):
    skewness = df[selected_cols].skew()
    print("Skewness:\n", skewness)

    # Plot histograms for selected columns
    df[selected_cols].hist(figsize=(16, 9), bins=30)
    plt.suptitle("Histograms of Selected Features", fontsize=16)
    plt.show()

def print_outliers(df, num_cols):
    for col in num_cols:
        outliers = detect_outliers_iqr(df, col).sum()
        percentage = round((outliers/len(df))*100, 2)
        print(f"{col}: {outliers} outliers detected ({percentage}%)")

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)
  
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df.reset_index(drop=True)

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

def scale_df(df, selected_columns, type='scalar'):
    if type == 'scalar':
        scaler = StandardScaler()
        return scaler.fit_transform(df[selected_columns])
    else:
        print('Invalid type provided')
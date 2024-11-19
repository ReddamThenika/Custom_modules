
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(df):
    df_cleaned = df.dropna()
    return df_cleaned

def fill_missing_values(df, method='mean'):
    if method == 'mean':
        return df.fillna(df.mean())
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Unsupported fill method")

def plot_histogram(df, column, bins=10):
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=bins, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_bar(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df, hue=column, palette="Set2", dodge=False, legend=False)
    plt.title(f"Bar Plot of {column}")
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def plot_box(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column], color='lightgreen')
    plt.title(f"Box Plot of {column}")
    plt.show()

def plot_scatter(df, x_column, y_column):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_column, y=y_column, data=df, color='orange')
    plt.title(f"Scatter Plot of {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def detect_outliers(df, column, threshold=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def eda_summary(df):
    print("Descriptive Statistics of the DataFrame:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)

def load_and_eda(file_path, hist_column=None, bar_column=None, box_column=None, scatter_x_column=None, scatter_y_column=None):
    df = pd.read_csv(file_path)
    df_cleaned = clean_data(df)
    eda_summary(df_cleaned)
    
    if hist_column:
        plot_histogram(df_cleaned, column=hist_column)
    if bar_column:
        plot_bar(df_cleaned, column=bar_column)
    if box_column:
        plot_box(df_cleaned, column=box_column)
    if scatter_x_column and scatter_y_column:
        plot_scatter(df_cleaned, x_column=scatter_x_column, y_column=scatter_y_column)
    
    plot_heatmap(df_cleaned)
    
    return df_cleaned
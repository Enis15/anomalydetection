import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Function definitions
'''

def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def summary_statistics(df):
    print('General Summary:')
    print(df.info())
    print('Summary Statistics:')
    print(df.describe())

def missing_values(df):
    print('Missing Values:')
    print(df.isnull().sum())

def distrib_plot(df):
    for col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, stat='count')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.title(f'Distribution of {col}')
        plt.show()

def distrib_plots_time(df):
    columns_num = df.shape[1]
    plots_per_row = 4
    plot_rows = (columns_num + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(plot_rows, plots_per_row, figsize=(7 * plot_rows , 7 * plot_rows))

    axes = axes.flatten() # Flatten the axes

    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.hist(df[col], bins=20, edgecolor='blue', alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def correlation_plot(df, dataset_name):
    if dataset_name == 'dataset2':
        df = df.drop({'nameOrig', 'nameDest'}, axis=1)
        df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

    if dataset_name == 'dataset3':
        # Replacing categorical values with dummy values
        cat_features = df.select_dtypes(include=['object']).columns
        for col in cat_features:
            df[col] = df[col].astype('category')
        # Categorical features to numerical features
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

    if dataset_name == 'dataset4':
        df = df.drop(['timestamp', 'sending_address', 'receiving_address'], axis=1)

    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap='coolwarm')
    plt.title(f'Correlation Matrix')
    plt.show()

if __name__ == '__main__':
    # Define the dataset names and paths
    datasets = {
        'dataset1': '../data/datasets/Labeled_DS/creditcard.csv',
        'dataset2': '../data/datasets/Labeled_DS/Fraud.csv',
        'dataset3': '../data/datasets/Labeled_DS/fin_paysys/finpays.csv',
        'dataset4': '../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv'
    }

    dataset_name = 'dataset4' # Adjust to your desired dataset from the dict
    file_path = datasets[dataset_name]

    df = load_data(file_path)
    summary_statistics(df)
    missing_values(df)
    distrib_plots_time(df)
    #distrib_plot(df, ['col name'])
    correlation_plot(df, dataset_name)

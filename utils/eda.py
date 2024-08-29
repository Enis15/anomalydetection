import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Function definitions
'''

def load_data(file_path):
    """
    Path to csv file
    Parameters:
        file_path (string): path to csv file
    Returns:
        dataset: Pandas Dataframe
    """
    dataset = pd.read_csv(file_path)
    return dataset

def summary_statistics(df):
    """
    General summary statistics
    Parameters:
        df (pandas Dataframe): Pandas Dataframe
    Returns:
        Summary statistics for the dataframe
    """
    print('General Summary:')
    print(df.info())
    print('Summary Statistics:')
    pd.set_option('display.max_columns', None)
    print(df.describe().apply(lambda x: x.apply('{0:.4f}'.format)))

def missing_values(df):
    """
    List of missing values for each feature
    Parameters:
        df (pandas Dataframe): Pandas Dataframe
    Returns:
        Missing values for the dataframe
    """
    print('Missing Values:')
    print(df.isnull().sum())

def distrib_plot(df, dataset_name, col_name=None):
    """
    Generate the distribution plots
    Parameters:
        df: Dataframe
        col_name: str, default is None to display distribution plots for all features, else specify the col_name
        dataset_name: str,
    Returns:
        Distribution plots
    """
    # If no specific name is provided, plot distribution for all columns
    if col_name is None:
        for col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], stat='count')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.title(f'Distribution of {col}')
            plt.grid(True)
            plt.savefig(f'{dataset_name}_{col_name}.png')
            plt.show()

    # If a specific feature name is provided
    elif col_name in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col_name], stat='count')
        plt.xlabel(col_name)
        plt.ylabel('Density')
        plt.title(f'Distribution of {col_name}')
        plt.grid(True)
        plt.savefig(f'{dataset_name}_{col_name}.png')
        plt.show()
    else:
        print(f'Feature {col_name} not found in the dataset')

def distrib_plots_time(df, dataset_name):
    """
    Generate the distribution plots for each feature on the same figure
    Parameters:
        df (pandas Dataframe): Pandas Dataframe
        dataset_name: str, name of the dataset
    Returns:
        Distribution plots for each feature on the same figure.
    """
    # Display multiple plots in one picture
    columns_num = df.shape[1] # No. of columns in the dataframe
    plots_per_row = 4 # No. of plots per rows
    plot_rows = (columns_num + plots_per_row - 1) // plots_per_row # Determine the no. of rows requires for all plots

    # Create a grid of subplots to fit all plots
    fig, axes = plt.subplots(plot_rows, plots_per_row, figsize=(7 * plot_rows , 7 * plot_rows))

    axes = axes.flatten() # Flatten the axes for easier indexing

    # Loop through each feature
    for i, col in enumerate(df.columns):
        ax = axes[i]

        # Create a histogram for feature distributions
        ax.hist(df[col], bins=20, edgecolor='blue', alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')

    # Turn of any remaining unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_distrib.png')
    plt.show()

def correlation_plot(df, dataset_name):
    """
    Generate correlation plot
    Parameters:
        df (pandas Dataframe): Pandas Dataframe
        dataset_name: str, name of the dataset. Based on the name specific preprocessing steps are used.
    Returns:
        Displays the correlation plot and saves it as a png file.
    """
    # Preprocessing for dataset1
    if dataset_name == 'dataset1':
        # Drop irrelavant features
        df = df.drop(['Unnamed: 0', 'trans_date_trans_time', 'trans_num', 'unix_time', 'dob', 'first', 'last', 'merch_zipcode'], axis=1)
        # Encoding categorical features with numerical variables
        cat_features = df.select_dtypes(include=['object']).columns
        for col in cat_features:
            df[col] = df[col].astype('category')
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

    # Preprocessing for dataset2
    if dataset_name == 'dataset2':
        # Drop specific features that aren't needed for correlation analysis
        df = df.drop(['nameOrig', 'nameDest'], axis=1)
        # Map feature 'type' to numerical values
        df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

    # Preprocessing for dataset3
    if dataset_name == 'dataset3':
        # Drop specific features that aren't needed for correlation analysis
        df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)
        # Identify categorical values
        cat_features = df.select_dtypes(include=['object']).columns
        # Convert categorical features to numerical values
        for col in cat_features:
            df[col] = df[col].astype('category') # Convert to category type
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

    # Preprocessing for dataset4
    if dataset_name == 'dataset4':
        # Identify categorical values
        cat_features = df.select_dtypes(include=['object']).columns
        # Convert categorical features to numerical values
        for col in cat_features:
            df[col] = df[col].astype('category') # Convert to category type
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

    # Calculate and generate the correlation matrix
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap='coolwarm')
    plt.title(f'Correlation Matrix')
    plt.savefig(f'{dataset_name}_corr.png')
    plt.show()

if __name__ == '__main__':
    # Define the dataset names and paths
    datasets = {
        'dataset1': '../data/datasets/Labeled_DS/creditcard.csv',
        'dataset2': '../data/datasets/Labeled_DS/Fraud.csv',
        'dataset3': '../data/datasets/Labeled_DS/fin_paysys/finpays.csv',
        'dataset4': '../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv'
    }

    dataset_name = 'dataset1' # Adjust to your desired dataset from the dict
    file_path = datasets[dataset_name]

    df = load_data(file_path)
    summary_statistics(df)
    missing_values(df)
    #distrib_plots_time(df, dataset_name) # Displays all plots on the same figure
    #distrib_plot(df, dataset_name, 'Class')
    #correlation_plot(df, dataset_name)

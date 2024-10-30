import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
Defining functions for exploratory data analysis
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
            plt.savefig(f'../results/eda_distributions/{dataset_name}_{col_name}.png')
            plt.show()

    # If a specific feature name is provided
    elif col_name in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col_name], stat='count')
        plt.xlabel(col_name)
        plt.ylabel('Density')
        plt.title(f'Distribution of {col_name}')
        plt.grid(True)
        plt.savefig(f'../results/eda_distributions/{dataset_name}_{col_name}.png')
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
    plt.savefig(f'../results/eda_distributions/{dataset_name}_distrib.png')
    plt.show()


def dataset1_preprocessing(df, dataset_name):
    if dataset_name == 'dataset1':
        # Drop irrelevant features
        df = df.drop(['Unnamed: 0', 'trans_date_trans_time', 'trans_num', 'unix_time', 'dob', 'first', 'last', 'merch_zipcode'], axis=1)
        # Encoding categorical features with numerical variables
        cat_features = df.select_dtypes(include=['object']).columns
        for col in cat_features:
            df[col] = df[col].astype('category')
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)
        return df
    else:
        print('Invalid dataset!')

def dataset2_preprocessing(df, dataset_name):
    if dataset_name == 'dataset2':
        # Drop specific features that aren't needed for correlation analysis
        df = df.drop(['nameOrig', 'nameDest'], axis=1)
        # Map feature 'type' to numerical values
        df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})
        cat_features = df.select_dtypes(include=['object']).columns
        for col in cat_features:
            df[col] = df[col].astype('category')
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)
        return df
    else:
        print('Invalid dataset')

def dataset3_preprocessing(df, dataset_name):
    if dataset_name == 'dataset3':
        # Drop specific features that aren't needed for correlation analysis
        #df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)
        # Identify categorical values
        cat_features = df.select_dtypes(include=['object']).columns
        # Convert categorical features to numerical values
        for col in cat_features:
            df[col] = df[col].astype('category') # Convert to category type
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)
        return df
    else:
        print('Invalid dataset')

def dataset4_preprocessing(df, dataset_name):
    if dataset_name == 'dataset4':
        # Dropping irrelevant columns for the anomaly detection
        df = df.drop(['timestamp', 'sending_address', 'receiving_address'], axis=1)
        pd.set_option('future.no_silent_downcasting', True)  # Ensure downcasting behavior is consistent with future versions of pandas
        df['anomaly'] = df['anomaly'].replace({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})
        df['anomaly'] = df['anomaly'].astype(int)
        # Identify categorical values
        cat_features = df.select_dtypes(include=['object']).columns
        # Convert categorical features to numerical values
        for col in cat_features:
            df[col] = df[col].astype('category') # Convert to category type
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)
        return df
    else:
        print('Invalid dataset!')

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
        data = dataset1_preprocessing(df, dataset_name)

    # Preprocessing for dataset2
    if dataset_name == 'dataset2':
        data = dataset2_preprocessing(df, dataset_name)

    # Preprocessing for dataset3
    if dataset_name == 'dataset3':
        data = dataset3_preprocessing(df, dataset_name)

    # Preprocessing for dataset4
    if dataset_name == 'dataset4':
        data = dataset4_preprocessing(df, dataset_name)

    # Calculate and generate the correlation matrix
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, annot=True, fmt='.2f', cmap='viridis', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for {dataset_name}')
    plt.savefig(f'../results/eda_plots/{dataset_name}_corr.png')
    plt.show()

def anomaly_vis(df, dataset_name):
    """
    Generate visualization for anomaly points
    Parameters:
        df (pandas Dataframe): Pandas Dataframe
        dataset_name: str, name of the dataset. Based on the name specific preprocessing steps are used.
    Returns:
        Scatter plot of normal and anomaly points
    """
    # Preprocessing steps
    if dataset_name == 'dataset1':
        data = dataset1_preprocessing(df, dataset_name)
        features = data.drop(columns=['is_fraud'])
        features = StandardScaler().fit_transform(features) # Standardize the data for better visualization
        anomaly = 'is_fraud' # Used for plotting to set the hue

    if dataset_name == 'dataset2':
        '''
        Due to computation complexity, stratified sampling can be used to improve runtime
        '''
        data = dataset2_preprocessing(df, dataset_name)
        data_sampled, _ = train_test_split(data, test_size=0.8, stratify=data['isFraud'], random_state=42)
        # data = data_sampled  # Used
        features = data.drop(columns=['isFraud'])
        anomaly = 'isFraud'

    if dataset_name == 'dataset3':
        data = dataset3_preprocessing(df, dataset_name)
        features = data.drop(columns=['fraud'])
        anomaly = 'fraud' # Used for plotting to set the hue

    if dataset_name == 'dataset4':
        data = dataset4_preprocessing(df, dataset_name)
        features = data.drop(columns=['anomaly'])
        anomaly ='anomaly' # Used for plotting to set the hue

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    data['TSNE1'] = tsne_results[:, 0]
    data['TSNE2'] = tsne_results[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue=anomaly, data=data, palette={0: 'blue', 1: 'red'})
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.title(f't-SNE Plot for {dataset_name}: Normal & Anomaly points')
    plt.savefig(f'../results/eda_plots/{dataset_name}_anomaly.png')
    plt.show()

# Define the dataset names and paths
datasets = {
    'dataset1': '../data/datasets/Labeled_DS/creditcard.csv',
    'dataset2': '../data/datasets/Labeled_DS/Fraud.csv',
    'dataset3': '../data/datasets/Labeled_DS/fin_paysys/finpays.csv',
    'dataset4': '../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv'
}

if __name__ == '__main__':

    dataset_name = 'dataset3' # Adjust to your desired dataset from the dict
    file_path = datasets[dataset_name]

    df = load_data(file_path)
    summary_statistics(df)
    missing_values(df)
    distrib_plots_time(df, dataset_name) # Displays all plots on the same figure
    distrib_plot(df, dataset_name, 'anomaly') # Specify the column/feature
    correlation_plot(df, dataset_name)
    anomaly_vis(df, dataset_name)
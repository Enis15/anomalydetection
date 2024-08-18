# ANOMALY DETECTION IN FINANCIAL SERVICES: A COMPARISON STUDY OF DIFFERENT METHODS USED FOR ANOMALY DETECTION

## Project Overview
This projects implements anomaly detection techniques on four different financial datasets using both supervised
and unsupervised learning models. The project is organized into different scripts for data analysis, hyperparameter
tuning, and model execution.

## Installation

### Prerequisites
The project runs on Python 3.10. You cna download it from the official Python webpage. (https://www.python.org/downloads/)

### Installing Dependencies
You can install the required dependencies using `pip`:
 pip install -r requirements.txt

### Dependencies
The project depends on the following libraries, as listed in the `requirements.txt` file:
    `matplotlib == 3.8.2`
    `pandas == 2.2.0`
    `scikit-learn == 1.4.0`
    `scipy == 1.12.0`
    `hyperopt == 0.2.7`
    `seaborn == 0.13.2`

## Usage

### Running the Project
1. Run the Utils scripts: Before running any model, make sure that all the scripts within the `utils/` folder are executed
successfully.
Example:
    python utils/eda.py #This script can be ignored if you do not want to perform exploratory data analysis
    python utils/logger.py
    python utils/parameter_tune.py
    python utils/supervised_learning.py
    python utils/unsupervised_learning.py
2. Run the Model Scripts: Once the scripts in the utils are executed, you can run the scripts in the `models/` folder.
Each scripts performs anomaly detection on one of the datasets.
3. Check the Results: Then the performance results, both graphically and as tables, are saved in the results folder with
a dedicated name for each dataset.

### Example Workflow
1. Run `eda.py` for exploratory data analysis (optional)
2. Configure `logger.py` the logger scripts
3. Run `parameter_tune.py` for hyperparameter tuning
4. Perform anomaly detection on dataset 1 with `ds1_creditcard.py`
5. Check `ROC_AUC_vs_Runtime(DS1).png` for the visual representation of the findings

## Scripts Overview

### Utils
`eda.py`: Conducts exploratory data analysis (EDA) on the datasets. The scripts is built on functions that take as input
the dataset name to return summary statistics, missing values, plots, etc.
`logger.py`: Configures logging to track the execution of scripts and model performance. # Script provided by Manuel Pangl
`parameter_tune.py`: Contains classes for hyperparameter tuning of the models. Helps in optimizing model performance.
`supervised_learning.py`: Defines the supervised learning models used in the project.
`unsupervised_learning.py`: Defines the unsupervised learning models used in the project.

### Models
`ds1_creditcard.py`: Performs anomaly detection with all defined models on dataset 1 and stores the results.
`ds2_fraud.py`: Performs anomaly detection with all defined models on dataset 2 and stores the results.
`ds3_finpays.py`: Performs anomaly detection with all defined models on dataset 3 and stores the results.
`ds4_metaverse.py`: Performs anomaly detection with all defined models on dataset 4 and stores the results.

### Data
The datasets are stores in the `data/datasets/Labeled_DS/` folder as CSV files.
`ds1_creditcard`: Refers to the dataset named `creditcard.csv`
`ds2_fraud`: Refers to the dataset named `Fraud.csv`
`ds3_finpays`: Refers to the dataset named 'bs140513_032310.csv`
`ds4_metaverse`: Refers tot he dataset named `metaverse_transactions_dataset.csv`

### Results
The performance results are stores in the `results/` folder. Each dataset has the results stored as q csv file and as plots.

Example:
Results of dataset 1
`Metrics(DS1).csv`: Stores the performance of dataset 1 on ROC AUC Score, F1 score and Runtime, across each model.
`ROC_AUC_vs_Runtime(DS1).png`: Presents a plot that shows the relationship between ROC AUC score and Runtime for different
models applied to dataset 1
`F1_score_vs_Runtime(DS1).png`: Presents a plot that shows the relationship between F1 score and Runtime for different
models applied to dataset 1

## Contributing
Special thank to Manuel Pangl (m.pangl.gmx.at) for providing the `logger.py` script.

## Contact
Enis Pinari
Email: enispinari1@gmail.com
Github: https://github.com/Enis15
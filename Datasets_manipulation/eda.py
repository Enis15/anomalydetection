import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_manipulation import data_preprocessing


df = pd.read_csv('../data/datasets/Unlabeled_DS/CreditCardTransaction.csv')

plt.boxplot(df['TrnxAmount'])
plt.show()

#df = data_preprocessing(df, 'Transaction Amount')



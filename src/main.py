import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)  # Change this to DEBUG for more information


# Load the data
logging.info('Loading data...')
logging.info('Loading test features...')
test_features = pandas.read_csv('data/test_set_features.csv')
logging.info('Loading training features...')
train_features = pandas.read_csv('data/training_set_features.csv')
logging.info('Loading training labels...')
train_labels = pandas.read_csv('data/training_set_labels.csv')
logging.info('Data loaded')


# Merge the training data
logging.info('Merging training data...')
train_data = train_features.merge(train_labels, on='respondent_id')
logging.info('Training data merged')


# DO NOT SHOW THIS PLOT. IT WILL BRING YOUR COMPUTER TO ITS KNEES
# Plot the data
# logging.info('Plotting data...')
# seaborn.pairplot(train_data, hue='h1n1_vaccine')
# plt.show()

# Data preprocessing

# Drop the respondent id
train_data = train_data.drop('respondent_id', axis=1)

# Check for missing values
missing_values = train_data.isnull().sum()
logging.info(f'Missing values found: {missing_values}')

# Check for duplicate values
duplicate_values = train_data.duplicated().sum()
logging.info(f'Duplicate values found: {duplicate_values}')


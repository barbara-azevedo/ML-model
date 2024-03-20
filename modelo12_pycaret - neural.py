from pycaret.classification import *
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset from excel sheet
dataset = pd.read_excel('dataset.xlsx')

# The dataset should already have clean data and only the useful data

# Initialize PyCaret
# Define target as column "Result" from data set
clf1 = setup(data=dataset, target='Result', preprocess=False)


# Create neural network model
nn_model = create_model('mlp')  # 'mlp' represents a neural network: perceptron multilayer

# Evaluate model performance
evaluate_model(nn_model)

# Elavuate model
prediction = predict_model(nn_model)


# Save false results in excel file for evaluation
prediction.to_excel('resultado_neural.xlsx', index=False)


from pycaret.classification import *
import pandas as pd

# Load dataset from excel sheet
dataset = pd.read_excel('dataset.xlsx')

# The dataset should already have clean data and only the useful data

# Initialize PyCaret
# Define target as column "Result" from data set
clf1 = setup(data=dataset, target='Result', preprocess=False)

# Separate dataset into trainng and testing - according to column "Treino-teste"
train_set = dataset[dataset['Treino-teste'] == 1]
test_set = dataset[dataset['Treino-teste'] == 0]

# Inicializar o PyCaret novamente para realizar a preparação do modelo
clf2 = setup(data=train_set, target='Result', preprocess=True)

# Check different model options
best_model = compare_models()

# Visualize each model performance by confusion matrix
plot_model(best_model, plot='confusion_matrix')

# Plot ROC
plot_model(best_model, plot='auc')

# Plot feature importance
plot_model(best_model, plot='feature')

# Summary of model comparation
plot_model(estimator = 'all', plot = 'summary')

# Evaluate model in test set
prediction = predict_model(best_model, data=test_set)

# Evaluate model performance
evaluate_model(best_model)

# Save trained model
save_model(best_model, 'modelo.h5')

# Finish PyCaret 
clf2 = None
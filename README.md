# Code for "Differentiation of white matter histopathology using b-tensor encoding and machine learning"
Code for the Machine learning pipeline used in "Differentiation of white matter histopathology using b-tensor encoding and machine learning" by
Ricardo Rios-Carrillo, Alonso Ramírez-Manzanares, Hiram Luna-Munguía, Mirelta Regalado, Luis Concha
 (bioRxiv 2023.02.17.529024; doi: https://doi.org/10.1101/2023.02.17.529024 )

Dependencies to run this project are on the requirements.yml file. You can use conda to create an enviroment to run the scripts (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Repository contents
Main scripts:
-rf_tunning_featureAnalysis.py : Load data.csv, wrangles the data, train a Random Forest model and save it on rf.sav. It also extracts relevants features on the random forest.
-model_predict.py: Use the model on rf.sav to classify the regional data on data.csv. Saves results on data_regional_classified.csv.
Both function plots some figures.

Auxiliary functions:
-rf_auxililary_functions.py
-model_predict_functions.py
-wrangle_data.py

Data:
- data.csv: Initial data.
- data_regional_classified.csv: Composed only of data of the regional nerves. Last solumns contian the results of classification.
- rf.sav: The random forest trained model.

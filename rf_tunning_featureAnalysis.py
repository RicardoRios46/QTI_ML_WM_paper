import pandas as pd
import numpy as np
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
# Need this line so text is text in svg export
plt.rcParams['svg.fonttype'] = 'none'

import wrangle_data as wd
import rf_auxililary_functions as rf_aux

# Utilities
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.inspection import permutation_importance

# GENERAL inputs
# ? input: general filename for all outputs
outputsFileName = 'paper_'
format_fig_outputs = 'png'

# ? input: threads to use in various process. -1=use all threads
n_jobs = -1

# LOAD right nerve data
# ? inputs: case, csv_filename
data_filename = 'data.csv'

# Features to train/test the models
features_list = ["FA","MD","ad","rd","uFA","C_c","MKi","MKad"]

# Filename to save the tunned RF model
modelFileName = 'rf.sav'



# LOAD/PREPARE DATA

df_right = wd.get_NerveData(data_filename, nerve='Right')

print("Subjects ID in the dataframe")
print(df_right.analysisID.unique())
print()






# RF TUNNING

# For reproducibility the seed is fixed. For a different experiemt change the seed and comment np.random.RandomState line
seed = 7
data_split_random_state = seed
np.random.RandomState(seed=seed)

# Extract data to train model 
train_classes = ["Intact","Injured","Injured_plus"]
df_right_train = df_right.query('Histology == @train_classes')
average_value = 'weighted'

# Extract features for sckit learn
X = df_right_train.loc[:,features_list] # features
# Extract labels for sckit learn
y, label = pd.factorize(df_right_train["Histology"], sort=True) # labels
# Split train data to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=data_split_random_state)


# Tunning a random forestwith grid search 
print("Tunning random forest")
rf_model = rf_aux.gridSearchRF(X_train, y_train, seed, n_jobs)

# Check the best hyperparameters obtianed in the grid search and check its score
print()
print("Optimized hyperparemeters:")
print(rf_model.best_params_)
print("Accuracy in traing data")
print(rf_model.best_score_)

# Test fitting wit thest set to validate score with unseen data
print("Accuaracy with test data")
print(rf_model.score(X_test, y_test))


# load best model in the gridSearch
rf = rf_model.best_estimator_

# save the model to disk
joblib.dump(rf, modelFileName)








# RF EVALUATION

# print metrics
print()
print("Metrics in accuracy data")
rf_aux.get_metrics(rf, X_test, y_test, average_value)
print()

# Print confusion matrix
metrics.ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test,cmap='PuBu')
plt.savefig(outputsFileName + 'confussionMatrix.' + format_fig_outputs)
#plt.show()

# Feature relevance Default Mean Decreace impurity
f_i = list(zip(features_list,rf.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.figure()
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.savefig(outputsFileName + 'featureRelevance_impurity.' + format_fig_outputs)
#plt.show()

# Feature relevance by permutation test
result_permutation_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=200, random_state=seed, n_jobs=n_jobs)
# convert to series to extract indexing order
feature_importante_permutation_series = pd.Series(result_permutation_importance.importances_mean, index=features_list)
# convert to dataframe to simple seaborn plots
feature_importante_permutation = pd.DataFrame(result_permutation_importance.importances.transpose(), columns = features_list)
# save fig
plt.figure()
sns.barplot(data=feature_importante_permutation,order=list(feature_importante_permutation_series.sort_values(ascending=False).index),palette='Blues_r')
plt.savefig(outputsFileName + 'featureRelevance_permutation.' + format_fig_outputs)
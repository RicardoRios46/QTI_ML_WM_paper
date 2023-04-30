import joblib

import wrangle_data as wd
import model_predict_functions as mpf

# inputs
features_list = ["FA","MD","ad","rd","uFA","C_c","MKi","MKad"]
data_filename = 'data.csv'
outputsFileName = 'paper_'
modelFileName = 'rf.sav'

# load RF model optimized with rf_tunning.py
rf = joblib.load(modelFileName)

# Load data to predict with the model (regional class)
df_regional, label = wd.get_predictData(data_filename)

# DATA preparation
# Extract features data to predict
Xp =df_regional.loc[:,features_list]

print("Subjects to classify:")
print(df_regional.analysisID.unique())

# Predict unseen data
print("Predict data")
ypA = mpf.model_predict(rf, Xp, label)
# Predict unseen data (probabilites)
yprobA = mpf.model_predictProb(rf, Xp)


# Add results back to a df
classColumnName = "Class"
probColumnName  = "Prob_"
df_regional = mpf.putResult2df(df_regional, ypA, classColumnName)
df_regional = mpf.putProbResult2df(df_regional, yprobA, probColumnName) 

# save fataframe of regional with new columns classified by the model
df_regional.to_csv("data_regional_classified.csv")

# save example  scatterpProblots results
main_feats = ["FA","ad"] # change to other two for different visualizations
scatterProbPlotsfileName = outputsFileName + 'classResults.png'
mpf.save_plot_predictions(df_regional, main_feats[0], main_feats[1], classColumnName, probColumnName, scatterProbPlotsfileName)
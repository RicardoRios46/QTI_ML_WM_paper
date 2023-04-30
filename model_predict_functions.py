import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def model_predict(model, X, label):
    '''Predict class for data'''
    y = model.predict(X)
    # Get resutls back to categories
    y = label[y]
    # ypA_rfrg results back to a data frame/series
    y = pd.Series(y)
    # Recover original indexes
    y.index = X.index

    return y


def model_predictProb(model, X):
    '''Predict probability for eash class (Intact, Injured, Injuredplus) for data'''
    # Predictions probabilites
    y = (model.predict_proba(X))
    # TODO: add labels to dataframe?   hint: df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])
    # back to dataframe
    y = pd.DataFrame(y, columns = ['Intact','Injured','Injured_plus'])
    # Recover original indexes
    y.index = X.index
    return y


def get_label(df):
    '''Extract labels for traiging data'''
    df3class = df.query('Histology == ["Intact","Injured","Injured_plus"]')
    # Extract labels to train data
    y, label = pd.factorize(df3class["Histology"], sort=True)
    return label

def putResult2df(df, y, columnName):
    '''Put obtianed class results  into the dataframe'''
    # Create new category columns to copy new results
    df[columnName] = df["Histology"]
    df.loc[y.index, columnName]  = y
    return df


def putProbResult2df(df, y, columnName):
    '''Put obtianed probability results into the dataframe'''
    column_list = ['Intact', 'Injured', 'Injured_plus']
    for i in range(3):
        columnNameIt = columnName + column_list[i]
        df[columnNameIt] = 0.0
        df.loc[y.index, columnNameIt]  = y.iloc[:,i]

    return df



def save_plot_predictions(df, x_data, y_data, classColumnName, probColumnName, outputFileName):
    '''Simple plot rseult of classified voxels'''
    df_Regional = df.query("Histology == 'Regional'")
    fig, axes = plt.subplots(2, 3, figsize=(45, 30))

    sns.scatterplot(ax=axes[0][0], data=df, x=x_data, y=y_data,
                    hue="Histology", style="experiment", alpha=.5, legend=True)
    sns.scatterplot(ax=axes[0][1], data=df_Regional, x=x_data, y=y_data,
                    hue=classColumnName, style="experiment", alpha=.8, legend=True)

    sns.scatterplot(ax=axes[1][0], data=df_Regional, x=x_data, y=y_data,
                    hue=probColumnName + "Intact", hue_norm=(0,1), style="experiment", alpha=.8, legend=True)

    sns.scatterplot(ax=axes[1][1], data=df_Regional, x=x_data, y=y_data,
                    hue=probColumnName + "Injured", hue_norm=(0,1), style="experiment", alpha=.8, legend=True)

    sns.scatterplot(ax=axes[1][2], data=df_Regional, x=x_data, y=y_data,
                    hue=probColumnName + "Injured_plus", hue_norm=(0,1), style="experiment", alpha=.8, legend=True)

    axes[0][0].set_title("Original")
    axes[0][1].set_title("Class classification")
    axes[1][0].set_title("Intact Probability")
    axes[1][1].set_title("Injured Probability")
    axes[1][2].set_title("Injured_plus Probability")

    fig.savefig(outputFileName)




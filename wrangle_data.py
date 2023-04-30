import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

from pandas.api.types import CategoricalDtype


# Functions to wrangle data

def wrangle_Cvalues(dataframe, lowBound = 0, upBound = 1):
    ''' Erase data with not valid C values '''
    dataframe = dataframe.query('C_MD > @lowBound and C_M > @lowBound and C_mu > @lowBound and C_c > @lowBound')
    dataframe = dataframe.query('C_MD < @upBound and C_M < @upBound and C_mu < @upBound and C_c < @upBound')

    return dataframe

def wrangle_MKvalues(dataframe, upBound = 5):
    ''' Erase data with not valid kurtosis values'''
    dataframe = dataframe.query('MKi < @upBound and MKad < @upBound')

    return dataframe



def convert2categorial(dataframe):
    ''' Transform string columns to categorical values'''
    dataframe.loc[:,["experiment"]] = dataframe.loc[:,["experiment"]].astype("category") 
    dataframe['analysisID'] = dataframe.analysisID.astype('category')

    voxel_nerve = CategoricalDtype(['Left', 'Right', 'Chiasm'])
    dataframe['voxel_nerve'] = dataframe['voxel_nerve'].astype(voxel_nerve)

    voxel_type = CategoricalDtype(['Intact', 'Experimental'])
    dataframe['voxel_type'] = dataframe['voxel_type'].astype(voxel_type)

    histology_DType = CategoricalDtype(['Intact', 'Regional', 'Injured', 'Injured_plus'],ordered=True)
    dataframe['Histology'] = dataframe['Histology'].astype(histology_DType)

    return dataframe



def wrangle_df(dataframe,lowCBound = 0,upCBound = 1,upKBound=5):
    '''Wrangle the data: Erase invalid metrics and NaNs'''

    # Erase invalid data for poor fitting (values outside theorical valid ranges)
    dataframe = wrangle_Cvalues(dataframe, lowCBound, upCBound)
    dataframe = wrangle_MKvalues(dataframe, upKBound)

    # Drop rows with NaN
    # In the data correspond to those with no histology data: C3, L10, IQ3 and IQ5
    dataframe.dropna()

    dataframe = convert2categorial(dataframe)


    return dataframe



# Functions to acquire subsets of initial dataset

def get_NerveData(dataFullPath, nerve='Full'):
    '''Extract some subset from the data by nerve'''
    "Nerve cases= Full, Left, Right, Both, Chiasm"
    # LOAD the data
    dfRaw = pd.read_csv(dataFullPath)
    # Wrangle the data
    df = wrangle_df(dfRaw)

    
    # get onle solicited nerve data
    if nerve=='Full':
        return df
    elif nerve=='Both':
        return df.query("voxel_nerve=='Left' or voxel_nerve=='Right'") 
    elif nerve=='Chiasm':
        return df.query("voxel_nerve=='Chiasm'") 
    else: #Specific left or right nerve
        return df.query("voxel_nerve==@nerve") 



def get_predictData(dataFullPath):
    '''Get data to predict from dataframe (Regional data)'''
    # LOAD the data
    dfRaw = pd.read_csv(dataFullPath)
    # Wrangle the data
    df = wrangle_df(dfRaw)


    # It has all 4 classes
    df3class = df.query('Histology == ["Intact","Injured","Injured_plus"]') # To extract predicion labels
    # Extract labels to predict data
    y, label = pd.factorize(df3class["Histology"], sort=True)



    # Extract only Regional data. Need this to inlcude also left nerve
    regional_ID = df.query('Histology == ["Regional"]').analysisID.unique()
    df = df.query("analysisID in @regional_ID")

    return df, label




# Auxiliary functions

def get_categoriesInDf(df, column, category):
    ''' Get the unique categories of the same subject in specified column'''
    #return df.query("@column == @category").analysisID.unique()
    df_tmp = df.loc[ df[column]==category , :]
    categories = df_tmp.analysisID.unique()

    return categories



def get_means(dataframe):
    ''' GEt a dataframe with means gropued by experiment -> analysisID -> histology '''
    dfMeans = dataframe.groupby(['experiment','analysisID','Histology']).mean()
    pd.set_option("display.max_rows", 20, "display.max_columns", None)
    #dfMeans
    dfMeans = dfMeans.dropna() #the grouping creates empyt variables like isq in control. We drop all that
    
    dfMeans = dfMeans.drop(columns=['x', 'y', 'z']) # this info is not usefull

    # Convert dataframe indexes to antoher datafram to acces them in a column way
    multIndx_df=dfMeans.index.to_frame(index=False)

    return dfMeans





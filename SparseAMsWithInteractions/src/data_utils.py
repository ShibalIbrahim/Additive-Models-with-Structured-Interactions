import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def load_data(load_directory='./',
              filename='pdb2019trv3_us.csv',
              remove_margin_of_error_variables=False): 
    """Loads Census data, and retrieves covariates and responses.
    
    Args:
        load_directory: Data directory for loading Census file, str.
        filename: file to load, default is 'pdb2019trv3_us.csv'.
        remove_margin_of_error_variables: whether to remove margin of error variables, bool scaler.
        
    Returns:
        df_X, covariates, pandas dataframe.
        df_y, target response, pandas dataframe.
    """
    file = os.path.join(load_directory, filename)
    df = pd.read_csv(file, encoding = "ISO-8859-1")
    df = df.set_index('GIDTR')
    
    # Drop location variables
    drop_location_variables = ['State', 'State_name', 'County', 'County_name', 'Tract', 'Flag', 'AIAN_LAND']
    df = df.drop(drop_location_variables, axis=1)
    
    target_response = 'Self_Response_Rate_ACS_13_17'
    # Remove extra response variables 
    # Remove response columns 'FRST_FRMS_CEN_2010' (Number of addresses in a 2010 Census Mailout/Mailback area where the first form mailed was completed and returned) and 'RPLCMNT_FRMS_CEN_2010' (Number of addresses in a 2010 Census Mailout/Mailback area where the replacement form was completed and returned)

    extra_response_variables = [
        'Census_Mail_Returns_CEN_2010',
        'Mail_Return_Rate_CEN_2010',
        'pct_Census_Mail_Returns_CEN_2010',
        'Low_Response_Score',
        'Self_Response_Rate_ACSMOE_13_17',
        'BILQ_Frms_CEN_2010',
        'FRST_FRMS_CEN_2010',
        'RPLCMNT_FRMS_CEN_2010',
        'pct_FRST_FRMS_CEN_2010',
        'pct_RPLCMNT_FRMS_CEN_2010']
    df = df.drop(extra_response_variables, axis=1)
    
    if remove_margin_of_error_variables:
        df = df[np.array([c for c in df.columns if 'MOE' not in c])]

    # Change types of covariate columns with dollar signs in their values e.g. income, housing price  
    df[df.select_dtypes('object').columns] = df[df.select_dtypes('object').columns].replace('[\$,]', '', regex=True).astype(np.float64)

    # Remove entries with missing predictions
    df_full = df.copy()
    df = df.dropna(subset=[target_response])

    df_y = df[[target_response]]
    df_X = df.drop([target_response], axis=1)


    return df_X, df_y, df_full

def process_data(df_X,
                 df_y,
                 val_ratio=0.1, 
                 test_ratio=0.1, 
                 seed=None,
                 standardize_response=False):
    """Preprocesses covariates and response and generates training, validation and testing sets.
    
      Features are processed as follows:
      Missing values are imputed using the mean. After imputation, all features are standardized. 

      Responses are processed as follow:
      Either standardized or not depending on user choice selected by standardize_response.

    Args:
        val_ratio: Percentage of samples to be used for validation, float scalar.
        test_ratio: Percentage of samples to be used for testing, float scalar.
        seed: for reproducibility of results, int scalar.
        standardize_response: whether to standardize target response or not, bool scalar.
        
    Returns:
        X_train: Training processed covariates, float numpy array of shape (N, p).
        y_train: Training (processed) responses, float numpy array of shape (N, ).
        X_val: Validation processed covariates, float numpy array of shape (Nval, p).
        y_val: Validation (processed) responses, float numpy array of shape (N, ).
        X_test: Test processed covariates, float numpy array of shape (Ntest, p).
        y_test: Test (processed) responses, float numpy array of shape (N, ).
        x_preprocessor: processor for covariates, sklearn transformer.
        y_preprocessor: processor for responses, sklearn transformer.
    """        
#     house_values_variables = []
#     for i in df_X.columns:
#         if 'House_Value' in i:
#             house_values_variables.append(i)        
#     df_X[house_values_variables] = 100*(df_X[house_values_variables]/(df_X[house_values_variables].max(axis=0)))
    
#     income_variables = []
#     for i in df_X.columns:
#         if 'INC' in i or 'Inc' in i:
#             income_variables.append(i)        
#     df_X[income_variables] = 100*(df_X[income_variables]/(df_X[income_variables].max(axis=0)))
        
    N, p = df_X.shape
    df_X_temp, df_X_test, df_y_temp, df_y_test = train_test_split(df_X, df_y, test_size=int(test_ratio*N), random_state=seed)
    df_X_train, df_X_val, df_y_train, df_y_val = train_test_split(df_X_temp, df_y_temp, test_size=int(val_ratio*N), random_state=seed)
    
    print("Number of training samples:", df_X_train.shape[0])
    print("Number of validation samples:", df_X_val.shape[0])
    print("Number of test samples:", df_X_test.shape[0])
    print("Number of covariates:", p)
        
    ''' Processing Covariates '''    
    continuous_features = df_X.columns
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

    x_preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, continuous_features)])

    X_train = x_preprocessor.fit_transform(df_X_train)
    X_val = x_preprocessor.transform(df_X_val)
    X_test = x_preprocessor.transform(df_X_test)
    
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)    
    X_train = np.round(X_train, decimals=6)
    X_val = np.round(X_val, decimals=6)
    X_test = np.round(X_test, decimals=6)
    
    ''' Processing Target Responses '''
    if standardize_response:
        y_preprocessor = StandardScaler()
    else:
        def identity_func(x):
            return np.array(x)
        y_preprocessor = FunctionTransformer(lambda x: np.array(x)) # acts as identity

    y_train = y_preprocessor.fit_transform(df_y_train)
    y_val = y_preprocessor.transform(df_y_val)
    y_test = y_preprocessor.transform(df_y_test)
                
    return X_train, y_train, X_val, y_val, X_test, y_test, (x_preprocessor, x_scaler), y_preprocessor


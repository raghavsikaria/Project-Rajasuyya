##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 23-6-2020
# Project: Project-Rajasuyya
# Description: Contains all Data processing utilities 
# exposed as static functions
# Code Sources & References:
#   1) Neural Networks in Action for Time Series Forecasting
#   by Kriti Mahajan and can be found here:
#   https://colab.research.google.com/drive/1PYj_6Y8W275ficMkmZdjQBOY7P0T2B3g#scrollTo=OkAmaZYuP20w
##########################################################

# Library imports
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataCleaningAndProcessingUtils:
    """Contains all Data processing utilities exposed as static functions.
    
    Checkout Colab Notebook by Kriti Mahajan posted in the Description. Much of
    the code has been recycled here and exposed as re-usable functions.
    """

    @staticmethod
    def read_csv_data(path: str, index_column: str = None) -> 'DataFrame':
        """Reads a CSV file from the path given, converts & returns a DataFrame."""

        df = pd.read_csv(path,index_col = index_column,parse_dates=True)
        if 'Unnamed: 0' in df.columns.values:
            df.drop(['Unnamed: 0'],inplace=True,axis=1)
        return df

    @staticmethod
    def process_na_data_in_df(df: 'DataFrame') -> 'DataFrame':
        """Removes all weekend & 'NA/nan' entries from the given DataFrame & returns it."""

        # Remove Weekends: Sat & Sun
        df = df[(df.index.dayofweek != 5)&(df.index.dayofweek != 6)]

        # Removing all columns with NA values > 10% entries
        df = df.loc[:, df.isna().sum()/df.shape[0] <= 0.1]

        # Dropping rows with all NA values
        df = df.dropna(axis=0,how='all')

        return df

    @staticmethod
    def process_interpolation_in_df(df: 'DataFrame') -> 'DataFrame':
        """Interpolates the given DataFrame's columns for NA values & returns it."""

        df = df.astype(float).interpolate(method ='linear',axis = 0,limit=30,limit_direction ='forward')
        return df

    @staticmethod
    def save_df_in_csv(df: 'DataFrame', path: str, index_label: str= None) -> None:
        """Saves the given DataFrame in CSV Format."""

        df.to_csv(path,index_label=index_label)
    
    @staticmethod
    def generate_df_with_daily_pct_change(df: 'DataFrame') -> 'DataFrame':
        """Saves the given DataFrame in CSV Format."""

        df_columns = df.columns.values
        daily_pct_change_df = pd.DataFrame(index=df.index)

        # Generating new columns as percentage change
        # on a single day basis
        for col in df_columns:   
            daily_pct_change_df[f'{col} returns'] = df[col].pct_change(1)

        # Dropping first row as it contains all NAN values!
        daily_pct_change_df.drop(daily_pct_change_df.index[0],inplace=True)
        
        return daily_pct_change_df

    @staticmethod
    def add_data_lag_to_df(df: 'DataFrame', number_of_lags: int = 10) -> 'DataFrame':
        """Adds lagged features to the given DataFrame & returns it."""

        df_columns = df.columns.values

        for column in df_columns:
            for lag in range(1,number_of_lags+1):
                df[f'{column} Lag_{lag}'] = df[f'{column}'].shift(lag)
        
        # Drops all NA value rows created by lagging of features
        df = df.dropna()
        return df

    @staticmethod
    def train_val_test_split_for_time_series(df: 'Dataframe',test_percentage: float = 0.08, validation_percentage: 'Float, as a percentage of Test%' = 0.5) -> 'Dataframe: training_data, Dataframe:validation_data, Dataframe:testing_data':
        """Splits the given DataFrame into training, validation & test dataframes and return them."""

        # Creating Training Data Dataframe
        number_of_test_observations =  int(np.round(test_percentage*len(df)))
        training_data = df[:-number_of_test_observations]
        testing_and_validation_data = df[-number_of_test_observations:]

        # Creating Validation + Testing Data Dataframe
        number_of_validation_obs = int(np.round(validation_percentage*len(testing_and_validation_data)))
        validation_data = testing_and_validation_data[:-number_of_validation_obs]
        testing_data = testing_and_validation_data[-number_of_validation_obs:]

        return training_data, validation_data, testing_data

    @staticmethod
    def data_z_score_standardization(training_data: 'Dataframe', validation_data: 'Dataframe', testing_data: 'Dataframe') -> 'Dataframe: training_data, Dataframe:validation_data, Dataframe:testing_data':
        """Centers the given training_data, validation_data & testing_data based on Mean & Std from Training Data and return normalized dataframes."""

        # Initializing standard scaler
        normalizer = StandardScaler()
        normalized_training_data = normalizer.fit_transform(training_data.values)
        normalized_validation_data = normalizer.transform(validation_data.values) 
        normalized_testing_data = normalizer.transform(testing_data.values)

        return normalized_training_data, normalized_validation_data, normalized_testing_data

    @staticmethod
    def data_minmax_scaler(training_data: 'Dataframe', validation_data: 'Dataframe', testing_data: 'Dataframe', feature_range: 'tuple, min & max range' = (-1,1)) -> 'Dataframe: training_data, Dataframe:validation_data, Dataframe:testing_data':
        """Scales the given training_data, validation_data & testing_data to the given feature range based on Mean & Std from Training Data and return normalized dataframes."""

        # Initializing minmax scaler
        minmax_scaler = MinMaxScaler(feature_range=feature_range)
        minmax_training_data = minmax_scaler.fit_transform(training_data)
        minmax_validation_data = minmax_scaler.transform(validation_data)
        minmax_testing_data = minmax_scaler.transform(testing_data)

        return minmax_training_data, minmax_validation_data, minmax_testing_data

    @staticmethod
    def split_df(df: 'DataFrame', n_steps_in: 'Int, # of days from past', n_steps_out: 'Int, forecast period') -> 'np.array, np.array':
        """Splits a multivariate sequence into samples.
        
        Code SOURCE: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
        """
        X, y = list(), list()
        for i in range(len(df)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out-1
            if out_end_ix > len(df):    break
            seq_x, seq_y = df[i:end_ix, :-1], df[end_ix-1:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    
    @staticmethod
    def adjust_df_for_dependent_variable(df: 'DataFrame', dependent_variable: str, keep_only_lag_variables: bool = True) -> 'DataFrame':
        """Pushes the dependent variable to the Column position in DataFrame."""

        df_columns = df.columns.values
        df_columns_adjusted = [column for column in df_columns if column != dependent_variable]
        if keep_only_lag_variables:
            df_columns_adjusted = [column for column in df_columns_adjusted if 'Lag' in column]
        df_columns_adjusted += [dependent_variable]
        
        return df[df_columns_adjusted]
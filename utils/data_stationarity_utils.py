##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 23-6-2020
# Project: Project-Rajasuyya
# Description: Contains all Data Stationarity utilities 
# exposed as static functions
##########################################################

# Library imports
import pandas as pd
from statsmodels.tsa.stattools import adfuller as aft
from tqdm import tqdm

class DataStationarityUtils:
    """Contains all Data Stationarity utilities exposed as static functions."""
    
    @staticmethod
    def apply_aft_test_to_df_column(df_column: 'DataFrame column') -> dict:
        """Applies AFT Stationarity checking test to given DataFrame Column & returns calculated parameters as dict."""

        # Conduct AFT test on the given DF Column
        t_stat,p_val,no_of_lags,no_of_observation,crit_val,_ = aft(df_column, autolag='AIC')

        return {'T Stat':t_stat,'P value':p_val,'Number of Lags':no_of_lags,'Number of Observations':no_of_observation,'Critical value @ 1%':crit_val['1%'],'Critical value @ 5%':crit_val['5%'],'Critical value @ 10%':crit_val['10%']}

    @staticmethod
    def check_data_stationarity(df: 'DataFrame') -> dict:
        """Checks for Data Stationarity over entire given DataFrame and returns results as a dict."""

        return_dict = {'data_stationarity_information_df':None, 'number_of_non_stationary_columns': None, 'non_stationary_columns': None}
        
        df_columns = df.columns.values
        data_stationarity_information = {}

        for column in tqdm(df_columns):
            # Conducting AFT test for specific column
            response = DataStationarityUtils.apply_aft_test_to_df_column(df[column])
            # Storing AFT test results for the column
            data_stationarity_information[column] = response

        # Converting stored AFT results into DataFrame
        data_stationarity_information_df = pd.DataFrame.from_dict(data_stationarity_information, orient='index')
        data_stationarity_information_df.index.name = "Features"
        # Generating stationarity verdict
        data_stationarity_information_df['Stationarity Verdict'] = data_stationarity_information_df['P value'] < 0.05
        # Finding out number of non stationary features
        non_stationary_columns = data_stationarity_information_df[data_stationarity_information_df['Stationarity Verdict'] == False].index.values
        
        return_dict.update({'data_stationarity_information_df':data_stationarity_information_df,'number_of_non_stationary_columns':non_stationary_columns,'non_stationary_columns':len(non_stationary_columns)})

        return return_dict
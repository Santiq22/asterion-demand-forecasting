# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import is_holiday, time_series_file_path, TabularResMLP
from dataclasses import dataclass
from pandas import read_csv, to_datetime, Series
from numpy import array
from torch import nn, load, no_grad, serialization, tensor, float32, int32
from torch.utils.data import DataLoader
# =============================================================================================== #

""" The idea of this script is to load a dataset given SUBGROUP and STORE_ID for every store
and subgroup in the ids_test.csv and to forecast the TOTAL_SALES for corresponding both variables 
based on the input DATE. This will be done using a trained model. """

# ======================================== Main classes ========================================= #
@dataclass
class ForecasterConfig:
    # Path to the submission folder
    folder_path = '../../../data/submissions'
    
    def __init__(self):
        # Path to the output submission file
        self.submission_file_path = os.path.join(self.folder_path, 'submission_TabularResMLP_v2.csv')
    
class Forecaster:
    # This class forecast a model and saves the predictions in ForecasterConfig.submission_file_path
    def __init__(self, list_of_subgroups : list[int], list_of_stores : list[int], columns_to_load : list[int],
                 trained_model : nn.Module, target : str, periods_to_predict : int):
        # Set the output path to save the trained model
        self.forecaster_config = ForecasterConfig()
        
        # List of subgroups
        self.list_of_subgroups = list_of_subgroups
        
        # List of stores
        self.list_of_stores = list_of_stores
        
        # Columns to load from the DataFrame
        self.columns_to_load = columns_to_load
        
        # Trained model to generate predictions
        self.trained_model = trained_model
        
        # Target variable to predict
        self.target = target
        
        # Periods to predict
        self.periods_to_predict = periods_to_predict
        
    def initiate_forecaster(self):
        try:
            # Load the ids_test_standardized.csv data
            df_ids_test = read_csv(os.path.join(ForecasterConfig.folder_path, 'ids_test_standardized.csv'))
            
            logging.info("ids_test_standardized.csv correctly loaded")
            
            # New DataFrame to store the predictions with the STORE_SUBGROUP_DATE_ID key
            df_new = df_ids_test.copy()
            
            # Iterate through the ids DataFrame
            #for n, sub in enumerate(df_ids_test.loc[:, 'SUBGROUP'].value_counts().index):
            for n, sub in enumerate(self.list_of_subgroups):
                if sub != 'basketball':
                    for m, store in enumerate(df_ids_test.loc[df_ids_test.loc[:, 'SUBGROUP'] == sub, 'STORE_ID'].value_counts().index):
                        # Load the corresponding DataFrame based on SUBGROUP and STORE
                        df = read_csv(time_series_file_path(subgroup = sub, store = store), index_col = 'DATE', 
                                    parse_dates = True, usecols = self.columns_to_load)
                        
                        # Label encode the subgroup and store
                        df['subgroup'] = self.list_of_subgroups.index(sub)
                        df['store'] = self.list_of_stores.index(store)
                        
                        # Get the dates where to make the predictions
                        dates = to_datetime(df_ids_test.loc[(df_ids_test.loc[:, 'SUBGROUP'] == sub) & 
                                                            (df_ids_test.loc[:, 'STORE_ID'] == store), 'DATE'], 
                                            format = '%Y-%m-%d').sort_values(ascending = True)
                                                
                        # Take the last periods_to_predict to forecast
                        df = df.iloc[-self.periods_to_predict:]
                        
                        # -------------------------------- Calendar features -------------------------------- #
                        df['day_of_week'] = array([date.dayofweek for date in dates])
                        df['day_of_month'] = array([date.day for date in dates])
                        df['day_of_year'] = array([date.dayofyear for date in dates])
                        df['month'] = array([date.month for date in dates])
                        df['is_holiday'] = array([is_holiday(date) for date in dates])
                        # ----------------------------------------------------------------------------------- #
                        
                        # Numerical columns
                        num_cols = df.select_dtypes(exclude = int).drop(columns = [self.target]).columns.to_list()
                        
                        # Categorical columns
                        cat_cols = df.select_dtypes(include = int).columns.to_list()
                        
                        # Numeric features
                        x_num = tensor(df[num_cols].values.astype("float32"), dtype = float32)
                        
                        # Categorical features
                        x_cat = tensor(df[cat_cols].values.astype("int32"), dtype = int32)
                        
                        # Generate the predictions
                        with no_grad():  # disables gradient tracking, faster
                            total_sales = self.trained_model(x_num, x_cat)
                            
                        # Convert back to numpy
                        total_sales = total_sales.cpu().numpy()
                        
                        # Predict using the trained model
                        predictions = Series(data = {date : prediction for (date, prediction) in zip(dates, total_sales)},
                                            index = dates)
                            
                        # Store the predictions
                        for date in predictions.index:
                            df_new.loc[(df_new.loc[:, 'SUBGROUP'] == sub) & 
                                    (df_new.loc[:, 'STORE_ID'] == store) & 
                                    (df_new.loc[:, 'DATE'] == date.strftime('%Y-%m-%d')), 
                                    ['TOTAL_SALES']] = predictions[date]
                            
                        print(n, sub + ' - ', m, store + " prediction made")
                        
                else:
                    # Skip in case the subgroup is basketball to del with it later
                    pass
                
            # Store the predictions for basketball
            for m, store in enumerate(df_ids_test.loc[df_ids_test.loc[:, 'SUBGROUP'] == 'basketball', 'STORE_ID'].value_counts().index):                
                # Get the dates where to make the predictions
                dates = to_datetime(df_ids_test.loc[(df_ids_test.loc[:, 'SUBGROUP'] == 'basketball') & 
                                                    (df_ids_test.loc[:, 'STORE_ID'] == store), 'DATE'], 
                                    format = '%Y-%m-%d').sort_values(ascending = True)
                
                # Generate the predictions
                total_sales = []
                for date in dates:
                    tmp = df_new.loc[(df_new.loc[:, 'SUBGROUP'] != 'basketball') &
                                     (df_new.loc[:, 'STORE_ID'] == store) & 
                                     (df_new.loc[:, 'DATE'] == date.strftime('%Y-%m-%d')),
                                     ['DATE', 'TOTAL_SALES']].groupby(['DATE']).mean()
                    total_sales.append(tmp['TOTAL_SALES'].values)
                
                # Predict using a basic model
                predictions = Series(data = {date : prediction for (date, prediction) in zip(dates, total_sales)},
                                     index = dates)
                
                for date in predictions.index:
                    df_new.loc[(df_new.loc[:, 'SUBGROUP'] == 'basketball') & 
                               (df_new.loc[:, 'STORE_ID'] == store) & 
                               (df_new.loc[:, 'DATE'] == date.strftime('%Y-%m-%d')), 
                               ['TOTAL_SALES']] = predictions[date]
                    
                print('basketball - ', m, store + " prediction made")
                            
            # Drop non necessary columns
            df_new.drop(columns = ['SUBGROUP', 'STORE_ID', 'DATE'], inplace = True)
                            
            # Save the new DataFrame with the corresponding predictions
            df_new.to_csv(self.forecaster_config.submission_file_path, index = False)
            
            logging.info("Submission file correctly loaded")
            
            return self.forecaster_config.submission_file_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    # Load the trained model
    serialization.add_safe_globals([TabularResMLP])
    model = load("trained_models/TabularResMLP_v2.pth", weights_only = False)
    model.eval()
    
    list_of_subgroups = []
    with open('../../../data/final_datasets/time_series/reformated_subgroup_names.txt', 'r') as file:
        for line in file:
            list_of_subgroups.append(line[:-1])
            
    list_of_stores = []
    with open('../../../data/final_datasets/time_series/transactions_store_names.txt', 'r') as file:
        for line in file:
            list_of_stores.append(line[:-1])
            
    columns_to_load = ['DATE',
                       'day_of_week',
                       'day_of_month',
                       'day_of_year',
                       'month',
                       'is_holiday',
                       'daily_total_quantity',
                       'daily_dispersion_quantity',
                       'daily_average_price',
                       'daily_dispersion_price',
                       'daily_dispersion_total_sales',
                       'daily_difference_total_quantity',
                       'daily_difference_dispersion_quantity',
                       'daily_difference_average_price',
                       'daily_difference_dispersion_price',
                       'daily_difference_total_sales',
                       'daily_difference_dispersion_total_sales',
                       'daily_percentage_change_average_price',
                       'daily_percentage_change_total_quantity',
                       'daily_percentage_change_total_sales',
                       'elasticity',
                       'difference_elasticity',
                       'daily_total_sales']
    
    obj = Forecaster(list_of_subgroups = list_of_subgroups,
                     list_of_stores = list_of_stores,
                     columns_to_load = columns_to_load,
                     trained_model = model,
                     target = 'daily_total_sales',
                     periods_to_predict = 7)
    obj.initiate_forecaster()
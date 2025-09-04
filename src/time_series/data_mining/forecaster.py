# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from pandas import read_csv, to_datetime, Series
from darts.models import TFTModel, Prophet
from darts import TimeSeries
# =============================================================================================== #

""" The idea of this script is to load a trained model given SUBGROUP and STORE_ID for every store
and subgroup in the ids_test.csv and to forecast the TOTAL_SALES for corresponding both variables 
based on the input DATE. """

# ======================================== Main classes ========================================= #
@dataclass
class ForecasterConfig:
    # Path to the submission folder
    folder_path = '../../data/submissions'
    
    def __init__(self, file_name : str):
        # Path to the output submission file
        self.submission_file_path = os.path.join(self.folder_path, file_name)
    
class Forecaster:
    # This class forecast a model and saves the predictions in ForecasterConfig.submission_file_path
    def __init__(self, predictions_file_name : str, model_name : str, variables : list[str], target : str,
                 periods_to_predict : int):
        # Set the output path to save the trained model
        self.forecaster_config = ForecasterConfig(predictions_file_name)
        
        # Name of the model used to predict for each subgroup and store
        self.model_name = model_name
        
        # Variables to use as covariates
        self.variables = variables
        
        # Variable to forecast
        self.target = target
        
        # Periods of prediction
        self.periods_to_predict = periods_to_predict
        
    def initiate_forecaster(self):
        try:
            # Load the ids_test_standardized.csv data
            df_ids_test = read_csv(os.path.join(ForecasterConfig.folder_path, 'ids_test_standardized.csv'))
            
            logging.info("ids_test_standardized.csv correctly loaded")
            
            # New DataFrame to store the predictions with the STORE_SUBGROUP_DATE_ID key
            df_new = df_ids_test.copy()
            
            # Iterate through the ids DataFrame
            for n, sub in enumerate(df_ids_test.loc[:, 'SUBGROUP'].value_counts().index):
                for m, store in enumerate(df_ids_test.loc[df_ids_test.loc[:, 'SUBGROUP'] == sub, 'STORE_ID'].value_counts().index):
                    # Load the corresponding DataFrame based on SUBGROUP and STORE
                    df = read_csv('../../data/final_datasets/time_series/transformed_time_series/transformed_time_series_by_store/{sub_1}/{sub_2}_{store}_time_series.csv'\
                                  .format(sub_1 = sub, sub_2 = sub, store = store),
                                  index_col = 'DATE', parse_dates = True)
                    
                    # Get the dates where to make the predictions
                    dates = to_datetime(df_ids_test.loc[(df_ids_test.loc[:, 'SUBGROUP'] == sub) & 
                                                        (df_ids_test.loc[:, 'STORE_ID'] == store), 'DATE'], 
                                        format = '%Y-%m-%d').sort_values(ascending = True)
                    
                    # Load the corresponding trained model
                    model = Prophet.load('trained_models/{sub_1}/{sub_2}_{store}_{model}.pkl'.format(sub_1 = sub, sub_2 = sub, store = store, model = self.model_name))
                    
                    # Shift the DataFrame to define the future covariates
                    df_tmp = df.shift(self.periods_to_predict, freq = 'D')
                    
                    # Define the future covariates
                    future_cov = TimeSeries.from_dataframe(df_tmp.iloc[-self.periods_to_predict:], value_cols = self.variables)
                        
                    # Predict using the loaded model
                    total_sales = model.predict(self.periods_to_predict, future_covariates = future_cov)
                        
                    # Predict using the benchmark model
                    predictions = Series(data = {date : prediction for (date, prediction) in zip(dates, total_sales.all_values()[:,0,0])},
                                         index = dates)
                        
                    # Store the predictions
                    for date in predictions.index:
                        df_new.loc[(df_new.loc[:, 'SUBGROUP'] == sub) & 
                                   (df_new.loc[:, 'STORE_ID'] == store) & 
                                   (df_new.loc[:, 'DATE'] == date.strftime('%Y-%m-%d')), 
                                   ['TOTAL_SALES']] = predictions[date]
                        
                    print(n, sub + ' -', m, store + " prediction made")
                    
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
    submission_file = 'submission_FP.csv'
    model = 'FP'
    obj = Forecaster(predictions_file_name = submission_file,
                     model_name = model,
                     variables = ["elasticity",
                                  "daily_average_quantity",
                                  "daily_total_quantity",
                                  "daily_dispersion_quantity", 
                                  "daily_average_price", 
                                  "daily_dispersion_price", 
                                  "daily_dispersion_total_sales",
                                  "daily_percentage_change_total_sales",
                                  "daily_percentage_change_total_quantity",
                                  "daily_percentage_change_average_price"],
                     target = 'daily_total_sales',
                     periods_to_predict = 7)
    obj.initiate_forecaster()
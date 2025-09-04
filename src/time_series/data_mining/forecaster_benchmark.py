# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv, to_datetime, Series
# =============================================================================================== #

""" The idea of this script is to load a dataset given SUBGROUP and STORE_ID for every store
and subgroup in the ids_test.csv and to forecast the TOTAL_SALES for corresponding both variables 
based on the input DATE. This will be done using a benchmark model, where it is chosen to be the
mean value of TOTAL_SALES in the last week. """

# ======================================== Main classes ========================================= #
@dataclass
class ForecasterConfig:
    # Path to the submission folder
    folder_path = '../../data/submissions'
    
    def __init__(self):
        # Path to the output submission file
        self.submission_file_path = os.path.join(self.folder_path, 'submission_benchmark.csv')
    
class Forecaster:
    # This class forecast a model and saves the predictions in ForecasterConfig.submission_file_path
    def __init__(self):
        # Set the output path to save the trained model
        self.forecaster_config = ForecasterConfig()
        
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
                    
                    # Get the average value of TOTAL_SALES in the last week before the week of forecast
                    total_sales = [df.loc[df.index[-7:], ['daily_total_sales']].mean().iloc[0] for a in range(7)]
                    
                    # Predict using the benchmark model
                    predictions = Series(data = {date : prediction for (date, prediction) in zip(dates, total_sales)},
                                         index = dates)
                        
                    # Store the predictions
                    for date in predictions.index:
                        df_new.loc[(df_new.loc[:, 'SUBGROUP'] == sub) & 
                                   (df_new.loc[:, 'STORE_ID'] == store) & 
                                   (df_new.loc[:, 'DATE'] == date.strftime('%Y-%m-%d')), 
                                   ['TOTAL_SALES']] = predictions[date]
                        
                    print(n, sub + ' - ', m, store + " prediction made")
                            
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
    obj = Forecaster()
    obj.initiate_forecaster()
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import concat, read_csv, DataFrame
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
# =============================================================================================== #

# This class will provide the paths for the outputs to the data standardization process
# * TSS stands for Time Series Standardization
@dataclass
class DataTSSConfig:
    # Dataset folder path defined as a class attribute
    folder_path = '../../data/final_datasets/time_series/standardized_time_series'
    
    def __init__(self, file_name : str):
        # Dataset file path
        self.dataset_file_path = os.path.join(self.folder_path, file_name)
        
# Class to standardize the different time series
class DataTSS:
    def __init__(self, data_input_name : str, data_output_name : str):        
        # Path to the output dataset
        self.data_tss_config = DataTSSConfig(data_output_name)
        
        # Path to the initial dataset
        self.path = os.path.join('../../data/final_datasets/time_series', data_input_name)        
        
    # This function initiates the data standardization process over the data
    def initiate_tss(self):
        try:
            # Load the initial dataset
            df = read_csv(self.path)
            
            logging.info("Reading of data completed")
            
            # Separate the time series
            df_ts = df[['QUANTITY', 'PRICE', 'TOTAL_SALES']]
            
            # Columns of the corresponding time series
            ts_columns = df_ts.columns
            
            # Drop time series variables so then it is possible to concatenate
            df.drop(ts_columns, axis = 1, inplace = True)
            
            # Pipeline to apply to the time series variables
            ts_pipeline = Pipeline(steps = [("scaler", StandardScaler())])
            
            # Object to apply standardization on the time series variables
            preprocesor = ColumnTransformer([("ts_pipeline", ts_pipeline, ts_columns)])
            
            # Standardize the time series variables
            df_ts_arr = preprocesor.fit_transform(df_ts)
            
            logging.info("Time series correctly standardized")
            
            # Convert standardized data into a DataFrame
            df_standardized = DataFrame(df_ts_arr)
            
            # Set a dict of new names for the standardized data
            new_names = {col_old: col_new  for col_old, col_new in 
                         zip(df_standardized.columns, df_ts.columns)}
            
            # Rename the columns of the data
            df_standardized.rename(columns = new_names, inplace = True)
            
            # Concatenate the standardized and the cut dataset
            df_new = concat((df, df_standardized), axis = 1)
            
            # Save the data
            df_new.to_csv(self.data_tss_config.dataset_file_path, index=False)
            
            logging.info("Standardized dataset saved")
            
            return self.data_tss_config.dataset_file_path            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    list_of_subgroups = []
    with open('../../data/final_datasets/time_series/reformated_subgroup_names.txt', 'r') as file:
        for line in file:
            list_of_subgroups.append(line[:-1])
    
    for n, sub in enumerate(list_of_subgroups):
        obj = DataTSS(sub + '_time_series.csv', sub + '_standardized_time_series.csv')
        data = obj.initiate_tss()
        print(n, sub + " time series standardized")
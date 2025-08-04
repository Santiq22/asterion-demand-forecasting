# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import month_plot, quarter_plot, plot_acf, plot_pacf
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the outputs to the feature engineering process
@dataclass
class FeatureEngineeringConfig:
    # Path to the time series folder
    folder_path = '../../data/final_datasets/time_series/transformed_time_series'
    
    def __init__(self, file_name : str):        
        # Time series file path
        self.time_series_file_path = os.path.join(self.folder_path, file_name)
    
# This class will perform the feature engineering process on the different datasets    
class FeatureEngineering:
    def __init__(self, subgroup : str, data_input_name : str):
        # Subgroup name
        self.subgroup = subgroup
        
        # Path to the initial dataset
        self.path = os.path.join('../../data/final_datasets/time_series', data_input_name)
        
    def initiate_feature_engineering(self):
        try:
            # Read the dataset using 'DATE' as a datetime64 index variable
            df = read_csv(self.path, index_col = 'DATE', parse_dates = True)
            
            logging.info("Dataset correctly loaded")
            
            # New dataset to store the time series
            df_new = DataFrame()
            
            # ------------------------------ Daily average quantity ----------------------------- #
            df_new['daily_average_quantity'] = df['QUANTITY'].resample('1d').mean()
            logging.info("Daily average quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------- Daily total quantity ------------------------------ #
            df_new['daily_total_quantity'] = df['QUANTITY'].resample('1d').sum()
            logging.info("Daily total quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------------- Daily quantity dispersion ---------------------------- #
            df_new['daily_dispersion_quantity'] = df['QUANTITY'].resample('1d').std()
            logging.info("Daily dispersion quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------- Daily average price ------------------------------- #
            df_new['daily_average_price'] = df['PRICE'].resample('1d').mean()
            logging.info("Daily average price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------ Daily price dispersion ----------------------------- #
            df_new['daily_dispersion_price'] = df['PRICE'].resample('1d').std()
            logging.info("Daily dispersion price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ! NOTE THAT THIS IS THE VARIABLE TO PREDICT
            # --------------------------------- Daily total sales ------------------------------- #
            df_new['daily_total_sales'] = df['TOTAL_SALES'].resample('1d').sum()
            logging.info("Daily total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # -------------------------- Daily total sales dispersion --------------------------- #
            df_new['daily_dispersion_total_sales'] = df['TOTAL_SALES'].resample('1d').std()
            logging.info("Daily dispersion total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # --------------------------------- First difference -------------------------------- #
            # ------------------------- Daily difference average quantity ----------------------- #
            df_new['daily_difference_average_quantity'] = df_new['daily_average_quantity'] - \
                df_new['daily_average_quantity'].shift(periods = 1)
            logging.info("Daily difference average quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- Daily difference total quantity ------------------------- #
            df_new['daily_difference_total_quantity'] = df_new['daily_total_quantity'] - \
                df_new['daily_total_quantity'].shift(periods = 1)
            logging.info("Daily difference total quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------- Daily difference quantity dispersion ---------------------- #
            df_new['daily_difference_dispersion_quantity'] = df_new['daily_dispersion_quantity'] - \
                df_new['daily_dispersion_quantity'].shift(periods = 1)
            logging.info("Daily difference dispersion quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # -------------------------- Daily difference average price ------------------------- #
            df_new['daily_difference_average_price'] = df_new['daily_average_price'] - \
                df_new['daily_average_price'].shift(periods = 1)
            logging.info("Daily difference average price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- Daily difference price dispersion ----------------------- #
            df_new['daily_difference_dispersion_price'] = df_new['daily_dispersion_price'] - \
                df_new['daily_dispersion_price'].shift(periods = 1)
            logging.info("Daily difference dispersion price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------------- Daily difference total sales ------------------------- #
            df_new['daily_difference_total_sales'] = df_new['daily_total_sales'] - \
                df_new['daily_total_sales'].shift(periods = 1)
            logging.info("Daily difference total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # --------------------- Daily difference dispersion total sales --------------------- #
            df_new['daily_difference_dispersion_total_sales'] = df_new['daily_dispersion_total_sales'] - \
                df_new['daily_dispersion_total_sales'].shift(periods = 1)
            logging.info("Daily difference dispersion total sales computed")
            # ----------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------- #
            
            # -------------------------------- Second difference -------------------------------- #
            # ---------------------- Daily second difference average quantity ------------------- #
            df_new['daily_second_difference_average_quantity'] = df_new['daily_difference_average_quantity'] - \
                df_new['daily_difference_average_quantity'].shift(periods = 1)
            logging.info("Daily second difference average quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------- Daily second difference total quantity --------------------- #
            df_new['daily_second_difference_total_quantity'] = df_new['daily_difference_total_quantity'] - \
                df_new['daily_difference_total_quantity'].shift(periods = 1)
            logging.info("Daily second difference total quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------- Daily second difference quantity dispersion ------------------- #
            df_new['daily_second_difference_dispersion_quantity'] = df_new['daily_difference_dispersion_quantity'] - \
                df_new['daily_difference_dispersion_quantity'].shift(periods = 1)
            logging.info("Daily second difference dispersion quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------- Daily second difference average price ---------------------- #
            df_new['daily_second_difference_average_price'] = df_new['daily_difference_average_price'] - \
                df_new['daily_difference_average_price'].shift(periods = 1)
            logging.info("Daily second difference average price computed")
            # ----------------------------------------------------------------------------------- #
            
            # --------------------- Daily second difference price dispersion -------------------- #
            df_new['daily_second_difference_dispersion_price'] = df_new['daily_difference_dispersion_price'] - \
                df_new['daily_difference_dispersion_price'].shift(periods = 1)
            logging.info("Daily second difference dispersion price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------- Daily second difference total sales ----------------------- #
            df_new['daily_second_difference_total_sales'] = df_new['daily_difference_total_sales'] - \
                df_new['daily_difference_total_sales'].shift(periods = 1)
            logging.info("Daily second difference total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # ----------------- Daily second difference dispersion total sales ------------------ #
            df_new['daily_second_difference_dispersion_total_sales'] = df_new['daily_difference_dispersion_total_sales'] - \
                df_new['daily_difference_dispersion_total_sales'].shift(periods = 1)
            logging.info("Daily second difference dispersion total sales computed")
            # ----------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------- #
            
            # -------------------- Daily percentage change in average price --------------------- #
            df_new['daily_percentage_change_average_price'] = df_new['daily_average_price'].pct_change(periods = 1)*100.0
            logging.info('Daily percentage change average price computed')
            # ----------------------------------------------------------------------------------- #
            
            # -------------------- Daily percetage change in total quantity --------------------- #
            df_new['daily_percentage_change_total_quantity'] = df_new['daily_total_quantity'].pct_change(periods = 1)*100.0
            logging.info('Daily percentage change in total quantity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------------------- Elasticity ------------------------------------ #
            df_new['elasticity'] = df_new['daily_percentage_change_total_quantity']/df_new['daily_percentage_change_average_price']
            logging.info('Elasticity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- First difference of elasticity -------------------------- #
            df_new['difference_elasticity'] = df_new['elasticity'] - df_new['elasticity'].shift(periods = 1)
            logging.info('First difference of elasticity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- Second difference of elasticity ------------------------- #
            df_new['second_difference_elasticity'] = df_new['difference_elasticity'] - \
                df_new['difference_elasticity'].shift(periods = 1)
            logging.info('Second difference of elasticity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------------- Cumulative total sales ------------------------------ #
            df_new['cumulative_total_sales'] = df_new['daily_total_sales'].cumsum()
            logging.info('Cumulative total sales computed')
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------------- Cumulative total quantity ---------------------------- #
            df_new['cumulative_total_quantity'] = df_new['daily_total_quantity'].cumsum()
            logging.info('Cumulative total quantity computed')
            # ----------------------------------------------------------------------------------- #
            
            # Drop NaN values created during shifting
            df_new.dropna(inplace = True)
            
            # Set the index column of the new DataFrame
            df_new.set_index(df_new['daily_average_quantity'].index, inplace = True)
            
            # Path to save the output
            feature_engineering_config = FeatureEngineeringConfig(self.subgroup + '_time_series.csv')
            
            # Save the final DataFrame
            df_new.to_csv(feature_engineering_config.time_series_file_path)
            
            return feature_engineering_config.time_series_file_path
            
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    list_of_subgroups = []
    with open('../../data/final_datasets/time_series/reformated_subgroup_names.txt', 'r') as file:
        for line in file:
            list_of_subgroups.append(line[:-1])
    
    for n, sub in enumerate(list_of_subgroups):
        obj = FeatureEngineering(subgroup = sub,
                                 data_input_name = sub + '_time_series.csv')
        data = obj.initiate_feature_engineering()
        print(n, sub + " plots made")
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import is_holiday
from dataclasses import dataclass
from pandas import read_csv, DataFrame
from numpy import inf, nan, array
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the outputs to the feature engineering process
@dataclass
class FeatureEngineeringConfig:
    # Path to the time series folder
    folder_path = '../../../data/final_datasets/time_series/transformed_time_series'
    
    def __init__(self, subgroup : str, store : str | None):
        if store != None:
            # Creates the directory where the time series will be stored given the subgroup
            os.makedirs(os.path.join(self.folder_path, subgroup), exist_ok = True)
                
            # Time series file path
            self.time_series_file_path = os.path.join(self.folder_path, 'transformed_time_series_by_store', subgroup,
                                                    subgroup + '_' + store + '_time_series.csv')
        else:
            # Time series file path
            self.time_series_file_path = os.path.join(self.folder_path, 'transformed_time_series_by_subgroup',
                                                      subgroup  + '_time_series.csv')
    
# This class will perform the feature engineering process on the different datasets    
class FeatureEngineering:
    def __init__(self, subgroup : str, store : str, group_by : str,
                 daily_average_quantity : bool,
                 daily_total_quantity : bool,
                 daily_dispersion_quantity : bool,
                 daily_average_price : bool,
                 daily_dispersion_price : bool,
                 daily_total_sales : bool,
                 daily_dispersion_total_sales : bool,
                 daily_difference_average_quantity : bool,
                 daily_difference_total_quantity : bool,
                 daily_difference_dispersion_quantity : bool,
                 daily_difference_average_price : bool,
                 daily_difference_dispersion_price : bool,
                 daily_difference_total_sales : bool,
                 daily_difference_dispersion_total_sales : bool,
                 daily_second_difference_average_quantity : bool,
                 daily_second_difference_total_quantity : bool,
                 daily_second_difference_dispersion_quantity : bool,
                 daily_second_difference_average_price : bool,
                 daily_second_difference_dispersion_price : bool,
                 daily_second_difference_total_sales : bool,
                 daily_second_difference_dispersion_total_sales : bool,
                 daily_percentage_change_average_price : bool,
                 daily_percentage_change_total_quantity : bool,
                 daily_percentage_change_total_sales : bool,
                 elasticity : bool,
                 difference_elasticity : bool,
                 second_difference_elasticity : bool,
                 cumulative_total_sales : bool,
                 cumulative_total_quantity : bool
                 ):
        # Subgroup name
        self.subgroup = subgroup
        
        # Store ID
        self.store = store
        
        # Group by this variable
        self.group_by = group_by
        
        # Path to the initial dataset
        self.path = os.path.join('../../../data/final_datasets/time_series', self.subgroup + '_time_series.csv')
        
        # ------------------------------------ Boolean flags ------------------------------------ #
        self.daily_average_quantity = daily_average_quantity
        self.daily_total_quantity = daily_total_quantity
        self.daily_dispersion_quantity = daily_dispersion_quantity
        self.daily_average_price = daily_average_price
        self.daily_dispersion_price = daily_dispersion_price
        self.daily_total_sales = daily_total_sales
        self.daily_dispersion_total_sales = daily_dispersion_total_sales
        self.daily_difference_average_quantity = daily_difference_average_quantity
        self.daily_difference_total_quantity = daily_difference_total_quantity
        self.daily_difference_dispersion_quantity = daily_difference_dispersion_quantity
        self.daily_difference_average_price = daily_difference_average_price
        self.daily_difference_dispersion_price = daily_difference_dispersion_price
        self.daily_difference_total_sales = daily_difference_total_sales
        self.daily_difference_dispersion_total_sales = daily_difference_dispersion_total_sales
        self.daily_second_difference_average_quantity = daily_second_difference_average_quantity
        self.daily_second_difference_total_quantity = daily_second_difference_total_quantity
        self.daily_second_difference_dispersion_quantity = daily_second_difference_dispersion_quantity
        self.daily_second_difference_average_price = daily_second_difference_average_price
        self.daily_second_difference_dispersion_price = daily_second_difference_dispersion_price
        self.daily_second_difference_total_sales = daily_second_difference_total_sales
        self.daily_second_difference_dispersion_total_sales = daily_second_difference_dispersion_total_sales
        self.daily_percentage_change_average_price = daily_percentage_change_average_price
        self.daily_percentage_change_total_quantity = daily_percentage_change_total_quantity
        self.daily_percentage_change_total_sales = daily_percentage_change_total_sales
        self.elasticity = elasticity
        self.difference_elasticity = difference_elasticity
        self.second_difference_elasticity = second_difference_elasticity
        self.cumulative_total_sales = cumulative_total_sales
        self.cumulative_total_quantity = cumulative_total_quantity
        # --------------------------------------------------------------------------------------- #
        
    def initiate_feature_engineering(self):
        try:
            # Read the dataset using 'DATE' as a datetime64 index variable
            df = read_csv(self.path, index_col = 'DATE', parse_dates = True)
            
            logging.info("Dataset correctly loaded")
            
            # New dataset to store the time series
            df_new = DataFrame()
            
            # Chose the filter
            if self.group_by == 'STORE_ID':
                filter = self.store
            elif self.group_by == 'SUBGROUP':
                filter = df[self.group_by].value_counts().index.to_numpy()[0]
            
            # ------------------------------ Daily average quantity ----------------------------- #
            if self.daily_average_quantity == True:
                df_new['daily_average_quantity'] = df['QUANTITY'].loc[
                    df[self.group_by] == filter
                    ].resample('1d').mean()
                logging.info("Daily average quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------- Daily total quantity ------------------------------ #
            if self.daily_total_quantity == True:
                df_new['daily_total_quantity'] = df['QUANTITY'].loc[
                    df[self.group_by] == filter
                    ].resample('1d').sum()
                logging.info("Daily total quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------------- Daily quantity dispersion ---------------------------- #
            if self.daily_dispersion_quantity == True:
                df_new['daily_dispersion_quantity'] = df['QUANTITY'].loc[
                    df[self.group_by] == filter
                    ].resample('1d').std()
                logging.info("Daily dispersion quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------- Daily average price ------------------------------- #
            if self.daily_average_price == True:
                df_new['daily_average_price'] = df['PRICE'].loc[
                    df[self.group_by] == filter
                    ].resample('1d').mean()
                logging.info("Daily average price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------ Daily price dispersion ----------------------------- #
            if self.daily_dispersion_price == True:
                df_new['daily_dispersion_price'] = df['PRICE'].loc[
                    df[self.group_by] == filter
                    ].resample('1d').std()
                logging.info("Daily dispersion price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ! NOTE THAT THIS IS THE VARIABLE TO PREDICT
            # --------------------------------- Daily total sales ------------------------------- #
            if self.daily_total_sales == True:
                df_new['daily_total_sales'] = df['TOTAL_SALES'].loc[
                    df[self.group_by] == filter
                    ].resample('1d').sum()
                logging.info("Daily total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # -------------------------- Daily total sales dispersion --------------------------- #
            if self.daily_dispersion_total_sales == True:
                df_new['daily_dispersion_total_sales'] = df['TOTAL_SALES'].loc[
                    df[self.group_by] == filter
                    ].resample('1d').std()
                logging.info("Daily dispersion total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # * FROM HERE ALL THE DATASETS CORRESPONDS TO THE SUBGROUP AND STORE ID GIVEN AT THE
            # * BEGGINING OF THE CODE
            
            # --------------------------------- First difference -------------------------------- #
            # ------------------------- Daily difference average quantity ----------------------- #
            if self.daily_difference_average_quantity == True:
                df_new['daily_difference_average_quantity'] = df_new['daily_average_quantity'] - \
                    df_new['daily_average_quantity'].shift(periods = 1)
                logging.info("Daily difference average quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- Daily difference total quantity ------------------------- #
            if self.daily_difference_total_quantity == True:
                df_new['daily_difference_total_quantity'] = df_new['daily_total_quantity'] - \
                    df_new['daily_total_quantity'].shift(periods = 1)
                logging.info("Daily difference total quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------- Daily difference quantity dispersion ---------------------- #
            if self.daily_difference_dispersion_quantity == True:
                df_new['daily_difference_dispersion_quantity'] = df_new['daily_dispersion_quantity'] - \
                    df_new['daily_dispersion_quantity'].shift(periods = 1)
                logging.info("Daily difference dispersion quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # -------------------------- Daily difference average price ------------------------- #
            if self.daily_difference_average_price == True:
                df_new['daily_difference_average_price'] = df_new['daily_average_price'] - \
                    df_new['daily_average_price'].shift(periods = 1)
                logging.info("Daily difference average price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- Daily difference price dispersion ----------------------- #
            if self.daily_difference_dispersion_price == True:
                df_new['daily_difference_dispersion_price'] = df_new['daily_dispersion_price'] - \
                    df_new['daily_dispersion_price'].shift(periods = 1)
                logging.info("Daily difference dispersion price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------------- Daily difference total sales ------------------------- #
            if self.daily_difference_total_sales == True:
                df_new['daily_difference_total_sales'] = df_new['daily_total_sales'] - \
                    df_new['daily_total_sales'].shift(periods = 1)
                logging.info("Daily difference total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # --------------------- Daily difference dispersion total sales --------------------- #
            if self.daily_difference_dispersion_total_sales == True:
                df_new['daily_difference_dispersion_total_sales'] = df_new['daily_dispersion_total_sales'] - \
                    df_new['daily_dispersion_total_sales'].shift(periods = 1)
                logging.info("Daily difference dispersion total sales computed")
            # ----------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------- #
            
            # -------------------------------- Second difference -------------------------------- #
            # ! Note that if it was not defined the first differences it will not be possible to compute
            # ! the second total differences
            # ---------------------- Daily second difference average quantity ------------------- #
            if self.daily_second_difference_average_quantity == True:
                df_new['daily_second_difference_average_quantity'] = df_new['daily_difference_average_quantity'] - \
                    df_new['daily_difference_average_quantity'].shift(periods = 1)
                logging.info("Daily second difference average quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------- Daily second difference total quantity --------------------- #
            if self.daily_second_difference_total_quantity == True:
                df_new['daily_second_difference_total_quantity'] = df_new['daily_difference_total_quantity'] - \
                    df_new['daily_difference_total_quantity'].shift(periods = 1)
                logging.info("Daily second difference total quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------- Daily second difference quantity dispersion ------------------- #
            if self.daily_second_difference_dispersion_quantity == True:
                df_new['daily_second_difference_dispersion_quantity'] = df_new['daily_difference_dispersion_quantity'] - \
                    df_new['daily_difference_dispersion_quantity'].shift(periods = 1)
                logging.info("Daily second difference dispersion quantity computed")
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------- Daily second difference average price ---------------------- #
            if self.daily_second_difference_average_price == True:
                df_new['daily_second_difference_average_price'] = df_new['daily_difference_average_price'] - \
                    df_new['daily_difference_average_price'].shift(periods = 1)
                logging.info("Daily second difference average price computed")
            # ----------------------------------------------------------------------------------- #
            
            # --------------------- Daily second difference price dispersion -------------------- #
            if self.daily_second_difference_dispersion_price == True:
                df_new['daily_second_difference_dispersion_price'] = df_new['daily_difference_dispersion_price'] - \
                    df_new['daily_difference_dispersion_price'].shift(periods = 1)
                logging.info("Daily second difference dispersion price computed")
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------- Daily second difference total sales ----------------------- #
            if self.daily_second_difference_total_sales == True:
                df_new['daily_second_difference_total_sales'] = df_new['daily_difference_total_sales'] - \
                    df_new['daily_difference_total_sales'].shift(periods = 1)
                logging.info("Daily second difference total sales computed")
            # ----------------------------------------------------------------------------------- #
            
            # ----------------- Daily second difference dispersion total sales ------------------ #
            if self.daily_second_difference_dispersion_total_sales == True:
                df_new['daily_second_difference_dispersion_total_sales'] = df_new['daily_difference_dispersion_total_sales'] - \
                    df_new['daily_difference_dispersion_total_sales'].shift(periods = 1)
                logging.info("Daily second difference dispersion total sales computed")
            # ----------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------- #
            
            # -------------------- Daily percentage change in average price --------------------- #
            if self.daily_percentage_change_average_price == True:
                df_new['daily_percentage_change_average_price'] = df_new['daily_average_price'].pct_change(periods = 1, fill_method = None)*100.0
                logging.info('Daily percentage change average price computed')
            # ----------------------------------------------------------------------------------- #
            
            # -------------------- Daily percetage change in total quantity --------------------- #
            if self.daily_percentage_change_total_quantity == True:
                df_new['daily_percentage_change_total_quantity'] = df_new['daily_total_quantity'].pct_change(periods = 1, fill_method = None)*100.0
                logging.info('Daily percentage change in total quantity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------- Daily percetage change in total sales ---------------------- #
            if self.daily_percentage_change_total_sales == True:
                df_new['daily_percentage_change_total_sales'] = df_new['daily_total_sales'].pct_change(periods = 1, fill_method = None)*100.0
                logging.info('Daily percentage change in total sales computed')
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------------------- Elasticity ------------------------------------ #
            if self.elasticity == True:
                df_new['elasticity'] = df_new['daily_percentage_change_total_quantity']/df_new['daily_percentage_change_average_price']
                logging.info('Elasticity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- First difference of elasticity -------------------------- #
            if self.difference_elasticity == True:
                df_new['difference_elasticity'] = df_new['elasticity'] - df_new['elasticity'].shift(periods = 1)
                logging.info('First difference of elasticity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------- Second difference of elasticity ------------------------- #
            if self.second_difference_elasticity == True:
                df_new['second_difference_elasticity'] = df_new['difference_elasticity'] - \
                    df_new['difference_elasticity'].shift(periods = 1)
                logging.info('Second difference of elasticity computed')
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------------- Cumulative total sales ------------------------------ #
            if self.cumulative_total_sales == True:
                df_new['cumulative_total_sales'] = df_new['daily_total_sales'].cumsum()
                logging.info('Cumulative total sales computed')
            # ----------------------------------------------------------------------------------- #
            
            # ---------------------------- Cumulative total quantity ---------------------------- #
            if self.cumulative_total_quantity == True:
                df_new['cumulative_total_quantity'] = df_new['daily_total_quantity'].cumsum()
                logging.info('Cumulative total quantity computed')
            # ----------------------------------------------------------------------------------- #
            
            # Replace infs by given values
            df_new.replace([inf, -inf], nan, inplace = True)
            
            # Fill the NaNs in the middle of the time series with the value inmediatly before (forward filling)
            df_new.ffill(inplace = True)
            
            # Fill the NaNs in the middle of the time series with the value inmediatly after (backward filling)
            df_new.bfill(inplace = True)            
            
            # Set the index column of the new DataFrame
            df_new.set_index(df_new['daily_average_quantity'].index, inplace = True)
            
            #print(df_new.info())
            
            # -------------------------------- Calendar features -------------------------------- #
            df_new['day_of_week'] = df_new.index.dayofweek.to_numpy()
            df_new['day_of_month'] = df_new.index.day.to_numpy()
            df_new['day_of_year'] = df_new.index.dayofyear.to_numpy()
            df_new['month'] = df_new.index.month.to_numpy()
            df_new['is_holiday'] = array([is_holiday(x) for x in df_new.index])
            # ----------------------------------------------------------------------------------- #
            
            # Path to save the output
            feature_engineering_config = FeatureEngineeringConfig(subgroup = self.subgroup, store = self.store)
            
            # Save the final DataFrame
            df_new.to_csv(feature_engineering_config.time_series_file_path)
            
            return feature_engineering_config.time_series_file_path
            
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    list_of_subgroups = []
    with open('../../../data/final_datasets/time_series/reformated_subgroup_names.txt', 'r') as file:
        for line in file:
            list_of_subgroups.append(line[:-1])
            
    list_of_stores = []
    with open('../../../data/final_datasets/time_series/transactions_store_names.txt', 'r') as file:
        for line in file:
            list_of_stores.append(line[:-1])
    
    for n, sub in enumerate(list_of_subgroups):
        for m, store in enumerate(list_of_stores[:1]):
            obj = FeatureEngineering(subgroup = sub,
                                     store = None,
                                     group_by = 'SUBGROUP',
                                     daily_average_quantity = True,
                                     daily_total_quantity = True,
                                     daily_dispersion_quantity = True,
                                     daily_average_price = True,
                                     daily_dispersion_price = True,
                                     daily_total_sales = True,
                                     daily_dispersion_total_sales = True,
                                     daily_difference_average_quantity = True,
                                     daily_difference_total_quantity = True,
                                     daily_difference_dispersion_quantity = True,
                                     daily_difference_average_price = True,
                                     daily_difference_dispersion_price = True,
                                     daily_difference_total_sales = True,
                                     daily_difference_dispersion_total_sales = True,
                                     daily_second_difference_average_quantity = True,
                                     daily_second_difference_total_quantity = True,
                                     daily_second_difference_dispersion_quantity = True,
                                     daily_second_difference_average_price = True,
                                     daily_second_difference_dispersion_price = True,
                                     daily_second_difference_total_sales = True,
                                     daily_second_difference_dispersion_total_sales = True,
                                     daily_percentage_change_average_price = True,
                                     daily_percentage_change_total_quantity = True,
                                     daily_percentage_change_total_sales = True,
                                     elasticity = True,
                                     difference_elasticity = True,
                                     second_difference_elasticity = True,
                                     cumulative_total_sales = True,
                                     cumulative_total_quantity = True)
            data = obj.initiate_feature_engineering()
            print(n, sub + ' - ', m, store + " time series made")
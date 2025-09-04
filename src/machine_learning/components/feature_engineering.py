# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.utils import time_series_file_path
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import concat, read_csv, DataFrame
from numpy import inf, nan
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the outputs to the feature engineering process
@dataclass
class FeatureEngineeringConfig:
    # Path to the machine learning folder
    folder_path = '../../../data/final_datasets/machine_learning'
    
    def __init__(self, file_name : str):
        # Creates the directory where the dataset will be stored
        os.makedirs(self.folder_path, exist_ok = True)
                
        # Dataset file path
        self.dataset_file_path = os.path.join(self.folder_path, file_name)
    
# This class will perform the feature engineering process on the different datasets    
class FeatureEngineering:
    def __init__(self, subgroup : list[str], store : list[str]):
        # Subgroup name list
        self.subgroup = subgroup
        
        # Store ID list
        self.store = store
        
    def initiate_feature_engineering(self):
        try:
            # Set the new DataFrame to store the time series
            df = read_csv(time_series_file_path(subgroup = 'coffee', store = 'S00139'), 
                          index_col = 'DATE', parse_dates = True)
            df_new = DataFrame(columns = df.columns)
            
            # To be sure about the dtpyes
            df_new = df_new.astype(df.dtypes)
            
            for n, sub in enumerate(self.subgroup):
                for m, store in enumerate(self.store):
                    # Read the dataset using 'DATE' as a datetime64 index variable
                    df = read_csv(time_series_file_path(subgroup = sub, store = store), 
                                  index_col = 'DATE', parse_dates = True)
                    
                    # Label encode the subgroup and store
                    df['subgroup'] = n
                    df['store'] = m
            
                    # Store the data in the new DataFrame
                    df_new = concat((df_new, df), axis = 0)
                    
                    #print(n, sub, '-', m, store)
                    
                logging.info("Dataset correctly concatened")
            
                # Change the dtpye of integer variables
                columns_tmp = df_new.select_dtypes(include = int).columns.to_list() + ['subgroup', 'store']
                df_new = df_new.astype({col : typ for (col, typ) in zip(columns_tmp, len(columns_tmp)*['int32'])})
            
                # Shuffle the data
                df_new = df_new.sample(frac = 1).reset_index(drop = True)
            
                logging.info("Dataset correctly shuffled")
                
                output = FeatureEngineeringConfig(sub + '_transactions.parquet')
                
                # Save the DataFrame
                #df_final.loc[df_final.loc[:, 'subgroup'] == n].to_parquet(output.dataset_file_path, index = False)
                df_new.loc[df_new.loc[:, 'subgroup'] == n].to_parquet(output.dataset_file_path)
                
                # Clear the DataFrame to avoid memory consuption
                df_new.drop(df_new.index, inplace = True)
                
                print(n, sub, 'dataset stored')
            
            """# Target variable
            y = df_new.loc[:, 'daily_total_sales'].copy()
                        
            # Quantitative predictors
            df_num = df_new.select_dtypes(exclude = int).drop(columns = ['daily_total_sales'])
            
            # Quantitative columns
            quantitative_columns = df_num.columns
                        
            # Pipeline to apply to the quantitative variables
            pipe = Pipeline(steps = [("scaler", StandardScaler())])
            
            # Object to apply standardization on the quantitative variables
            preprocesor = ColumnTransformer([("pipeline", pipe, quantitative_columns)], n_jobs = 7)
            
            # Standardize the quantitative variables
            df_arr = preprocesor.fit_transform(df_num)
            
            logging.info("Dataset correctly standardized")
            
            # Convert standardized data into a DataFrame
            df_standardized = DataFrame(df_arr)
            
            # Set a dict of new names for the standardized data
            new_names = {col_old: col_new  for col_old, col_new in 
                         zip(df_standardized.columns, df_num.columns)}
            
            # Rename the columns of the data
            df_standardized.rename(columns = new_names, inplace = True)
            
            # Concatenate the standardized and the cut dataset
            df_final = concat((df_new.select_dtypes(include = int), df_standardized, y), axis = 1)"""
            
            return None
            
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
    
    obj = FeatureEngineering(subgroup = list_of_subgroups,
                             store = list_of_stores)
    data = obj.initiate_feature_engineering()
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv
from numpy import array, savetxt
# =============================================================================================== #

# This class will provide the paths for the outputs to the data spliting process
@dataclass
class DataSplitConfig:
    # Dataset folder path defined as a class attribute
    folder_path = '../../data/final_datasets/time_series'
    
    def __init__(self, file_name : str):
        # Dataset file path
        self.dataset_file_path = os.path.join(self.folder_path, file_name)
        
# Class to split the different time series of the transactions dataset
class DataSplit:
    def __init__(self, split_on_variable : str):        
        # Path to the initial dataset
        self.path = os.path.join('../../data/initial_datasets/eci_transactions.csv')
        
        # Variable where to perform the split
        self.split_on_variable = split_on_variable
        
    # This function initiates the data split process over the transactions data
    def initiate_data_split(self):
        try:
            # Load the initial dataset
            df = read_csv(self.path)
            
            logging.info("Reading of transactions data completed")
            
            # List of all subgroups availables
            subgroups = df[self.split_on_variable].value_counts().index.to_numpy()
            
            # Empty list of subgroup reformated names to store
            subgroup_names = []
            
            # Isolate each time series based on self.split_on_variable
            for n, sub in enumerate(subgroups):
                # Temporal DataFrame object
                df_tmp = df.loc[df[self.split_on_variable] == sub]
                
                # Set the name of the output file
                sub_tmp = sub.lower().replace("'s", '').replace('-', '_').replace('&', 'and').replace(' ', '_')
                name_tmp = sub_tmp + '_time_series.csv'
                
                # Temporal DataSplitConfig object to set the output's path
                data_split_config_tmp = DataSplitConfig(name_tmp)
                
                # ----- Clean the split dataset
                # Check for columns where there are missining values
                missing_value_cols = df_tmp.isnull().sum().loc[df_tmp.isnull().sum() > 0].index
            
                # Check if there is a column with missing values
                if len(missing_value_cols) > 0:
                    for col in missing_value_cols:
                        df_tmp[col].fillna(method = 'backfill', inplace = True)
                # -----------------------------
                
                # Save the split and clean dataset
                df_tmp.to_csv(data_split_config_tmp.dataset_file_path, index=False)
                
                # Store the different subgroups reformated names
                subgroup_names.append(sub_tmp)
                
                # To check step by step
                print(n, sub_tmp)
            
            logging.info("Split datasets saved")
            
            # Save the list of subgroups reformated names
            with open(os.path.join(DataSplitConfig.folder_path, 'reformated_subgroup_names.txt'), 'w') as file:
                for item in subgroup_names:
                    file.write(item + '\n')
            
            return DataSplitConfig.folder_path
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    variable = 'SUBGROUP'
    
    obj = DataSplit(variable)
    data = obj.initiate_data_split()
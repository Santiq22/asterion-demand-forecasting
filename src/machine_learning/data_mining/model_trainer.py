# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import train_model, TabularResMLP
from dataclasses import dataclass
from pandas import concat, read_parquet
from numpy import array, savetxt

# --- Neural Network
import torch
from torch import nn
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class ModelTrainerConfig:
    # Path to the trained models folder
    folder_path = 'trained_models/'
    
    def __init__(self, model_name : str, subgroup : str | None, store : str | None):        
        # Creates the directory where the trained models will be stored given the subgroup
        if subgroup != None:
            os.makedirs(os.path.join(self.folder_path, subgroup), exist_ok = True)
    
        # Path to the output trained model saved as .pth files
        if subgroup != None:
            self.trained_model_file_path = os.path.join(self.folder_path, subgroup,
                                                        subgroup + '_' + store + '_' + model_name + '.pth')
        else:
            self.trained_model_file_path = os.path.join(self.folder_path, model_name + '.pth')
    
class ModelTrainer:
    # This class trains the model and saves it in ModelTrainerConfig.trained_model_file_path
    def __init__(self, subgroup : str | None, store : str | None, list_of_subgroups : list[str] | None, model_name : str, 
                 target = None | str, model = nn.Module, number_of_splits = None | int):
        
        # Set the output path to save the trained model
        self.model_trainer_config = ModelTrainerConfig(subgroup = subgroup, store = store, model_name = model_name)
        
        # List of subgroups
        self.list_of_subgroups = list_of_subgroups
        
        # Path to the initial dataset
        if self.list_of_subgroups == None:
            self.path = os.path.join('../../../data/final_datasets/machine_learning', subgroup, subgroup + '_' + store + '_transactions.parquet')
        
        # Target variable to use when optimizing the forecaster
        self.target = target
        
        # Model object to be trained
        self.model = model
        
        # Number of splits to do divide the dataset while doing CV
        self.number_of_splits = number_of_splits
        
    def initiate_model_trainer(self):
        try:
            # Load the dataset containing the time series, for a given subgroup and store
            if list_of_subgroups == None:
                df = read_parquet(self.path)
            else:
                dfs = []
                for n, sub in enumerate(list_of_subgroups):
                    #df = read_parquet(os.path.join('../../../data/final_datasets/machine_learning', sub + '_transactions.parquet'))
                    df = read_parquet(os.path.join('../../../data/final_datasets/machine_learning', sub + '_transactions.parquet'), 
                                      columns = ['day_of_week',
                                                 'day_of_month',
                                                 'day_of_year',
                                                 'month',
                                                 'is_holiday',
                                                 'subgroup',
                                                 'store',
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
                                                 'daily_total_sales'])
                    dfs.append(df)
                    print(n, sub, 'loaded')
                    
            # Concatenate DataFrames
            dfs = concat(dfs, axis = 0)
            
            logging.info("Dataset correctly loaded")
            
            # Shuffle the data
            dfs = dfs.sample(frac = 1).reset_index(drop = True)
            
            # Numerical columns
            num_cols = dfs.select_dtypes(exclude = int).drop(columns = [self.target]).columns.to_list()
            
            # Categorical columns
            cat_cols = dfs.select_dtypes(include = int).columns.to_list()
            
            # Cardinalities of categorical vars (unique values in each)
            cat_cardinalities = [dfs[col].nunique() for col in cat_cols]

            # Instantiate the model
            self.model = self.model(cat_cardinalities, num_features = len(num_cols), out_dim = 1)
            
            # Train the model
            trained_model, train_loss, validation_loss = train_model(model = self.model,
                                                                     df = dfs,
                                                                     num_cols = num_cols,
                                                                     cat_cols = cat_cols,
                                                                     target_col = self.target,
                                                                     n_epochs = 25)
                
            logging.info("Model correctly trained")
            
            # Save the model
            torch.save(trained_model, self.model_trainer_config.trained_model_file_path)
            
            logging.info("Trained model saved")
            
            # Save losses functions
            train_loss = array(train_loss)
            validation_loss = array(validation_loss)
            savetxt(os.path.join(ModelTrainerConfig.folder_path, 'train_loss_v2.txt'), train_loss)
            savetxt(os.path.join(ModelTrainerConfig.folder_path, 'validation_loss_v2.txt'), validation_loss)
                        
            return self.model_trainer_config.trained_model_file_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    list_of_subgroups = []
    with open('../../../data/final_datasets/time_series/reformated_subgroup_names.txt', 'r') as file:
        for line in file:
            list_of_subgroups.append(line[:-1])
    
    obj = ModelTrainer(subgroup = None,
                       store = None,
                       list_of_subgroups = list_of_subgroups,
                       model_name = 'TabularResMLP_v2',
                       target = 'daily_total_sales',
                       model = TabularResMLP)
    mod = obj.initiate_model_trainer()
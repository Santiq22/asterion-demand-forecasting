# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models_ts
from dataclasses import dataclass
from pandas import read_csv
from numpy import arange
from darts.models import TFTModel, Prophet
from darts import TimeSeries
from itertools import product
from random import choice
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class ModelTrainerConfig:
    # Path to the trained models folder
    folder_path = 'trained_models/'
    
    def __init__(self, subgroup : str, store : str, model_name : str):        
        # Creates the directory where the trained models will be stored given the subgroup
        os.makedirs(os.path.join(self.folder_path, subgroup), exist_ok = True)
    
        # Path to the output trained model saved as .pkl files
        self.trained_model_file_path = os.path.join(self.folder_path, subgroup,
                                                    subgroup + '_' + store + '_' + model_name + '.pkl')
    
class ModelTrainer:
    # This class trains the model and saves it in ModelTrainerConfig.trained_model_file_path
    def __init__(self, subgroup : str, store : str, model_name : str, past_covariates : str | list[str] | None,
                 future_covariates : str | list[str], parameters : dict[str, int | float | str] | list[dict[str, int | float | str]], 
                 date_of_split : str, periods_to_predict : int, target = None | str, model = None | str, 
                 number_of_splits = None | int, test_size = None | int, best_fit = None | bool):
        # Set the output path to save the trained model
        self.model_trainer_config = ModelTrainerConfig(subgroup = subgroup, store = store, model_name = model_name)
        
        # Path to the initial dataset
        self.path = os.path.join('../../data/final_datasets/time_series/transformed_time_series/transformed_time_series_by_store', 
                                 subgroup, subgroup + '_' + store + '_time_series.csv')
        
        # Name of the model to train
        self.model_name = model_name
        
        # Names of past covariates to use
        self.past_covariates = past_covariates
        
        # Names of future covariates to use
        self.future_covariates = future_covariates
        
        # List of dictionaries of hyperparameter names and values to optimize the model using grid search CV
        self.parameters = parameters
        
        # Date where to split the dataset to build the train dataset
        self.date_of_split = date_of_split
        
        # Periods in the future to give to future covariates
        self.periods_to_predict = periods_to_predict
        
        # Target variable to use when optimizing the forecaster
        self.target = target
        
        # Model object to be either trained or optimized and trained
        self.model = model
        
        # Number of splits to do divide the dataset while doing CV
        self.number_of_splits = number_of_splits
        
        # Test size in CV
        self.test_size = test_size
        
        # Wheter to perform CV or not
        self.best_fit = best_fit
        
    def initiate_model_trainer(self):
        try:
            # Load the dataset containing the time series, for a given subgroup and store
            df = read_csv(self.path, index_col = 'DATE', parse_dates = True)
            
            logging.info("Dataset correctly loaded")
            
            if self.best_fit == True:
                # Perform CV
                report = evaluate_models_ts(df = df,
                                            variables = self.variables,
                                            target = self.target,
                                            model = self.model,
                                            parameters = self.parameters,
                                            number_of_splits = self.number_of_splits,
                                            test_size = self.test_size)
                
                # Print best-fit hyperparameters
                print(report['best_fit_hyperparameters'])
            
                # Dataset to train the model
                y = TimeSeries.from_series(df.loc[:self.date_of_split, self.target])
            
                # Train best fit model
                model_to_train = self.model(**report['best_fit_hyperparameters'])
            
                # Fit the model
                model_to_train = model_to_train.fit(y)
                
            else:
                # Train best fit model
                model_to_train = self.model(**self.parameters)
                
                # Target time series
                y = TimeSeries.from_dataframe(df.iloc[:-21], value_cols = self.target)
                
                # Past covariates to use as extra variables
                past_cov = TimeSeries.from_dataframe(df.iloc[:-21], value_cols = self.past_covariates)
                
                # Shift the covariates
                #df_tmp = df.shift(7, freq = 'D')
                
                # Drop the NaN values generated by the shift
                #df_tmp.dropna(inplace = True)
                
                # Future covariates to use as extra variables
                #future_cov = TimeSeries.from_dataframe(df_tmp, value_cols = self.future_covariates)
                future_cov = TimeSeries.from_dataframe(df.iloc[:-21], value_cols = self.future_covariates)
            
                # Fit the model
                model_to_train = model_to_train.fit(y, past_covariates = past_cov, future_covariates = future_cov)
            
            logging.info("Model correctly fit")
            
            # Save the model using the utils function save_object
            #save_object(file_path = self.model_trainer_config.trained_model_file_path, obj = model_to_train)
            model_to_train.save(self.model_trainer_config.trained_model_file_path)
            
            logging.info("Trained model saved")
                        
            return self.model_trainer_config.trained_model_file_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    #parameters = [{'add_seasonalities' : [{'name' : "annual_seasonality", 'seasonal_periods' : annual, 'fourier_order' : 30},
    #                                     {'name' : "monthly_seasonality", 'seasonal_periods' : month, 'fourier_order' : 30},
    #                                     {'name' : "weekly_seasonality", 'seasonal_periods' : week, 'fourier_order' : 30}]} for (week, month, annual) in product(arange(5, 8.5, 0.3),
    #                                                                                                                                                             arange(27, 32, 0.5),
    #                                                                                                                                                             arange(358, 363, 0.5))]
    #parameters = {'add_seasonalities' : [{'name' : "annual_seasonality", 'seasonal_periods' : 365.25, 'fourier_order' : 20, 'prior_scale' : 100.0},
    #                                     {'name' : "monthly_seasonality", 'seasonal_periods' : 30.5, 'fourier_order' : 20, 'prior_scale' : 1000.0},
    #                                     {'name' : "weekly_seasonality", 'seasonal_periods' : 7.0, 'fourier_order' : 20, 'prior_scale' : 1000.0}],
    #              'changepoint_prior_scale' : 10.0}
    prediction_steps = 7
    
    list_of_subgroups = []
    with open('../../data/final_datasets/time_series/reformated_subgroup_names.txt', 'r') as file:
        for line in file:
            list_of_subgroups.append(line[:-1])
            
    list_of_stores = []
    with open('../../data/final_datasets/time_series/transactions_store_names.txt', 'r') as file:
        for line in file:
            list_of_stores.append(line[:-1])
    
    for m, sub in enumerate(list_of_subgroups):
        for n, store in enumerate(list_of_stores[:1]):
            #sub = 'basketball'
            df_tmp = read_csv(os.path.join('../../data/final_datasets/time_series/transformed_time_series/transformed_time_series_by_store', 
                                 sub, sub + '_' + store + '_time_series.csv'))
            parameters = {'input_chunk_length' : df_tmp.shape[0] - prediction_steps - 21,
                        'output_chunk_length' : prediction_steps,
                        'hidden_size' : 64,
                        'num_attention_heads' : 8,
                        'full_attention' : True,
                        'dropout' : 0.1,
                        'optimizer_kwargs' : {'lr': 1.0e-3},
                        'batch_size' : 32,
                        'n_epochs' : 350,
                        'random_state' : 42}
            obj = ModelTrainer(subgroup = sub,
                               store = store,
                               model_name = 'TFT',
                               past_covariates = ["daily_average_quantity",
                                                  "daily_total_quantity",
                                                  "daily_dispersion_quantity", 
                                                  "daily_average_price", 
                                                  "daily_dispersion_price", 
                                                  "daily_dispersion_total_sales",
                                                  "daily_percentage_change_total_sales",
                                                  "daily_percentage_change_total_quantity",
                                                  "daily_percentage_change_average_price"],
                               future_covariates = ["elasticity"],
                               parameters = parameters,
                               date_of_split = '2023-12-24',
                               target = 'daily_total_sales',
                               model = TFTModel,
                               number_of_splits = 10,
                               test_size = 14,
                               periods_to_predict = prediction_steps,
                               best_fit = False)
            mod = obj.initiate_model_trainer()
            print(m, sub, ' -', store + " model trained")
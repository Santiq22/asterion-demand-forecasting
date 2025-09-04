# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import best_fit_optimizer
from dataclasses import dataclass
from pandas import read_csv
from numpy import array, nan
# =============================================================================================== #

# ======================================== Main classes ========================================= #

# This class will provide the path for the output to the dataset containing the best fit parameters
@dataclass
class DemandOptimizerConfig:
    # Dataset folder path defined as a class attribute
    folder_path = '../../data/final_datasets/cleaned_datasets'
    
    def __init__(self, file_name : str):
        # Dataset file path
        self.dataset_file_path = os.path.join(self.folder_path, file_name)
        
# This class will perform the best fit parameter tuning based on a given demand law
class DemandOptimizer:
    def __init__(self, data_input_name : str):
        # Object representing the path of the output dataset
        self.demand_optimizer_config = DemandOptimizerConfig(data_input_name)
        
    # This initiate the process of finding the best fit
    def initiate_optimization(self):
        try:
            # Read the dataset
            df_products = read_csv(self.demand_optimizer_config.dataset_file_path)
            
            logging.info("Products dataset correctly read")
            
            # List of empty solutions
            sols = []
            
            # Loop through the different subgroups
            for sub in df_products['subgroup']:
                if sub != 'basketball':
                    # Read the corresponding dataset
                    df = read_csv(os.path.join('../../data/final_datasets/time_series/transformed_time_series/transformed_time_series_by_subgroup', sub + '_time_series.csv'))
                    
                    # Variables
                    var = df.loc[:, ['daily_total_quantity', 'daily_average_price']].values
                    base_price = df_products.loc[df_products['subgroup'] == sub, ['base_price']].values
                    p = var[:, 1]
                    Q = var[:, 0]/152.0    # We divide by 152 to take the 'daily average total quantity by store'
                    
                    # Call optimizer
                    solution = best_fit_optimizer(price = p, quantity = Q, base_price = base_price)
                    
                    print('=============================', sub, solution)
                    
                    # Append solution
                    sols.append([solution[0], solution[1]])
                else:
                    # Append NaNs so then they can be replaced by mean values
                    sols.append([nan, nan])
                    
                    print('=============================', sub, solution)
                
            logging.info("Best fit procedure completed")
                
            # Convert solutions into array
            sols = array(sols)
                
            # Add new columns to DataFrame
            df_products['estimated_daily_total_average_demand'] = sols[:, 0]
            df_products['elasticity'] = sols[:, 1]
            
            # Fill the NaNs
            df_products.fillna(value = {'estimated_daily_total_average_demand' : df_products['estimated_daily_total_average_demand'].mean(),
                                        'elasticity' : df_products['elasticity'].mean()}, inplace = True)
            
            # Save the DataFrame
            df_products.to_csv(self.demand_optimizer_config.dataset_file_path, index = False)
                
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    obj = DemandOptimizer(data_input_name = 'eci_product_master_standardized_clean.csv')
    obj.initiate_optimization()
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import optimal_price, Q
from dataclasses import dataclass
from pandas import concat, read_csv
from numpy import array, savetxt, nan
# =============================================================================================== #

# ======================================== Main classes ========================================= #

# This class will provide the path for the output to the dataset containing the pricing strategy
@dataclass
class OptimalPriceConfig:
    # Dataset folder path defined as a class attribute
    folder_path = '../../data/submissions'
    
    def __init__(self, file_name : str):
        # Dataset file path
        self.dataset_file_path = os.path.join(self.folder_path, file_name)
        
# This class will perform the computation of the optimal price based on product, store and date
class OptimalPrice:
    def __init__(self, data_input_name : str, data_output_name : str, threshold : float):
        # Object representing the path of the output dataset
        self.demand_optimizer_config = OptimalPriceConfig(data_output_name)
        
        # Path to the dataset containing the parameters of the power law
        self.path = os.path.join('../../data/final_datasets/cleaned_datasets', data_input_name)
        
        # Threshold factor to avoid loss of profit
        self.threshold = threshold
        
    # This initiate the process of computing the optimal price
    def initiate_optimal_price_computation(self):
        try:
            # Read the dataset containing the power law parameters
            df_products = read_csv(self.path)
            
            # Read the submission file
            df_submission = read_csv(self.demand_optimizer_config.dataset_file_path)
            
            logging.info("Products and submission datasets correctly read")
            
            # Add STORE_ID, SUBGROUP and DATE columns to the submission file
            for i, value in enumerate(df_submission['STORE_SUBGROUP_DATE_ID']):
                df_submission.loc[i, 'DATE'] = value[-10:]
                value_tmp = value[:-11].split('_')
                df_submission.loc[i, 'STORE_ID'] = value_tmp[0]
                df_submission.loc[i, 'SUBGROUP'] = value_tmp[1]
                
            logging.info("DATE, STORE_ID and SUBGROUP variables added")

            # Standardize the SUBGROUP variable
            for n, sub in enumerate(df_submission['SUBGROUP']):
                df_submission.loc[n, 'SUBGROUP'] = sub.lower().replace("'s", '').replace('-', '_').replace('&', 'and').replace(' ', '_')
                
            logging.info("SUBGROUP variable correctly standardized")
            
            # Loop through the submission dataset
            for n, sub in enumerate(df_submission['SUBGROUP']):
                # Get parameters needed to compute optimal price
                Q_base = df_products.loc[df_products.loc[:, 'subgroup'] == sub, ['estimated_daily_total_average_demand']].values[0][0]
                e = df_products.loc[df_products.loc[:, 'subgroup'] == sub, ['elasticity']].values[0][0]
                T = df_submission.loc[n, 'TOTAL_SALES']
                p_base = df_products.loc[df_products.loc[:, 'subgroup'] == sub, ['base_price']].values[0][0]
                costos = df_products.loc[df_products.loc[:, 'subgroup'] == sub, ['costos']].values[0][0]
                
                # Compute optimal price for given date, product and store
                price = optimal_price(Q_base = Q_base, e = e, T = T, p_base = p_base)
                
                # Check if optimal price is larger than 1.1*costos to avoid loss of profit
                if price > self.threshold*costos:
                    df_submission.loc[n, 'OPTIMAL_PRICE'] = price
                else:
                    price = self.threshold*costos
                    df_submission.loc[n, 'OPTIMAL_PRICE'] = price
                
                # Percentage of added value for given date, product and store
                added_value = (price - costos)/costos*100.0
                df_submission.loc[n, 'ADDED_VALUE'] = added_value
                
                # Total profit for given date, product and store
                Q_est = Q(parameters = array([Q_base, e]), p = price, p_base = p_base)
                profit = T - costos*Q_est
                df_submission.loc[n, 'PROFIT'] = profit
                
                # Elasticity for the corresponding product
                df_submission.loc[n, 'ELASTICITY'] = e
                
                print(n, sub)
                
            logging.info("Pricing strategy computed")
                
            # Save the DataFrame
            df_submission.to_csv(self.demand_optimizer_config.dataset_file_path, index = False)
            
            logging.info("Dataset saved")
                
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    obj = OptimalPrice(data_input_name = 'eci_product_master_standardized_clean.csv', 
                       data_output_name = 'submission_final.csv',
                       threshold = 1.1)
    obj.initiate_optimal_price_computation()
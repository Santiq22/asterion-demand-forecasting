# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import concat, read_csv, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the outputs to the data cleaning process
@dataclass
class DataCleaningConfig:
    def __init__(self, file_name : str):
        # Dataset file path
        self.dataset_file_path = os.path.join('../../data/final_datasets/cleaned_datasets', file_name)
    
# Class to deal with the missing values of the dataset
class DataCleaning:
    def __init__(self, data_input_name : str, data_output_name : str):
        # Attribute representing the cleaned dataset filepath
        self.data_cleaning_config = DataCleaningConfig(data_output_name)
        
        # Path to the initial dataset
        self.path = os.path.join('../../data/initial_datasets', data_input_name)
        
    # This function initiates the data cleaning process over the stores data
    def initiate_data_cleaning_stores(self):
        try:
            # Load the initial dataset
            df = read_csv(self.path)
            
            logging.info("Reading of stores data completed")
            
            # Check for columns where there are missining values
            missing_value_cols = df.isnull().sum().loc[df.isnull().sum() > 0].index
            
            # Check if there is a column with missing values
            if len(missing_value_cols) > 0:
                for col in missing_value_cols:
                    df[col].fillna(value = 'Other', inplace = True)
                
            logging.info("Filling of NaNs completed")
            
            # Standardize the STORE_ID variable
            for n, s in enumerate(df.loc[:, 'STORE_ID'].str.replace('TORE', '0')):
                s = s[1:]
                s = 'S' + s
                df.loc[n, 'STORE_ID'] = s
                
            # Save the cleaned dataset
            df.to_csv(self.data_cleaning_config.dataset_file_path, index=False)
            
            logging.info("Cleaned dataset saved")
            
            return self.data_cleaning_config.dataset_file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    # This function initiates the data cleaning process over the clients data
    def initiate_data_cleaning_clients(self):
        try:
            # Load the initial dataset
            df = read_csv(self.path)
            
            logging.info("Reading of client data completed")
            
            # Check for columns where there are missining values
            missing_value_cols = df.isnull().sum().loc[df.isnull().sum() > 0].index
            
            # Check if there is a column with missing values
            if len(missing_value_cols) > 0:
                for col in missing_value_cols:
                    df[col].fillna(value = 'Other', inplace = True)
                    
            logging.info("Filling of NaNs completed")
                
            # Save the cleaned dataset
            df.to_csv(self.data_cleaning_config.dataset_file_path, index=False)
            
            logging.info("Cleaned dataset saved")
            
            return self.data_cleaning_config.dataset_file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    # This function initiates the data cleaning process over the transactions data
    def initiate_data_cleaning_transactions(self):
        try:
            # Load the initial dataset
            df = read_csv(self.path)
            
            logging.info("Reading of transactions data completed")
            
            # Check for columns where there are missining values
            missing_value_cols = df.isnull().sum().loc[df.isnull().sum() > 0].index
            
            # Check if there is a column with missing values
            if len(missing_value_cols) > 0:
                for col in missing_value_cols:
                    df[col].ffill(inplace = True)
                    #df[col].fillna(value = float(round(df[col].mean())), inplace = True)
                    
            logging.info("Filling of NaNs completed")
                
            # Save the cleaned dataset
            df.to_csv(self.data_cleaning_config.dataset_file_path, index=False)
            
            logging.info("Cleaned dataset saved")
            
            return self.data_cleaning_config.dataset_file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    # This function initiates the data cleaning process over the products data
    def initiate_data_cleaning_products(self):
        try:
            # Load the initial dataset
            df = read_csv(self.path)
            
            logging.info("Reading of products data completed")
            
            # Separate the numerical variables
            df_numeric = df.select_dtypes(include = ['number'])
            
            # Columns of the corresponding quantitative variables
            numeric_columns = df_numeric.columns
            
            # Drop quantitative variables so then it is possible to concatenate
            df.drop(numeric_columns, axis = 1, inplace = True)
            
            # Pipeline to apply to the quantitative variables in the products data
            numerical_pipeline = Pipeline(steps = [("scaler", StandardScaler())])
            
            # Object to apply standardization on the numeric variables
            preprocesor = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numeric_columns)])
            
            # Standardize the quantitative variables
            df_standardized_arr = preprocesor.fit_transform(df_numeric)
            
            logging.info("Dataset correctly standardized")
            
            # Convert standardized data into a DataFrame
            df_standardized = DataFrame(df_standardized_arr)
            
            # Set a dict of new names for the standardized data
            new_names = {col_old: col_new  for col_old, col_new in 
                         zip(df_standardized.columns, df_numeric.columns)}
            
            # Rename the columns of the data
            df_standardized.rename(columns = new_names, inplace = True)
            
            # Concatenate the standardized and the cut dataset
            df_new = concat((df, df_standardized), axis = 1)
            
            # Save the data
            df_new.to_csv(self.data_cleaning_config.dataset_file_path, index=False)
            
            logging.info("Standardized dataset saved")
            
            return self.data_cleaning_config.dataset_file_path            
        except Exception as e:
            raise CustomException(e, sys)
        
# =============================================================================================== #

if __name__ == "__main__":
    input_data = 'eci_transactions.csv'
    output_data = 'eci_transactions_clean.csv'
    
    obj = DataCleaning(input_data, output_data)
    data = obj.initiate_data_cleaning_transactions()
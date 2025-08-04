# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import month_plot, quarter_plot, plot_acf, plot_pacf
# =============================================================================================== #

# ====================================== Plot features ========================================== #
# Properties to decorate the plots.
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'   
plt.rcParams['font.sans-serif'] = 'New Century Schoolbook' # 'Times', 'Liberation Serif', 'Times New Roman'
#plt.rcParams['font.serif'] = ['Helvetica']
plt.rcParams['font.size'] = 10   # Antes era 15
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.edgecolor'] = 'k'
plt.rcParams['legend.markerscale'] = 1
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width']= 0.5
plt.rcParams['xtick.major.size']= 5.0
plt.rcParams['xtick.minor.width']= 0.5
plt.rcParams['xtick.minor.size']= 3.0
plt.rcParams['ytick.major.width']= 0.5
plt.rcParams['ytick.major.size']= 5.0
plt.rcParams['ytick.minor.width']= 0.5
plt.rcParams['ytick.minor.size']= 3.0
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the outputs to the making plots process
@dataclass
class PlotsMakerConfig:
    # Path to the plots folder
    folder_path = '../../plots'
    
    def __init__(self, subgroup : str, file_name : str):
        # Creates the directory where the plots will be stored given the subgroup
        os.makedirs(os.path.join(self.folder_path, subgroup), exist_ok = True)
        
        # Plots file path
        self.plots_file_path = os.path.join(os.path.join(self.folder_path, subgroup), file_name)

# Class to make the statistical plots for a given time series     
class PlotsMaker:
    def __init__(self, subgroup : str, data_input_name : str, variable : str, first_day : str, 
                 last_day : str, model_type : str, period : int, lags : int, seasonal_plot = True,
                 m_plot = True, q_plot = True, acf_plot = True, pacf_plot = True):
        # Subgroup name
        self.subgroup = subgroup
        
        # Path to the initial dataset
        self.path = os.path.join('../../data/final_datasets/time_series/transformed_time_series', data_input_name)
        
        # Time series variable
        self.variable = variable
        
        # First day time series
        self.first_day = first_day
        
        # Last day time series
        self.last_day = last_day
        
        # Type of model for seasonal decompose
        self.model_type = model_type
        
        # Period for seasonal decompose
        self.period = period
        
        # Lags for ACF and PACF
        self.lags = lags
        
        # Flag of seasonal decomposition plot
        self.seasonal_plot = seasonal_plot
        
        # Flag of month plot
        self.m_plot = m_plot
        
        # Flag of quarter plot
        self.q_plot = q_plot
        
        # Flag of acf plot
        self.acf_plot = acf_plot
        
        # Flag of pacf_plot
        self.pacf_plot = pacf_plot
    
    # This function initiates the process of making the plots    
    def initiate_plots_maker(self):
        try:
            # Read the dataset using 'DATE' as a datetime64 index variable
            df = read_csv(self.path, index_col = 'DATE', parse_dates = True)
            
            logging.info("Dataset correctly loaded")
            
            # Chose the time series corresponding to variable
            ts = df[self.variable]
            
            logging.info("Time series correctly chosen")
            
            # ------------------------------ Seasonal decomposition ----------------------------- #
            if self.seasonal_plot == True:
                sd = seasonal_decompose(ts[self.first_day:self.last_day],
                                        model = self.model_type, 
                                        period = self.period)
                fig_sd = sd.plot()
                fig_sd.set_size_inches(10, 7)
                fig_sd.set_dpi(400)
                fig_sd.suptitle(self.subgroup.replace('_', ' ').capitalize())
                plots_maker_config = PlotsMakerConfig(self.subgroup, self.subgroup + '_' + self.variable + '_seasonal_decomposition.png')
                plt.savefig(plots_maker_config.plots_file_path, bbox_inches='tight')
                logging.info("Seasonal decomposition plot done")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------------ Month plot ----------------------------------- #
            if self.m_plot == True:
                _, ax = plt.subplots(1, 1, figsize = (7, 4), dpi = 400)
                month_plot(ts.resample('ME').mean(), ylabel = self.variable.replace('_', ' ').capitalize(), ax = ax)
                ax.set_title(self.subgroup.replace('_', ' ').capitalize() + ' - ' + self.variable.replace('_', ' ').capitalize())
                ax.set_xlabel('Month')
                plots_maker_config = PlotsMakerConfig(self.subgroup, self.subgroup + '_' + self.variable + '_month_plot.png')
                plt.savefig(plots_maker_config.plots_file_path, bbox_inches='tight')
                logging.info("Month plot done")
            # ----------------------------------------------------------------------------------- #
            
            # ----------------------------------- Quarter plot ---------------------------------- #
            if self.q_plot == True:
                _, ax = plt.subplots(1, 1, figsize = (7, 4), dpi = 400)
                quarter_plot(ts.resample('QE').mean(), ylabel = self.variable.replace('_', ' ').capitalize(), ax = ax)
                ax.set_title(self.subgroup.replace('_', ' ').capitalize() + ' - ' + self.variable.replace('_', ' ').capitalize())
                ax.set_xlabel('Quarter')
                plots_maker_config = PlotsMakerConfig(self.subgroup, self.subgroup + '_' + self.variable + '_quarter_plot.png')
                plt.savefig(plots_maker_config.plots_file_path, bbox_inches='tight')
                logging.info("Quarter plot done")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------------- ACF plot ------------------------------------ #
            if self.acf_plot == True:
                _, ax = plt.subplots(1, 1, figsize = (7, 4), dpi = 400)
                plot_acf(ts, lags = self.lags, ax = ax)
                ax.set_xlabel('Days of lag')
                ax.set_ylabel('Autocorrelation coefficient')
                ax.set_title(self.subgroup.replace('_', ' ').capitalize() + ' - ' + self.variable.replace('_', ' ').capitalize())
                plots_maker_config = PlotsMakerConfig(self.subgroup, self.subgroup + '_' + self.variable + '_acf_plot.png')
                plt.savefig(plots_maker_config.plots_file_path, bbox_inches='tight')
                logging.info("ACF plot done")
            # ----------------------------------------------------------------------------------- #
            
            # ------------------------------------ PACF plot ------------------------------------ #
            if self.pacf_plot == True:
                _, ax = plt.subplots(1, 1, figsize = (7, 4), dpi = 400)
                plot_pacf(ts, lags = self.lags, ax = ax)
                ax.set_xlabel('Days of lag')
                ax.set_ylabel('Partial autocorrelation coefficient')
                ax.set_title(self.subgroup.replace('_', ' ').capitalize() + ' - ' + self.variable.replace('_', ' ').capitalize())
                plots_maker_config = PlotsMakerConfig(self.subgroup, self.subgroup + '_' + self.variable + '_pacf_plot.png')
                plt.savefig(plots_maker_config.plots_file_path, bbox_inches='tight')
                logging.info("PACF plot done")
            # ----------------------------------------------------------------------------------- #
            
            return plots_maker_config.folder_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #
        
if __name__ == "__main__":
    """list_of_subgroups = []
    with open('../../data/final_datasets/time_series/reformated_subgroup_names.txt', 'r') as file:
        for line in file:
            list_of_subgroups.append(line[:-1])
    
    for n, sub in enumerate(list_of_subgroups):
        obj = PlotsMaker(subgroup = sub,
                         data_input_name = sub + '_time_series.csv',
                         variable = 'daily_total_sales',
                         first_day = '2023-01-01',
                         last_day = '2023-12-31',
                         model_type = 'additive',
                         period = 7,
                         lags = 50)
        data = obj.initiate_plots_maker()
        print(n, sub + " plots made")"""
        
    list_of_variables = ['daily_average_quantity',
                        'daily_total_quantity',
                        'daily_dispersion_quantity',
                        'daily_average_price',
                        'daily_dispersion_price',
                        'daily_total_sales',
                        'daily_dispersion_total_sales',
                        'daily_difference_average_quantity',
                        'daily_difference_total_quantity',
                        'daily_difference_dispersion_quantity',
                        'daily_difference_average_price',
                        'daily_difference_dispersion_price',
                        'daily_difference_total_sales',
                        'daily_difference_dispersion_total_sales',
                        'daily_second_difference_average_quantity',
                        'daily_second_difference_total_quantity',
                        'daily_second_difference_dispersion_quantity',
                        'daily_second_difference_average_price',
                        'daily_second_difference_dispersion_price',
                        'daily_second_difference_total_sales',
                        'daily_second_difference_dispersion_total_sales',
                        'daily_percentage_change_average_price',
                        'daily_percentage_change_total_quantity',
                        'elasticity',
                        'difference_elasticity',
                        'second_difference_elasticity',
                        'cumulative_total_sales',
                        'cumulative_total_quantity']
    
    for n, var in enumerate(list_of_variables):
        obj = PlotsMaker(subgroup = 'coffee',
                         data_input_name = 'coffee_time_series.csv',
                         variable = var,
                         first_day = '2023-01-01',
                         last_day = '2023-12-31',
                         model_type = 'additive',
                         period = 7,
                         lags = 50,
                         seasonal_plot = True,
                         m_plot = False,
                         q_plot = False,
                         acf_plot = False,
                         pacf_plot = False)
        data = obj.initiate_plots_maker()
        print(n, var + " plot made")
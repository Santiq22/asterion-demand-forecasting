# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:36:44 2024

@author: Santiago Collazo
"""
# ========================================= Packages ============================================ #
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../'))
from src.exception import CustomException

import argparse
from darts import TimeSeries
from dataclasses import dataclass
from datetime import date, timedelta
import dill
from holidays import US
import lightgbm as lgb
import math
from numpy import array, isin, issubdtype, floating, mean, random, sum, unique, where
from numpy import cos, exp, expm1, log, log1p, polyfit, sin, zeros, inf, nan, pi, float32, ndarray
from pandas import date_range, read_csv, to_datetime, DataFrame, Series, Timestamp
from scipy import optimize
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
r2 = make_scorer(r2_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple

# --- Neural Network
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#from sklearnex import patch_sklearn
#patch_sklearn()
# =============================================================================================== #

# ======================================= Main functions ======================================== #
# Save an object in a given file path
def save_object(file_path, obj):
    try:
        # Directory path where is located file_path
        dir_path = os.path.dirname(file_path)
        
        # Create recursively all directories needed to contain dir_path
        os.makedirs(dir_path, exist_ok = True)
        
        # Open file_path in write and byte modes and pickle obj to file_obj
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
# Function that performs a GridSearchCV or a RandomizedSearchCV based on a ML approach
def evaluate_models_ml(X_train, y_train, X_test, y_test, model, parameters, n_iterations = None):
    try:
        # Instantiate model
        model_obj = model

        # Instantiate a GridSearchCV object using 5-fold CV or a RandomizedSearchCV
        if n_iterations == None:
            gs = GridSearchCV(model_obj, parameters, scoring = r2, cv = 5, n_jobs = 8, verbose = 2)
        else:
            gs = RandomizedSearchCV(model_obj, parameters, n_iter = n_iterations, 
                                    scoring = r2, cv = 5, n_jobs = 8, verbose = 2, random_state = 10)
            
        # Perform a grid or randomized search cv
        gs.fit(X_train, y_train)

        # Set best fit parameters to the corresponding model
        model_obj.set_params(**gs.best_params_)
            
        # Train the model
        model_obj.fit(X_train, y_train)
            
        # Predictions over training data
        y_train_pred = model_obj.predict(X_train)

        # Predictions over test data
        y_test_pred = model_obj.predict(X_test)

        # R^2 over training data
        training_model_score = r2_score(y_train, y_train_pred)

        # R^2 over test data
        test_model_score = r2_score(y_test, y_test_pred)

        # Save the trained model, training and test R^2 into the report dictionary
        report = {'trained_model': model_obj, 'training_score': training_model_score, 
                  'test_score': test_model_score, 'best_fit_hyperparameters': gs.best_params_}

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
# Function that performs a GridSearchCV or a RandomizedSearchCV based on a TS approach
def evaluate_models_ts(df : DataFrame, variables : str | list[str], target : str, model : object, 
                       parameters : list[dict[str, int | float | str]], number_of_splits : int, 
                       test_size : int):
    try:
        # Instantiate TimeSeriesSplit object
        tscv = TimeSeriesSplit(n_splits = number_of_splits, test_size = test_size)
        
        # Split data in training and test sample
        idxs = list(tscv.split(df))
        
        # Loop over the different dicts of set of parameters to train the model
        n_optim = None
        expected_test_error_old = -inf
        for n, param in enumerate(parameters):
            # List to store R^2 values
            metric = []
            
            # Loop over different indexes combinations to generate the training sets
            for (idx_train, idx_test) in idxs:
                # Define the training and test DataFrame
                if variables == None:
                    df_train = TimeSeries.from_series(df.loc[df.index[idx_train], target].copy())
                    df_test = df.loc[df.index[idx_test], target].copy()
                else:
                    pass
                
                # Instantiate model
                model_obj = model(**param)
                
                # Fit the model
                model_obj.fit(df_train)
                
                # Predict for the trained model
                predictions = model_obj.predict(test_size)
                
                # Compute the metric
                r2 = r2_score(y_true = df_test, y_pred = predictions.all_values()[:,0,0])
                metric.append(r2)
            
            # Compute the mean R^2 value
            expected_test_error_new = array(metric).mean()
            
            # Print the expected R^2 value
            print(n, expected_test_error_new)
            
            if expected_test_error_new > expected_test_error_old:
                n_optim = n
                expected_test_error_old = expected_test_error_new 

        # Set best fit parameters to the corresponding model
        model_obj = model(**parameters[n_optim])
        
        # Train the best fit model - The last df_train would be the largest
        model_obj.fit(df_train)
                
        # Predictions over test data
        y_test_pred = model_obj.predict(test_size)

        # R^2 over test data
        test_model_score = r2_score(y_true = df_test, y_pred = y_test_pred.all_values()[:,0,0])

        # Save the trained model and test R^2 into the report dictionary
        report = {'trained_model': model_obj, 'test_score': test_model_score, 
                  'best_fit_hyperparameters': parameters[n_optim]}

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def go_up_to(folder_name : str) -> Path:
    """
    Traverse up from the current working directory until the given folder is found.
    Returns the Path to that folder if found, otherwise raises FileNotFoundError.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if parent.name == folder_name:
            return parent
    raise FileNotFoundError(f"Folder '{folder_name}' not found above {Path.cwd()}")
    
# Path of the time series based on subgroup and store
def time_series_file_path(subgroup, store) -> str:
    return os.path.join(go_up_to('asterion-demand-forecasting'), 'data/final_datasets/time_series/transformed_time_series/transformed_time_series_by_store', 
                        subgroup, '{sub}_{store}_time_series.csv'.format(sub = subgroup, store = store))
    
# Check if a date is a US holiday
def is_holiday(date):
    if date in US(years = date.year):
        return 1
    else:
        return 0

# Empirical rule from fastai
def fastai_emb_sz(n_cat : int) -> int:
    return min(600, round(1.6*(n_cat**0.56)))

# Training function for the NN
def train_model(model : torch.nn.Module, df : DataFrame, num_cols : List[str], cat_cols : List[str], 
                target_col : str, task : str = "regression", test_size : float = 0.2, batch_size : int = 8192, 
                lr : float = 1e-3, weight_decay : float = 1e-5, n_epochs : int = 10, 
                device : str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[torch.nn.Module, List[float], List[float]]:
    """
    Train the TabularResMLP model.
    
    task is either regression or classification

    Returns:
        model: trained PyTorch model
        train_losses: list of average training loss per epoch
        val_losses: list of average validation loss per epoch
    """

    # --- Split train/val ---
    df_train, df_val = train_test_split(df, test_size = test_size, random_state = 42)

    train_ds = TabularDataset(df_train, num_cols, cat_cols, target_col)
    val_ds = TabularDataset(df_val, num_cols, cat_cols, target_col)

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    # --- Model & optimizer ---
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)

    if task == "regression":
        criterion = torch.nn.MSELoss()
    elif task == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    train_losses, val_losses = [], []

    # --- Training loop ---
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)

            # Set gradient to zero
            optimizer.zero_grad()
            
            # Make predictions
            preds = model(xb_num, xb_cat)

            # Compute loss function
            loss = criterion(preds, yb)

            # Compute gradient of loss function
            loss.backward()
            
            # Perform a single optimization step to update parameter
            optimizer.step()
            
            # Compute total loss
            total_loss += loss.item()*xb_num.size(0)

        avg_train_loss = total_loss/len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_num, xb_cat, yb in val_loader:
                xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
                
                # Make predictions
                preds = model(xb_num, xb_cat)
                
                # Compute loss function
                loss = criterion(preds, yb)
                
                # Compute total loss
                val_loss += loss.item()*xb_num.size(0)

        avg_val_loss = val_loss/len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses

# Demand: it is a power law function
def Q(parameters : ndarray, p : ndarray, p_base : float) -> ndarray:
    # Parameters to fit
    Q_base, e = parameters[0], parameters[1]
    
    return Q_base*(p/p_base)**e

# Optimal price: it is a power law function
def optimal_price(Q_base : float, e : float, T : ndarray, p_base : float) -> ndarray:    
    return (p_base**e*T/Q_base)**(1.0/(1.0 + e))

# Chi squared to fit demand
def chi_2(parameters : ndarray, x : ndarray, y : ndarray, p_base : float) -> float:
    # Predictions based on parameters, price and base price
    Q_pred = Q(parameters, x, p_base)
    
    # Compute chi squared function
    chi = sum((Q_pred - y)**2)/Q_pred.shape[0]
    return chi

# Optimizer to find the best fit parameters of the demand
def best_fit_optimizer(price : ndarray, quantity : ndarray, base_price : float) -> ndarray:
    # Set the bounds to perform the predictions
    bounds = ((0.0, 100.0), (-200.0, 0.0))
    
    # Carry out the optimization
    opt = optimize.differential_evolution(chi_2, bounds, strategy='best1bin', maxiter=150, 
                                          args = (price, quantity, base_price), popsize=75, 
                                          recombination=0.4, mutation=(0.2, 0.5),
                                          tol=1.0e-10, atol=0.0, disp=True, polish=True, 
                                          x0=array([1.0, -1.0]), seed=10, workers=8)
    
    solution = opt.x
    return solution

def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """ Computes the date of the n-th occurrence of a given weekday in a specific month and year. """
    
    d = date(year, month, 1)
    
    # d.weekday() will always correspond to the 1st of the given month
    days_ahead = (weekday - d.weekday()) % 7
    
    # Sums the days to reach the first desired weekday in the month
    d += timedelta(days = days_ahead)
    
    # Sums the weeks to reach the n-th desired weekday in the month
    d += timedelta(weeks = n - 1)
    
    # Check if still in the same month
    if d.month == month:
        return d
    else:
        raise ValueError("The corresponding occurence is in the next month.")

def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """ Computes the date of the last weekday of a given month. """
    
    next_month = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
    last_day = next_month - timedelta(days = 1)
    days_back = (last_day.weekday() - weekday) % 7
    
    return last_day - timedelta(days = days_back)

def easter_sunday(year: int) -> date:
    """ Computes the date of the easter sunday corresponding to year. """
    
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8)//25
    g = (b - f + 1)//3
    h = (19*a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month, day = divmod(h + l - 7*m + 114, 31)
    return date(year, month, day + 1)

def random_param_grid(sample_size: int, seed: int = 42):
    """
    A generator that yields random combinations of hyperparameters for tuning.
    This is for doing a random search instead of an exhaustive grid search.
    We selected this approach because it faster that use the classical grid search, 
    and we don't have a lot of time to do it.
    """
    rng = random.default_rng(seed)
    for _ in range(sample_size):
        max_depth = int(rng.choice([-1, 5, 8, 12, 20]))
        num_leaves = int(rng.choice([15, 31, 63, 127, 255]))
        if max_depth > 0: num_leaves = min(num_leaves, 2**max_depth - 1)
        yield {
            "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08])), "num_leaves": num_leaves, "max_depth": max_depth,
            "min_data_in_leaf": int(rng.choice([10, 20, 40, 60, 100])), "feature_fraction": float(rng.choice([0.6, 0.8, 0.95, 1.0])),
            "bagging_fraction": float(rng.choice([0.6, 0.8, 1.0])), "bagging_freq": int(rng.choice([0, 1, 3, 5])),
            "lambda_l1": float(rng.choice([0.0, 0.1, 1.0, 5.0])), "lambda_l2": float(rng.choice([0.0, 0.1, 1.0, 5.0])),
            "min_gain_to_split": float(rng.choice([0.0, 0.02, 0.05, 0.1])),
        }
        
def winsorize_series(s: Series, upper_q: float = 0.995) -> Series:
    """
    Clips the extreme upper values of a series to a given quantile.
    This helps to reduce the impact of outliers on the model.
    """
    return s.clip(upper = s.quantile(upper_q)) if not s.empty else s

def make_expanding_folds_by_date(dates: Series, n_splits: int, min_train: int, val_size: int):
    """
    Creates time-series-aware cross-validation folds.
    The training set expands over time, and the validation set is a fixed block
    that moves forward in time. 
    """
    uniq_dates = unique(to_datetime(dates).values)
    if len(uniq_dates) < min_train + val_size: return []
    
    folds, n = [], len(uniq_dates)
    max_start = n - val_size*n_splits
    start = max(min_train, max_start)
    for k in range(n_splits):
        train_end_idx = start + k*val_size
        val_end_idx = min(n, train_end_idx + val_size)
        
        if (val_end_idx - train_end_idx) < 3: break
        
        d_train, d_val = uniq_dates[:train_end_idx], uniq_dates[train_end_idx:val_end_idx]
        tr_mask, va_mask = isin(dates, d_train), isin(dates, d_val)
        folds.append((where(tr_mask)[0], where(va_mask)[0]))
    return folds

def time_decay_weights(dates: Series, halflife_days: float) -> ndarray:
    """
    Calculates sample weights that give more importance to recent data.
    The weight decreases exponentially with the age of the data point.
    """
    ages = (dates.max() - dates).dt.days.values.astype(float)
    return exp(-log(2)*ages/max(halflife_days, 1.0))

def tune_lgb_random_search(X: DataFrame, y_log: Series, dates: Series, cfg: TrainConfig):
    """
    Performs a random search for the best LightGBM hyperparameters using
    time-series cross-validation.
    """
    # Get the different folds to perform CV
    folds = make_expanding_folds_by_date(dates, cfg.n_splits, cfg.min_train, cfg.val_size)
    
    if not folds: return None
    
    best_score, best_params, best_metrics = inf, None, None
    for params in random_param_grid(cfg.n_iter, cfg.seed):
        r2s = []
        
        for tr_idx, va_idx in folds:
            # Split the dataset
            X_tr, y_tr, X_va, y_va = X.iloc[tr_idx], y_log.iloc[tr_idx], X.iloc[va_idx], y_log.iloc[va_idx]
            
            # Train the model
            model = lgb.train(
                {**BASE_PARAMS, **params}, lgb.Dataset(X_tr, label = y_tr), num_boost_round = cfg.max_rounds,
                valid_sets = [lgb.Dataset(X_va, label = y_va)], callbacks = [lgb.early_stopping(cfg.early_stop, verbose = False)]
            )
            
            # Make predictions
            preds_log = model.predict(X_va, num_iteration = model.best_iteration)
            r2s.append(r2_score(expm1(y_va), expm1(preds_log)))
            
        score = -mean(r2s)
        if score < best_score:
            best_score, best_params, best_metrics = score, params, {"r2_mean": -score}
            
    return best_params, best_metrics
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# Residual feed-forward block
class ResidualFFBlock(nn.Module):
    """
    Pre-norm residual MLP block:
      x -> LN -> Linear(h) -> GELU -> Dropout -> Linear(w) -> + skip
    """
    def __init__(self, dim : int, hidden : int, p_drop : float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(p_drop)
        
        # Kaiming/He init is fine for GELU
        nn.init.kaiming_uniform_(self.lin1.weight, a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.lin2.weight, a = math.sqrt(5))

    def forward(self, x):
        y = self.norm(x)
        y = F.gelu(self.lin1(y))
        y = self.drop(y)
        y = self.lin2(y)
        return x + y
    
# Tabular Residual Multilayer Perceptron
class TabularResMLP(nn.Module):
    """
    Feed-forward (MLP) for mixed tabular data:
      - categorical features via embeddings
      - numeric features with BatchNorm
      - Deep residual MLP backbone
    Args:
        cat_cardinalities: list with ints (unique values per categorical feature)
        num_features: number of numeric features
        hidden_profile: list[int] widths for each residual block's bottleneck
        p_drop: dropout in residual blocks
        emb_dropout: dropout applied to concatenated embeddings
        out_dim: number of targets (1 for regression, #classes for classification)
    """
    def __init__(self, cat_cardinalities : List[int], num_features : int, out_dim : int, 
                 hidden_profile : Optional[List[int]] = None, p_drop : float = 0.10, 
                 emb_dropout : float = 0.05):
        super().__init__()

        # Build embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings = card + 1, embedding_dim = fastai_emb_sz(card))
            for card in cat_cardinalities
        ])
        emb_dim_total = sum(emb.embedding_dim for emb in self.embeddings)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Numeric path
        self.num_bn = nn.BatchNorm1d(num_features)

        # Concatenated feature dimension
        model_dim = emb_dim_total + num_features

        # Default hidden profile (8 residual blocks, tapered)
        if hidden_profile is None:
            #hidden_profile = [1024, 1024, 768, 768, 768, 512, 512, 512]
            hidden_profile = [1024, 768, 768, 512, 512]

        # Input projection to model_dim (identity if already model_dim)
        # Here we keep it simple: operate directly at concatenated dim, and let blocks expand inside.
        self.blocks = nn.Sequential(*[
            ResidualFFBlock(model_dim, h, p_drop = p_drop) for h in hidden_profile
        ])

        # Final head: PreNorm -> MLP head
        self.final_norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, max(256, model_dim // 2)),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(max(256, model_dim // 2), out_dim),
            nn.Softplus()     # nn.ReLU() is also possible
        )

        # Initialize head
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a = math.sqrt(5))

    def forward(self, x_num : torch.Tensor, x_cat : torch.Tensor) -> torch.Tensor:
        """
        x_num: (B, p) float tensor
        x_cat: (B, l) long tensor with category indices
        """
        # Embeddings
        emb_list = []
        for i, emb in enumerate(self.embeddings):
            emb_list.append(emb(x_cat[:, i]))
            
        x_emb = torch.cat(emb_list, dim = 1) if emb_list else None
        x_emb = self.emb_dropout(x_emb)

        # Numeric
        xn = self.num_bn(x_num)

        # Concatenate
        x = torch.cat([xn, x_emb], dim = 1)

        # Residual backbone
        x = self.blocks(x)

        # Head
        x = self.final_norm(x)
        out = self.head(x)
        return out
    
# ---- Example usage ----
# Suppose your 7 categorical features have these cardinalities:
# cat_cardinalities = [12, 5, 31, 7, 104, 18, 52]
# model = TabularResMLP(cat_cardinalities, num_features=28, out_dim=1)  # regression
# logits = model(x_num, x_cat)

# Dataset
class TabularDataset(Dataset):
    def __init__(self, df : DataFrame, num_cols : List[str], cat_cols : List[str], target_col : str):
        self.num_data = df[num_cols].values.astype(float32)
        self.cat_data = df[cat_cols].values
        self.y = df[target_col].values.astype(float32)

        self.num_data = torch.tensor(self.num_data, dtype = torch.float32)
        self.cat_data = torch.tensor(self.cat_data, dtype = torch.int32)
        
        # Targets: regression -> float, classification -> long
        if issubdtype(self.y.dtype, floating):
            self.y = torch.tensor(self.y, dtype = torch.float32).unsqueeze(1)
        else:
            self.y = torch.tensor(self.y, dtype = torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.num_data[idx], self.cat_data[idx], self.y[idx]
    
@dataclass
class RetailCalendarV2Config:
    locale: str = "US"
    include_core: bool = True
    include_country: bool = True
    add_windows: Optional[dict[str, tuple[int, int]]] = None
    extra_events: Optional[list[tuple[str, str]]] = None

    def __post_init__(self):
        if self.add_windows is None:
            self.add_windows = {
                "BlackFriday": (14, 2), "CyberMonday": (0, 6), "Christmas": (21, 0),
                "ThanksgivingUS": (3, 3), "NewYearDay": (0, 1), "Halloween": (7, 0)
            }
            
@dataclass
class RetailEvent:
    """A simple container to hold the name and date of a retail event."""
    name: str
    dt: date
    
class RetailCalendarV2:
    """
    Builds a feature-rich calendar DataFrame for retail forecasting.
    It includes holidays, special events, and windows around these events. 
    This could be used in the future to add more events.
    """
    def __init__(self, config: RetailCalendarV2Config):
        self.cfg = config

    def _get_events_for_year(self, y: int) -> list[RetailEvent]:
        events = []
        
        # Core events
        if self.cfg.include_core:
            events.extend([RetailEvent("NewYearDay", date(y, 1, 1)), RetailEvent("ValentinesDay", date(y, 2, 14)), 
                           RetailEvent("Christmas", date(y, 12, 25))])
        
        # Check if country is US to add special days to the list of events
        if self.cfg.include_country and self.cfg.locale.upper() == "US":
            tg = nth_weekday_of_month(y, 11, weekday = 3, n = 4)
            events.extend([
                RetailEvent("MemorialDay", last_weekday_of_month(y, 5, weekday=0)),
                RetailEvent("IndependenceDay", date(y, 7, 4)),
                RetailEvent("LaborDay", nth_weekday_of_month(y, 9, weekday = 0, n = 1)),
                RetailEvent("Halloween", date(y, 10, 31)),
                RetailEvent("ThanksgivingUS", tg),
                RetailEvent("BlackFriday", tg + timedelta(days = 1)),
                RetailEvent("CyberMonday", tg + timedelta(days = 4)),
                RetailEvent("EasterSunday", easter_sunday(y)),
            ])
        return events

    def build(self, start: Timestamp, end: Timestamp) -> DataFrame:
        years = range(start.year, end.year + 1)
        
        # Get a list with all the events for the given period of years
        all_events = [ev for y in years for ev in self._get_events_for_year(y)]
        
        # Check if there are some extra events to add to the list of all the events
        if self.cfg.extra_events:
            all_events.extend([RetailEvent(name, to_datetime(s).date()) for name, s in self.cfg.extra_events])
        
        cal = DataFrame({"DATE": date_range(start, end, freq="D")})
        if not cal.empty:
            cal["DATE_ONLY"] = cal["DATE"].dt.date
            
            # Get a list with the unique name of the events
            unique_event_names = sorted(list({e.name for e in all_events}))
            
            # Create one column for each event
            for name in unique_event_names:
                cal[f"cal_is_{name}"] = 0
            
            # Fill with 1 in the row corresponding to the date of the event
            for ev in all_events:
                if start.date() <= ev.dt <= end.date():
                    cal.loc[cal["DATE_ONLY"] == ev.dt, f"cal_is_{ev.name}"] = 1
            
            # Add windows corresponding to a given event
            for ev_name, (lead, lag) in self.cfg.add_windows.items():
                col = f"cal_in_window_{ev_name}"
                cal[col] = 0
                ev_dates = [e.dt for e in all_events if e.name == ev_name]
                for ed in ev_dates:
                    # The mask includes the extremes of the window
                    mask = (cal["DATE_ONLY"] >= ed - timedelta(days = lead)) & (cal["DATE_ONLY"] <= ed + timedelta(days = lag))
                    cal.loc[mask, col] = 1
                    
            cal.drop(columns = ["DATE_ONLY"], inplace=True)
        return cal
    
@dataclass
class TrainConfig:
    """Holds all the hyperparameters and settings for the model training process."""
    n_iter: int = 25
    n_splits: int = 4
    min_train: int = 120
    val_size: int = 28
    seed: int = 42
    max_rounds: int = 3000
    early_stop: int = 200
    decay_halflife_days: float = 120.0
    horizons: tuple[int, ...] = tuple(range(1, 8))
    
@dataclass
class ClusterConfig:
    """Configuration for the customer/store clustering part of the model."""
    k_range: tuple[int, int] = (4, 16)
    min_series_days: int = 90
    use_features: tuple[str, ...] = ("avg_sales", "std_sales", "cv", "trend60", "avg_price", "promo_rate", "elasticity28")

BASE_PARAMS = {"objective": "regression", "metric": "rmse", "boosting_type": "gbdt", 
               "verbosity": -1, "random_state": 42, "n_jobs": -1}

class StoreAwareForecastingV6:
    """
    An end-to-end forecasting system that clusters similar time series
    and trains specialized models for each cluster, plus a global fallback model.
    """
    def __init__(self, test_ids_file: str, output_folder: str, retail_locale: str, 
                 extra_events: Optional[list[tuple[str, str]]], cfg: TrainConfig, cluster_cfg: ClusterConfig):
        
        self.test_ids_file = test_ids_file
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok = True)
        self.cfg, self.cluster_cfg, self.retail_locale, self.extra_events = cfg, cluster_cfg, retail_locale, extra_events or []
        self.calendar, self.global_data, self.global_model = None, None, None
        self.store_perf, self.global_encoders = {}, {}
        self.cluster_assignments, self.cluster_models = {}, {}
        self.clusterer, self.cluster_scaler = None, None

    def _standardize_columns(self, df: DataFrame) -> DataFrame:
        """ Rename the columns of the input DataFrame. """
        
        return df.rename(columns = {
        "store_id": "STORE_ID", "Store_ID": "STORE_ID",
        "date": "DATE", "Date": "DATE",
        "sku": "SKU", "Sku": "SKU",
        "price": "PRICE",
        "total_sales": "TOTAL_SALES",
        "quantity": "QUANTITY",
        "transaction_id": "TRANSACTION_ID",
        "subgroup": "SUBGROUP",
        "store_subgroup_date_id": "STORE_SUBGROUP_DATE_ID",
        "product_name": "PRODUCT_NAME",
        "category": "CATEGORY",
        "group": "GROUP",
        "brand": "BRAND",
        "base_price": "BASE_PRICE",
        "initial_ticket_price": "INITIAL_TICKET_PRICE",
        "costos": "COST",
    })

    def load_historical_data(self) -> Optional[DataFrame]:
        
        transactions_file = "../data/initial_datasets/eci_transactions.csv"
        products_file = "../data/initial_datasets/eci_product_master.csv"
        stores_file = "../data/initial_datasets/eci_stores.csv"
        
        if not os.path.exists(transactions_file):
            print("Transaction data not found!"); return None
            
        tx = read_csv(transactions_file)
        pr = read_csv(products_file)
        st = read_csv(stores_file)
        tx, pr, st = self._standardize_columns(tx), self._standardize_columns(pr), self._standardize_columns(st)
        
        # Defensive type casting for merge keys
        st['STORE_ID'] = st['STORE_ID'].astype(str)
        tx['STORE_ID'] = tx['STORE_ID'].astype(str)
        pr['SKU'] = pr['SKU'].astype(str).str.lower()
        if 'sku' in pr.columns: pr['sku'] = pr['sku'].astype(str).str.lower()
        tx['SKU'] = tx['SKU'].astype(str).str.lower()

        # Use the last 2 years of data
        tx["DATE"] = to_datetime(tx["DATE"])
        tx = tx[tx["DATE"] >= tx["DATE"].max() - timedelta(days = 730)].copy()
        
        if "AVG_SALES_PRICE" not in tx.columns:
            if "TOTAL_SALES" in tx.columns and "QUANTITY" in tx.columns:
                tx["AVG_SALES_PRICE"] = (tx["TOTAL_SALES"]/tx["QUANTITY"].replace(0, nan)).fillna(method = "ffill").fillna(0.0)
                
            else: tx["AVG_SALES_PRICE"] = 0.0
            
        if not pr.empty:
            if "subgroup" in pr.columns:
                pr["subgroup"] = pr["subgroup"].astype(str)
                tx = tx.merge(pr[["SKU", "subgroup"]].drop_duplicates(), on = "SKU", how = "left")
                
        if 'subgroup' not in tx.columns: tx['subgroup'] = 'unknown'
        
        if not st.empty: tx = tx.merge(st[["STORE_ID", "STORE_TYPE", "REGION"]].drop_duplicates(), on = "STORE_ID", how = "left")
        
        for col in ["COMPETITION_EFFECT", "SUBSTITUTION_EFFECT"]:
            if col not in tx.columns: tx[col] = 0.0
        
        grp = ["DATE", "STORE_ID", "subgroup"]
        daily = tx.groupby(grp).agg(
            TOTAL_SALES = ("TOTAL_SALES", "sum"), AVG_SALES_PRICE = ("AVG_SALES_PRICE", "mean"),
            STORE_TYPE = ("STORE_TYPE", "first"), REGION = ("REGION", "first"),
        ).reset_index()
        daily["subgroup"] = daily["subgroup"].fillna('unknown').astype(str)
        
        if daily.empty:
            print("Warning: No data after processing and aggregation.")
            return None
        
        print("Historical data loaded successfully.")
        return daily

    def load_test_ids(self) -> DataFrame:
        """ Loads and redefine the test_ids dataset """
        
        df = read_csv(self.test_ids_file)
        df[["STORE", "SUBGROUP", "DATE"]] = df["STORE_SUBGROUP_DATE_ID"].str.split("_", expand = True)
        df["DATE"] = to_datetime(df["DATE"])
        df['STORE'] = df['STORE'].astype(str)
        return df

    def build_calendar(self, start: Timestamp, end: Timestamp):
        """ Initializes and builds the retail calendar for the full date range of our data. """
        
        cfg = RetailCalendarV2Config(locale = self.retail_locale, extra_events = self.extra_events)
        self.calendar = RetailCalendarV2(cfg).build(start, end)

    def analyze_store_performance(self, df: DataFrame):
        """ Calculates some basic performance stats for each series, used for fallbacks. """
        
        g = df.groupby(["STORE_ID", "subgroup"]).agg(
            avg_sales = ("TOTAL_SALES", "mean"), sales_std = ("TOTAL_SALES", "std"), data_points = ("TOTAL_SALES", "size"),
        ).reset_index()
        self.store_perf = g.set_index(["STORE_ID", "subgroup"]).to_dict("index")

    @staticmethod
    def _fourier_terms(ts: Timestamp, period: int, K: int = 2) -> dict[str, float]:
        """ Generates Fourier terms (sine/cosine) to help the model capture seasonality. """
        
        day = ts.dayofyear
        out = {}
        for k in range(1, K + 1):
            out[f"sin_{period}_{k}"] = sin(2*pi*k*day/period)
            out[f"cos_{period}_{k}"] = cos(2*pi*k*day/period)
        return out

    def _add_price_features(self, df: DataFrame) -> DataFrame:
        """
        Adds price-related features, including a rough calculation of price elasticity
        over a rolling 28-day window.
        """
        
        df = df.sort_values("DATE").copy()
        df["log_price"] = log1p(df["AVG_SALES_PRICE"])
        e_vals = zeros(len(df))
        q, p = df["TOTAL_SALES"].astype(float), df["AVG_SALES_PRICE"].astype(float)
        for i in range(len(df)):
            start, end = max(0, i - 27), i + 1
            w_q, w_p = q.iloc[start:end], p.iloc[start:end]
            mask = (w_q > 0) & (w_p > 0)
            if mask.sum() >= 10:
                y, x = log(w_q[mask].clip(lower = 1e-3).values), log(w_p[mask].clip(lower = 1e-3).values)
                try: e_vals[i] = polyfit(x, y, 1)[0]
                except Exception: pass
        df["elasticity_28d"] = e_vals
        
        return df

    def create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        HEre we create the main feature engineering pipeline for a single time series.
        lags, rolling averages, date components, and calendar features
        """
        df = df.sort_values("DATE").reset_index(drop=True).copy()
        df["SALES_F"] = winsorize_series(df["TOTAL_SALES"])
        df["day_of_week"], df["month"], df["year"] = df["DATE"].dt.dayofweek, df["DATE"].dt.month, df["DATE"].dt.year
        for lag in [1, 7, 14, 28]: df[f"sales_lag_{lag}"] = df["SALES_F"].shift(lag)
        for w in [7, 14, 28]: df[f"sales_ma_{w}"] = df["SALES_F"].rolling(w, min_periods=1).mean()
        f7 = df["DATE"].apply(lambda ts: self._fourier_terms(ts, 7, K=2))
        f365 = df["DATE"].apply(lambda ts: self._fourier_terms(ts, 365, K=2))
        df = pd.concat([df, pd.DataFrame(list(f7)).reset_index(drop=True)], axis=1)
        df = pd.concat([df, pd.DataFrame(list(f365)).reset_index(drop=True)], axis=1)
        df = self._add_price_features(df)
        if self.calendar is not None:
            df = df.merge(self.calendar, on="DATE", how="left")
            for c in self.calendar.columns:
                if c != "DATE": df[c] = df[c].fillna(0)
        return df
    
    def _feature_columns(self, df_cols: list[str]) -> list[str]:
        base = [c for c in df_cols if "lag" in c or "ma" in c or c.startswith("sin_") or c.startswith("cos_") or c in ["day_of_week", "month", "year"]]
        price = ["log_price", "elasticity_28d"]
        cal = [c for c in df_cols if c.startswith("cal_")]
        final_features = base + price + cal + ["store_type_encoded", "region_encoded", "horizon"]
        return sorted(list(set(final_features)))

    def _prepare_multi_horizon(self, df_feat: pd.DataFrame, horizons: tuple[int, ...]):
        """
        Transforms the data for a direct multi-horizon forecasting strategy.
        It creates a separate training example for each forecast horizon (e.g., 1-day ahead, 2-days ahead, etc.)
        """
        frames = []
        for h in horizons:
            tmp = df_feat.copy()
            tmp["target"] = np.log1p(tmp["TOTAL_SALES"].shift(-h))
            tmp["horizon"] = h
            frames.append(tmp.dropna(subset=["target"]))
        all_df = pd.concat(frames, ignore_index=True).sort_values(["DATE", "horizon"])
        feature_columns = self._feature_columns(all_df.columns)
        X = all_df[feature_columns].fillna(0)
        return X, all_df["target"], all_df["DATE"], feature_columns

    def _encode_store_meta(self, df: pd.DataFrame, enc_store: LabelEncoder, enc_region: LabelEncoder) -> pd.DataFrame:
        df["STORE_TYPE"], df["REGION"] = df["STORE_TYPE"].fillna("Unknown").astype(str), df["REGION"].fillna("Unknown").astype(str)
        st_classes, rg_classes = set(enc_store.classes_), set(enc_region.classes_)
        st_vals = [v if v in st_classes else "Unknown" for v in df["STORE_TYPE"]]
        rg_vals = [v if v in rg_classes else "Unknown" for v in df["REGION"]]
        df["store_type_encoded"], df["region_encoded"] = enc_store.transform(st_vals), enc_region.transform(rg_vals)
        return df

    def _series_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a set of summary statistics for each time series.
        These descriptors are then used as features for the clustering algorithm
        """
        rows = []
        for (sid, sg), g in df.groupby(["STORE_ID", "subgroup"], sort=False):
            n, g = len(g), g.sort_values("DATE")
            if n < self.cluster_cfg.min_series_days: continue
            avg_sales, std_sales = g["TOTAL_SALES"].mean(), g["TOTAL_SALES"].std()
            recent = g.tail(60)
            slope = np.polyfit(np.arange(len(recent)), recent["TOTAL_SALES"].values, 1)[0] if len(recent) > 1 else 0.0
            ref_price = g["AVG_SALES_PRICE"].median() if "AVG_SALES_PRICE" in g.columns and not g["AVG_SALES_PRICE"].isnull().all() else 0.0
            promo_rate = (g["AVG_SALES_PRICE"] < 0.95 * ref_price).mean() if ref_price > 0 else 0.0
            anchor_e = self._add_price_features(g.tail(28).copy())
            elasticity28 = anchor_e["elasticity_28d"].iloc[-1] if not anchor_e.empty else 0.0
            rows.append({"STORE_ID": str(sid), "subgroup": str(sg), "n_days": n, "avg_sales": avg_sales, "std_sales": std_sales,
                "cv": std_sales / avg_sales if avg_sales > 0 else 0.0, "trend60": slope, "avg_price": g["AVG_SALES_PRICE"].mean(),
                "promo_rate": promo_rate, "elasticity28": elasticity28})
        return pd.DataFrame(rows)

    def fit_clusterer(self, df_all: pd.DataFrame):
        """
        Groups similar time series together using KMeans clustering.
        It automatically selects the best number of clusters k using the silhouette score.
        """
        print("Fitting clusterer with dynamic k selection...")
        desc = self._series_descriptors(df_all)
        if desc.empty or len(desc) < max(self.cluster_cfg.k_range):
            print(f"Not enough series ({len(desc)}) for clustering. Skipping."); return
        X = desc[list(self.cluster_cfg.use_features)].fillna(0.0).values
        self.cluster_scaler = StandardScaler()
        Xs = self.cluster_scaler.fit_transform(X)
        best_score, best_k = -1, self.cluster_cfg.k_range[0]
        for k in range(*self.cluster_cfg.k_range):
            clusterer = KMeans(n_clusters=k, n_init='auto', random_state=self.cfg.seed)
            labels = clusterer.fit_predict(Xs)
            score = silhouette_score(Xs, labels)
            print(f"For k={k}, silhouette score is {score:.4f}")
            if score > best_score: best_score, best_k = score, k
        print(f"Optimal k selected: {best_k} with score {best_score:.4f}")
        self.clusterer = KMeans(n_clusters=best_k, n_init='auto', random_state=self.cfg.seed)
        desc["cluster"] = self.clusterer.fit_predict(Xs)
        self.cluster_assignments = {(r["STORE_ID"], r["subgroup"]): int(r["cluster"]) for _, r in desc.iterrows()}

    def train_global_and_cluster_models(self, df_all: pd.DataFrame):
        """
        Orchestrates the main training process:
        1. Tunes hyperparameters on the full dataset.
        2. Trains a specialized model for each identified cluster.
        3. Trains a global model on all data as a fallback.
        """
        print("Generating features for all series...")
        feat_parts = [self.create_store_features(g) for _, g in df_all.groupby(["STORE_ID", "subgroup"])]
        if not feat_parts:
            print("No features generated, aborting training.")
            return
        feat = pd.concat(feat_parts, ignore_index=True)
        feat["STORE_TYPE"], feat["REGION"] = feat["STORE_TYPE"].fillna("Unknown").astype(str), feat["REGION"].fillna("Unknown").astype(str)
        self.global_encoders["store_type"] = LabelEncoder().fit(feat["STORE_TYPE"].unique().tolist() + ["Unknown"])
        self.global_encoders["region"] = LabelEncoder().fit(feat["REGION"].unique().tolist() + ["Unknown"])
        feat = self._encode_store_meta(feat, self.global_encoders["store_type"], self.global_encoders["region"])
        
        print("Tuning hyperparameters on global data...")
        X, y, dates, f_cols = self._prepare_multi_horizon(feat, self.cfg.horizons)
        tuned = tune_lgb_random_search(X, y, dates, self.cfg)
        tuned_params = tuned[0] if tuned else {}
        if tuned: print(f"Global tuning complete. Best mean R2: {tuned[1]['r2_mean']:.3f}")
        
        clusters = sorted(list(set(self.cluster_assignments.values())))
        print(f"Training {len(clusters)} cluster models...")
        for cid in clusters:
            keys = {k for k, c in self.cluster_assignments.items() if c == cid}
            cluster_feat_idx = feat.set_index(['STORE_ID', 'subgroup']).index.map(lambda x: (str(x[0]), str(x[1]))).isin(keys)
            X_c, y_c, d_c, _ = self._prepare_multi_horizon(feat[cluster_feat_idx], self.cfg.horizons)
            if X_c.empty: print(f"Skipping cluster {cid}, no data."); continue
            tail_c = max(28, int(len(X_c) * 0.1))
            X_tr_c, y_tr_c = X_c.iloc[:-tail_c], y_c.iloc[:-tail_c]
            w_tr_c = time_decay_weights(d_c.iloc[:-tail_c], self.cfg.decay_halflife_days)
            X_va_c, y_va_c = X_c.iloc[-tail_c:], y_c.iloc[-tail_c:]
            model_c = lgb.train({**BASE_PARAMS, **tuned_params}, lgb.Dataset(X_tr_c, label=y_tr_c, weight=w_tr_c),
                num_boost_round=self.cfg.max_rounds, valid_sets=[lgb.Dataset(X_va_c, label=y_va_c)],
                callbacks=[lgb.early_stopping(self.cfg.early_stop), lgb.log_evaluation(1000)])
            self.cluster_models[cid] = {"model": model_c, "feature_columns": f_cols}
            print(f"Cluster {cid} model trained.")
        
        print("Training global model...")
        tail_size = max(28 * 5, int(len(X) * 0.05))
        X_tr, y_tr, w_tr = X.iloc[:-tail_size], y.iloc[:-tail_size], time_decay_weights(dates.iloc[:-tail_size], self.cfg.decay_halflife_days)
        X_va, y_va = X.iloc[-tail_size:], y.iloc[-tail_size:]
        model = lgb.train({**BASE_PARAMS, **tuned_params}, lgb.Dataset(X_tr, label=y_tr, weight=w_tr),
            num_boost_round=self.cfg.max_rounds, valid_sets=[lgb.Dataset(X_va, label=y_va)],
            callbacks=[lgb.early_stopping(self.cfg.early_stop), lgb.log_evaluation(1000)])
        self.global_model = {"model": model, "feature_columns": f_cols}
        print("Global model trained.")
        
    def _features_for_anchor(self, hist: pd.DataFrame) -> pd.DataFrame:
        """Generates the feature set for the very last data point of a series"""
        return self.create_store_features(hist).tail(1)
        
    def _fallback_predict(self, store: str, subgroup: str) -> float:
        """
        Provides a simple prediction when a proper model forecast isn't possible
        (e.g., for a brand new store with no history).
        """
        key = (store, subgroup)
        if key in self.store_perf: return self.store_perf[key]["avg_sales"]
        mean_val = self.global_data[self.global_data["subgroup"] == subgroup]["TOTAL_SALES"].mean()
        return mean_val if pd.notna(mean_val) else self.global_data["TOTAL_SALES"].mean()

    def train_all(self):
        """Runs the complete training pipeline from data loading to model training"""
        print("TRAINING")
        self.global_data = self.load_historical_data()
        if self.global_data is None or self.global_data.empty: return
        test_ids = self.load_test_ids()
        
        start_date_hist = self.global_data["DATE"].min()
        end_date_hist = self.global_data["DATE"].max()
        start_date_test = test_ids["DATE"].min()
        end_date_test = test_ids["DATE"].max()
        
        start = min(d for d in [start_date_hist, start_date_test] if pd.notna(d))
        end = max(d for d in [end_date_hist, end_date_test] if pd.notna(d))

        if pd.isna(start) or pd.isna(end):
            print("Could not determine a valid date range. Aborting."); return

        self.build_calendar(start, end)
        self.analyze_store_performance(self.global_data)
        self.fit_clusterer(self.global_data)
        self.train_global_and_cluster_models(self.global_data)
        print("\nTraining complete. Models are ready for prediction.")
        
    def predict_for_test_period(self):
        """
        Generates predictions for all the IDs in the test file and saves them to a CSV.
        """
        print("PREDICTING")
        test_ids = self.load_test_ids()
        preds = []
        for (store, subgroup), group_df in test_ids.groupby(["STORE", "SUBGROUP"]):
            hist = self.global_data[(self.global_data["STORE_ID"] == store) & (self.global_data["subgroup"] == subgroup)]
            if hist.empty:
                for _, row in group_df.iterrows():
                    preds.append({"STORE_SUBGROUP_DATE_ID": row["STORE_SUBGROUP_DATE_ID"], "TOTAL_SALES": self._fallback_predict(store, subgroup)})
                continue
            # The "anchor" is the last day of historical data which we predict from.
            t0, f_anchor, key = hist["DATE"].max(), self._features_for_anchor(hist), (store, subgroup)
            for _, row in group_df.iterrows():
                h = max(1, min(7, (pd.Timestamp(row["DATE"]) - t0).days))
                f_anchor_h = f_anchor.copy()
                f_anchor_h["horizon"] = h
                cid = self.cluster_assignments.get(key)
                model_info = self.cluster_models.get(cid) if cid is not None and cid in self.cluster_models else self.global_model
                if not model_info: pred = self._fallback_predict(store, subgroup)
                else:
                    f_pred = self._encode_store_meta(f_anchor_h, self.global_encoders["store_type"], self.global_encoders["region"])
                    X = f_pred.reindex(columns=model_info["feature_columns"], fill_value=0)
                    p_log = model_info["model"].predict(X, num_iteration=model_info["model"].best_iteration)[0]
                    pred = np.expm1(p_log)
                preds.append({"STORE_SUBGROUP_DATE_ID": row["STORE_SUBGROUP_DATE_ID"], "TOTAL_SALES": max(0.1, float(pred))})
        dfp = pd.DataFrame(preds)
        out = f"{self.output_folder}/competition_predictions_v6_final.csv"
        dfp.to_csv(out, index=False)
        print(f"Predictions saved to: {out}")
        return dfp
# =============================================================================================== #
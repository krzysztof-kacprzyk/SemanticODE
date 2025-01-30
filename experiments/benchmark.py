import optuna
import os
import json
import pickle
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import torch
from scipy.interpolate import BSpline
import pandas as pd
import time
import copy
from datetime import datetime
from semantic_odes.api import BSplineBasisFunctions, PolynomialBasisFunctions, SemanticODE, create_full_composition_library, CompositionMap
import pysindy as ps
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import importlib
import sympy
from experiments.neural_ode import NeuralODERegressor
from experiments.deeponet import DeepONet
from experiments.neural_laplace import NeuralLaplaceRegressor
import torch.nn as nn

def mean_rmse(y_true_list, y_pred_list):
    rmse_scores = [np.sqrt(np.mean((y_true_i - y_pred_i)**2)) for y_true_i, y_pred_i in zip(y_true_list, y_pred_list)]
    return np.mean(rmse_scores)
def std_rmse(y_true_list, y_pred_list):
    rmse_scores = [np.sqrt(np.mean((y_true_i - y_pred_i)**2)) for y_true_i, y_pred_i in zip(y_true_list, y_pred_list)]
    return np.std(rmse_scores)

INF = 1.0e9


def interpolate_nans(y):
    """
    Interpolate np.nan values in a one-dimensional numpy array using linear interpolation.
    If interpolation is impossible, return an array of zeros.

    Parameters:
    y (numpy.ndarray): Input one-dimensional array with possible np.nan values.

    Returns:
    numpy.ndarray: Array with np.nan values imputed or zeros if interpolation is impossible.
    """
    threshold = 10
    y = np.where(np.abs(y) > threshold, np.nan, y)
    y = np.where(np.abs(y) < -threshold, np.nan, y)
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(len(y))
    mask = np.isnan(y)
    valid = ~mask

    # Check if interpolation is possible
    if np.count_nonzero(valid) < 2:
        if np.count_nonzero(valid) == 1:
            y[mask] = y[valid][0]
            return y
        else:
            return np.zeros_like(y)

    # Perform linear interpolation
    f = interp1d(
        x[valid],
        y[valid],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate')
    
    y[mask] = f(x[mask])
    return y



def clip_to_finite(y):
    # Fill np.nan values with INF
    y = np.where(np.isnan(y), INF, y)
    return np.clip(y,-INF,INF)

class YNormalizer:
    """Normalize y values."""

    def __init__(self):
        self.fitted = False

    def fit(self, ys):
        """Fit normalization parameters."""
        if self.fitted:
            raise RuntimeError('Already fitted.')
        if isinstance(ys, list):
            Y = np.concatenate(ys, axis=0)
        else:
            Y = ys
        self.y_mean = np.mean(Y)
        self.y_std = np.std(Y)
        self.fitted = True

    def transform(self, ys):
        """Normalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [(y - self.y_mean) / self.y_std for y in ys]
        else:
            return (ys - self.y_mean) / self.y_std
    
    def inverse_transform(self, ys):
        """Denormalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [y * self.y_std + self.y_mean for y in ys]
        else:
            return ys * self.y_std + self.y_mean
    
    def save(self, path):
        """Save normalization parameters using json"""
        y_normalization = {'y_mean': self.y_mean, 'y_std': self.y_std}
        full_path = os.path.join(path, "y_normalizer.json")
        with open(full_path, 'w') as f:
            json.dump(y_normalization, f)
    
    def load(path):
        """Load normalization parameters using json"""
        with open(path, 'r') as f:
            y_normalization = json.load(f)
        ynormalizer = YNormalizer()
        ynormalizer.set_params(y_normalization['y_mean'], y_normalization['y_std'])
        return ynormalizer

    def load_from_benchmark(timestamp, name, benchmark_dir='benchmarks'):
        """Load normalization parameters from a benchmark."""
        path = os.path.join(benchmark_dir, timestamp, name, 'y_normalizer.json')
        return YNormalizer.load(path)

    def set_params(self, y_mean, y_std):
        """Set normalization parameters."""
        self.y_mean = y_mean
        self.y_std = y_std
        self.fitted = True

    def fit_transform(self, ys):
        """Fit normalization parameters and normalize y values."""
        self.fit(ys)
        return self.transform(ys)
    

class YStandardizer:
    """Standardize y values between 0 and 1."""

    def __init__(self):
        self.fitted = False

    def fit(self, ys):
        """Fit normalization parameters."""
        if self.fitted:
            raise RuntimeError('Already fitted.')
        if isinstance(ys, list):
            Y = np.concatenate(ys, axis=0)
        else:
            Y = ys
        self.y_max = np.max(Y)
        self.y_min = np.min(Y)
        self.fitted = True

    def transform(self, ys):
        """Normalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [(y - self.y_min) / (self.y_max - self.y_min) for y in ys]
        else:
            return (ys - self.y_min) / (self.y_max - self.y_min)
    
    def inverse_transform(self, ys):
        """Denormalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [y * (self.y_max - self.y_min) + self.y_min for y in ys]
        else:
            return ys * (self.y_max - self.y_min) + self.y_min
    
    def save(self, path):
        """Save normalization parameters using json"""
        y_normalization = {'y_mean': self.y_mean, 'y_std': self.y_std}
        full_path = os.path.join(path, "y_standardizer.json")
        with open(full_path, 'w') as f:
            json.dump(y_normalization, f)
    
    def load(path):
        """Load normalization parameters using json"""
        with open(path, 'r') as f:
            y_standardizer_details = json.load(f)
        y_standardizer = YStandardizer()
        y_standardizer.set_params(y_standardizer_details['y_mean'], y_standardizer_details['y_std'])
        return y_standardizer

    def load_from_benchmark(timestamp, name, benchmark_dir='benchmarks'):
        """Load normalization parameters from a benchmark."""
        path = os.path.join(benchmark_dir, timestamp, name, 'y_standardizer.json')
        return YStandardizer.load(path)

    def set_params(self, y_max, y_min):
        """Set normalization parameters."""
        self.y_max = y_max
        self.y_min = y_min
        self.fitted = True

    def fit_transform(self, ys):
        """Fit normalization parameters and normalize y values."""
        self.fit(ys)
        return self.transform(ys)



def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def make_json_serializable(dictionary):
    if is_json_serializable(dictionary):
        return dictionary
    else:
        for key, value in dictionary.items():
            if is_json_serializable(value):
                continue
            elif isinstance(value, dict):
                dictionary[key] = make_json_serializable(value)
            else:
                dictionary[key] = {
                    'class': value.__class__.__name__,
                    'value': make_json_serializable(value.__dict__) if hasattr(value, '__dict__') else str(value)
                }
    return dictionary

def generate_indices(n, train_size, val_size, seed=0):
    gen = np.random.default_rng(seed)
    train_indices = gen.choice(n, int(n*train_size), replace=False)
    train_indices = [i.item() for i in train_indices]
    val_indices = gen.choice(list(set(range(n)) - set(train_indices)), int(n*val_size), replace=False)
    val_indices = [i.item() for i in val_indices]
    test_indices = list(set(range(n)) - set(train_indices) - set(val_indices))
    return train_indices, val_indices, test_indices

def run_benchmark_random_splits(dataset, method, dataset_split = [0.7,0.15,0.15], n_trials=10, n_tune=100, seed=0, benchmarks_dir='results', experiment_name='untitled'):
    """
    Runs a set of benchmarks on a dataset
    Args:
    """

    # Add a row to the DataFrame
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    experiment_dir = os.path.join(benchmarks_dir, experiment_name)

    # Check if there exists a file summary.json in the benchmarks directory
    if os.path.exists(os.path.join(experiment_dir, 'summary.csv')):
        # Load as a DataFrame
        summary = pd.read_csv(os.path.join(experiment_dir, 'summary.csv'))
    else:
        # Create folder if does not exist
        os.makedirs(experiment_dir, exist_ok=True)
        # Create
        summary = pd.DataFrame(columns=['timestamp', 'dataset', 'method', 'n_trials', 'n_tune', 'train_size', 'val_size', 'seed', 'test_loss_mean', 'test_loss_std', 'time_elapsed'])
        # Save the summary
        summary.to_csv(os.path.join(experiment_dir, 'summary.csv'), index=False)
        

   
    time_start = time.time()
    benchmark = method
    losses, single_run_time, model = benchmark.run_random_splits(dataset, dataset_split, n_trials=n_trials, n_tune=n_tune, seed=seed, experiment_dir=experiment_dir, timestamp=timestamp)
    time_end = time.time()

    # Add a new row to the summary
    results = {
        'timestamp': [timestamp],
        'dataset_name': [dataset.get_name()],
        'method': [method.get_name()],
        'n_trials': [n_trials],
        'n_tune': [n_tune],
        'train_size': [dataset_split[0]],
        'val_size': [dataset_split[1]],
        'seed': [seed],
        'test_loss_mean': [np.mean(losses)],
        'test_loss_std': [np.std(losses)],
        'time_elapsed': time_end - time_start,
        'single_run_time': single_run_time
        }

    # concatenate the results to the summary
    summary = pd.concat([summary, pd.DataFrame(results)], ignore_index=True)
    
    # Save the summary
    summary.to_csv(os.path.join(experiment_dir, 'summary.csv'), index=False)

    return summary, model

def run_benchmark(dataset, method, dataset_split = [0.7,0.15,0.15], n_trials=10, n_tune=100, seed=0, benchmarks_dir='results', experiment_name='untitled'):
    """
    Runs a set of benchmarks on a dataset
    Args:
    """

    # Add a row to the DataFrame
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    experiment_dir = os.path.join(benchmarks_dir, experiment_name)

    # Check if there exists a file summary.json in the benchmarks directory
    if os.path.exists(os.path.join(experiment_dir, 'summary.csv')):
        # Load as a DataFrame
        summary = pd.read_csv(os.path.join(experiment_dir, 'summary.csv'))
    else:
        # Create folder if does not exist
        os.makedirs(experiment_dir, exist_ok=True)
        # Create
        summary = pd.DataFrame(columns=['timestamp', 'dataset', 'method', 'n_trials', 'n_tune', 'train_size', 'val_size', 'seed', 'test_loss_mean', 'test_loss_std', 'time_elapsed'])
        # Save the summary
        summary.to_csv(os.path.join(experiment_dir, 'summary.csv'), index=False)
        

    # Generate train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(len(dataset), dataset_split[0], dataset_split[1], seed=seed)
    
    time_start = time.time()
    benchmark = method
    losses, single_run_time, model = benchmark.run(dataset, train_indices, val_indices, test_indices, n_trials=n_trials, n_tune=n_tune, seed=seed, experiment_dir=experiment_dir, timestamp=timestamp)
    time_end = time.time()

    # Add a new row to the summary
    results = {
        'timestamp': [timestamp],
        'dataset_name': [dataset.get_name()],
        'method': [method.get_name()],
        'n_trials': [n_trials],
        'n_tune': [n_tune],
        'train_size': [dataset_split[0]],
        'val_size': [dataset_split[1]],
        'seed': [seed],
        'test_loss_mean': [np.mean(losses)],
        'test_loss_std': [np.std(losses)],
        'time_elapsed': time_end - time_start,
        'single_run_time': single_run_time
        }

    # concatenate the results to the summary
    summary = pd.concat([summary, pd.DataFrame(results)], ignore_index=True)
    
    # Save the summary
    summary.to_csv(os.path.join(experiment_dir, 'summary.csv'), index=False)

    return summary, model


class BaseBenchmark(ABC):
    """Base class for benchmarks."""

    def __init__(self):
        self.name = self.get_name()


    def tune(self, n_trials, seed, experiment_dir):
        """Tune the benchmark."""

        def objective(trial):
            model = self.get_model_for_tuning(trial, seed)
            val_loss = self.train(model, tuning=True)[1]['val_loss']
            print(f'[Trial {trial.number}] val_loss: {val_loss}')
            return val_loss
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler,direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_hyperparameters = best_trial.params

        print('[Best hyperparameter configuration]:')
        print(best_hyperparameters)

        tuning_dir = os.path.join(experiment_dir, self.name, self.timestamp, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)

        # Save best hyperparameters
        hyperparam_save_path = os.path.join(tuning_dir, f'hyperparameters.json')
        with open(hyperparam_save_path, 'w') as f:
            json.dump(best_hyperparameters, f)
        
        # Save optuna study
        study_save_path = os.path.join(tuning_dir, f'study_{seed}.pkl')
        with open(study_save_path, 'wb') as f:
            pickle.dump(study, f)

        # Save trials dataframe
        df = study.trials_dataframe()
        df.set_index('number', inplace=True)
        df_save_path = os.path.join(tuning_dir, f'trials_dataframe.csv')
        df.to_csv(df_save_path)

        print(f'[Tuning complete], saved tuning results to {tuning_dir}')

        return best_hyperparameters
    
    def run_random_splits(self, dataset, dataset_split, n_trials, n_tune, seed, experiment_dir, timestamp, **kwargs):
        """Run the benchmark."""
        self.experiment_dir = experiment_dir
        self.timestamp = timestamp

        # Create a numpy random generator
        rng = np.random.default_rng(seed)

        # Generate seeds for training
        training_seeds = rng.integers(0, 2**31 - 1, size=1000)
        training_seeds = [s.item() for s in training_seeds[:n_trials]]

        print(f"[Testing for {n_trials} trials]")

        # Train the model n_trials times
        test_losses = []
        run_times = []
        for i in range(n_trials):
            if n_trials == 1:
                training_seed = seed
            else:
                training_seed = training_seeds[i]

            print(f"[Training trial {i+1}/{n_trials}] seed: {training_seed}")

            # Generate train, validation, and test indices for tuning
            train_indices, val_indices, test_indices = generate_indices(len(dataset), dataset_split[0], dataset_split[1], seed=training_seed)
        
            # Prepare the data
            self.prepare_data(dataset, train_indices, val_indices, test_indices)

            # Tune the model
            if n_tune > 0:
                print(f"[Tuning for {n_tune} trials]")
                best_hyperparameters = self.tune(n_trials=n_tune, seed=training_seed, experiment_dir=experiment_dir)
            else:
                print(f"[No tuning, using default hyperparameters]")
                best_hyperparameters = None

            model = self.get_final_model(best_hyperparameters, training_seed)
            
            start_time = time.time()
            model, results = self.train(model)
            test_loss = results['test_loss']
            end_time = time.time()
            print(f"[Test loss]: {test_loss}")
            test_losses.append(test_loss)
            run_times.append(end_time - start_time)

        # Save the losses
        df = pd.DataFrame({'seed':training_seeds,'test_loss': test_losses})
        results_folder = os.path.join(experiment_dir, self.name, self.timestamp, 'final')
        os.makedirs(results_folder, exist_ok=True)
        test_losses_save_path = os.path.join(results_folder, f'results.csv')
        df.to_csv(test_losses_save_path, index=False)
        average_single_run_time = np.mean(run_times)
        return test_losses, average_single_run_time, model

    def run(self, dataset, train_indices, val_indices, test_indices, n_trials, n_tune, seed, experiment_dir, timestamp, **kwargs):
        """Run the benchmark."""
        self.experiment_dir = experiment_dir
        self.timestamp = timestamp

        # Create a numpy random generator
        rng = np.random.default_rng(seed)

        # Seed for tuning
        tuning_seed = seed

        # Generate seeds for training
        training_seeds = rng.integers(0, 2**31 - 1, size=n_trials)
        training_seeds = [s.item() for s in training_seeds]

        # Prepare the data
        self.prepare_data(dataset, train_indices, val_indices, test_indices)

        # Tune the model
        if n_tune > 0:
            print(f"[Tuning for {n_tune} trials]")
            best_hyperparameters = self.tune(n_trials=n_tune, seed=tuning_seed, experiment_dir=experiment_dir)
        else:
            print(f"[No tuning, using default hyperparameters]")
            best_hyperparameters = None

        print(f"[Training for {n_trials} trials with best hyperparameters]")

        # Train the model n_trials times
        test_losses = []
        run_times = []
        for i in range(n_trials):
            if n_trials == 1:
                print(f"[Training final model] seed: {seed}")
                model =self.get_final_model(best_hyperparameters, seed)
            else:
                print(f"[Training trial {i+1}/{n_trials}] seed: {training_seeds[i]}")
                model = self.get_final_model(best_hyperparameters, training_seeds[i])
            start_time = time.time()
            model, results = self.train(model)
            test_loss = results['test_loss']
            end_time = time.time()
            print(f"[Test loss]: {test_loss}")
            test_losses.append(test_loss)
            run_times.append(end_time - start_time)

        # Save the losses
        df = pd.DataFrame({'seed':training_seeds,'test_loss': test_losses})
        results_folder = os.path.join(experiment_dir, self.name, self.timestamp, 'final')
        os.makedirs(results_folder, exist_ok=True)
        test_losses_save_path = os.path.join(results_folder, f'results.csv')
        df.to_csv(test_losses_save_path, index=False)
        average_single_run_time = np.mean(run_times)
        return test_losses, average_single_run_time, model

    def prepare_data(self, dataset, train_indices, val_indices, test_indices):
        """Prepare the data for the benchmark."""
        X, T, Y = dataset.get_X_T_Y()

        self.X_train = X[train_indices,:]
        self.T_train = T[train_indices,:]
        self.Y_train = Y[train_indices,:]

        self.X_val = X[val_indices,:]
        self.T_val = T[val_indices,:]
        self.Y_val =  Y[val_indices,:]

        self.X_test = X[test_indices,:]
        self.T_test = T[test_indices,:]
        self.Y_test = Y[test_indices,:]

        self.X_train_val = np.concatenate([self.X_train, self.X_val], axis=0)
        self.T_train_val = np.concatenate([self.T_train, self.T_val], axis=0)
        self.Y_train_val = np.concatenate([self.Y_train, self.Y_val], axis=0)


    @abstractmethod
    def train(self, model, tuning=False):
        """
        Train the benchmark. Returns a dictionary with train, validation, and test loss
        Returns:
            dict: Dictionary with train, validation, and test loss {'train_loss': float, 'val_loss': float, 'test_loss': float}
        """
        pass
       
    @abstractmethod
    def get_model_for_tuning(self, trial, seed):
        """Get the model."""
        pass

    @abstractmethod
    def get_final_model(self, hyperparameters, seed):
        """Get the model."""
        pass
    
    @abstractmethod
    def get_name(self):
        """Get the name of the benchmark."""
        pass

class SemanticODEBenchmark(BaseBenchmark):
    """SketchODE"""

    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'SemanticODE-'+self.config['subtype']
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        opt_config = copy.deepcopy(self.config['opt_config'])
        
        opt_config['lr'] = trial.suggest_loguniform('lr', 1e-3, 1.0)
        opt_config['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
        opt_config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])

        basis_function_type = trial.suggest_categorical('basis_function_type', ['polynomial', 'bspline'])

        if basis_function_type == 'polynomial':
            num_basis_functions = trial.suggest_int('num_basis_functions', 1, 4)
            basis_functions = PolynomialBasisFunctions(n_basis=num_basis_functions)
        elif basis_function_type == 'bspline':
            num_basis_functions = trial.suggest_int('num_basis_functions', 4, 8)
            basis_functions = BSplineBasisFunctions(n_basis=num_basis_functions,include_bias=True,include_linear=True)
     
        max_branches = self.config['max_branches']

        composition_library = self.config['composition_library']
        model = SemanticODE(self.config['t_range'],basis_functions,composition_library,max_branches=max_branches,seed=seed,opt_config=opt_config)
        return model
       
    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        opt_config = copy.deepcopy(self.config['opt_config'])

        if parameters is not None:
            opt_config['lr'] = parameters['lr']
            opt_config['weight_decay'] = parameters['weight_decay']
            opt_config['batch_size'] = parameters['batch_size']
            basis_function_type = parameters['basis_function_type']
        else:
            basis_function_type = self.config['basis_function_type']

 
        max_branches = self.config['max_branches']
        
        if basis_function_type == 'polynomial':
            num_basis_functions = self.config['num_basis_functions']
            basis_functions = PolynomialBasisFunctions(n_basis=num_basis_functions)
        elif basis_function_type == 'bspline':
            num_basis_functions = self.config['num_basis_functions']
            basis_functions = BSplineBasisFunctions(n_basis=num_basis_functions,include_bias=True,include_linear=True)
        
        composition_library = self.config['composition_library']
        model = SemanticODE(self.config['t_range'],basis_functions,composition_library,max_branches=max_branches,seed=seed,opt_config=opt_config, verbose=self.config['verbose'])

        return model
        

    def train(self, model, tuning=False):

        val_loss = None
        test_loss = None

        # We combine the training and validation data because we do not do hyperparameter tuning
        # We will use validation data for early stopping and validation throughout
        # model.fit(self.X_train_val,self.T_train_val,self.Y_train_val)

        


        if model.composition_map is not None:
            model.fit(self.X_train_val,self.T_train_val,self.Y_train_val, composition_map=model.composition_map)
        else:
            if len(model.composition_library) == 1:
                composition_map = CompositionMap([((-np.inf,np.inf),tuple(model.composition_library[0]))])
                model.fit(self.X_train_val,self.T_train_val,self.Y_train_val, composition_map=composition_map)
            else:
                model.fit(self.X_train_val,self.T_train_val,self.Y_train_val)

        normalize = False

        if tuning:
            y_val_pred = model.predict(self.X_val, self.T_val)
            if normalize:
                val_loss_per_sample = np.sqrt(np.mean((y_val_pred - self.Y_val) ** 2, axis=1))
                range_per_sample = np.max(self.Y_val, axis=1) - np.min(self.Y_val, axis=1)
                val_loss = np.mean(val_loss_per_sample / range_per_sample)
            else:
                val_loss = mean_rmse(self.Y_val, y_val_pred)

        if not tuning:
            y_test_pred = model.predict(self.X_test, self.T_test)
            if normalize:
                test_loss_per_sample = np.sqrt(np.mean((y_test_pred - self.Y_test) ** 2, axis=1))
                range_per_sample = np.max(self.Y_test, axis=1) - np.min(self.Y_test, axis=1)
                test_loss = np.mean(test_loss_per_sample / range_per_sample)
            else:
                test_loss = mean_rmse(self.Y_test, y_test_pred)

        return model, {'val_loss': val_loss, 'test_loss': test_loss}
    
    def evaluate(self, model):
        val_loss = None
        test_loss = None

        normalize = False

       
        y_val_pred = model.predict(self.X_val, self.T_val)
        if normalize:
            val_loss_per_sample = np.sqrt(np.mean((y_val_pred - self.Y_val) ** 2, axis=1))
            range_per_sample = np.max(self.Y_val, axis=1) - np.min(self.Y_val, axis=1)
            val_loss = np.mean(val_loss_per_sample / range_per_sample)
        else:
            val_loss = mean_rmse(self.Y_val, y_val_pred)
            val_std = std_rmse(self.Y_val, y_val_pred)

      
        y_test_pred = model.predict(self.X_test, self.T_test)
        if normalize:
            test_loss_per_sample = np.sqrt(np.mean((y_test_pred - self.Y_test) ** 2, axis=1))
            range_per_sample = np.max(self.Y_test, axis=1) - np.min(self.Y_test, axis=1)
            test_loss = np.mean(test_loss_per_sample / range_per_sample)
        else:
            test_loss = mean_rmse(self.Y_test, y_test_pred)
            test_std = std_rmse(self.Y_test, y_test_pred)

        return model, {'val_loss': val_loss, 'test_loss': test_loss, 'test_loss_std': test_std, 'val_loss_std': val_std}


    


class SINDyBenchmark(BaseBenchmark):
    """SINDy benchmark."""
    def __init__(self,sparsity=2):
        self.fitted = False
        self.library_type = 'general'
        self.threshold = 0.01
        self.sparsity = sparsity
        super().__init__()
    def get_name(self):
        return f'SINDy-{self.sparsity}'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""

        polynomial_library = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
        fourier_library = ps.FourierLibrary(n_frequencies=3,include_cos=False)
        exp_library = ps.CustomLibrary(library_functions=[lambda x : np.exp(x), lambda x : np.log(1+np.abs(x))], function_names=[lambda x: f"exp({x})", lambda x: f"log(1+{x})"])
        
        if self.library_type == 'general':
            fourier_full_library = ps.FourierLibrary(n_frequencies=3,include_cos=True,include_sin=True)
            exp_only_library = ps.CustomLibrary(library_functions=[lambda x : np.exp(x)], function_names=[lambda x: f"exp({x})"])
            library = ps.GeneralizedLibrary([polynomial_library, fourier_full_library, exp_only_library])
        elif self.library_type == 'polynomial':
            library = polynomial_library
        elif self.library_type == 'polynomial_2':
            polynomial = ps.PolynomialLibrary(degree=4, include_interaction=True, include_bias=True)
            library = ps.GeneralizedLibrary([polynomial, exp_library])

        # optimizer_type = trial.suggest_categorical('optimizer_type', ['stlsq', 'sr3', 'frols', 'ssr'])        
        # optimizer_type = 'stlsq'  
        optimizer_type = 'miosr'

        if optimizer_type == 'stlsq':
            # optimizer_threshold = trial.suggest_float('optimizer_threshold', 1e-3, 1, log=True)
            optimizer_threshold = self.threshold
            if self.alpha is None:
                alpha = trial.suggest_float('alpha_stlsq', 1e-3, 1, log=True)
            else:
                alpha = self.alpha
            optimizer = ps.STLSQ(threshold=optimizer_threshold,alpha=alpha,normalize_columns=True)
        elif optimizer_type == 'sr3':
            optimizer_threshold = trial.suggest_float('optimizer_threshold', 1e-3, 1, log=True)
            optimizer = ps.SR3(threshold=optimizer_threshold)
        elif optimizer_type == 'frols':
            alpha = trial.suggest_float('alpha', 1e-3, 1, log=True)
            optimizer = ps.FROLS(alpha=alpha)
        elif optimizer_type == 'ssr':
            alpha = trial.suggest_float('alpha', 1e-3, 1, log=True)
            optimizer = ps.SSR(alpha=alpha)
        elif optimizer_type == 'miosr':
            alpha = trial.suggest_float('alpha_miosr', 1e-3, 1, log=True)
            if self.sparsity == 0:
                sparsity = trial.suggest_int('sparsity', 1, 20)
            else:
                sparsity = self.sparsity
            optimizer = ps.MIOSR(alpha=alpha,target_sparsity=None,group_sparsity=(sparsity,1),normalize_columns=True)

        differentiation_kind = trial.suggest_categorical('differentiation_kind', ['finite_difference', 'spline', 'trend_filtered'])
        if differentiation_kind == 'finite_difference':
            k = trial.suggest_int('k', 1, 5)
            differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
        elif differentiation_kind == 'spline':
            s = trial.suggest_float('s', 1e-3, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
        elif differentiation_kind == 'trend_filtered':
            order = trial.suggest_int('order', 0, 2)
            alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
        elif differentiation_kind == 'smoothed_finite_difference':
            window_length = trial.suggest_int('window_length', 1, 5)
            differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})

        model = ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, feature_library=library)

        return model

    def get_final_model(self, parameters, seed):
        """Get model for testing."""

        polynomial_library = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
        fourier_library = ps.FourierLibrary(n_frequencies=3, include_cos=False)
        exp_library = ps.CustomLibrary(library_functions=[lambda x : np.exp(x), lambda x : np.log(1+np.abs(x))], function_names=[lambda x: f"exp({x})", lambda x: f"log(1+{x})"])

        if self.library_type == 'general':
            fourier_full_library = ps.FourierLibrary(n_frequencies=3,include_cos=True,include_sin=True)
            exp_only_library = ps.CustomLibrary(library_functions=[lambda x : np.exp(x)], function_names=[lambda x: f"exp({x})"])
            library = ps.GeneralizedLibrary([polynomial_library, fourier_full_library, exp_only_library])
        elif self.library_type == 'polynomial':
            library = polynomial_library
        elif self.library_type == 'polynomial_2':
            polynomial = ps.PolynomialLibrary(degree=4, include_interaction=True, include_bias=True)
            library = ps.GeneralizedLibrary([polynomial, exp_library])

        if parameters is None:
            return ps.SINDy(optimizer=ps.STLSQ(threshold=1e-3), differentiation_method=ps.SINDyDerivative(kind='finite_difference', k=2), feature_library=library)
        else:
            # optimizer_type = parameters['optimizer_type']
            # optimizer_type = 'stlsq'
            optimizer_type = 'miosr'
            if optimizer_type == 'stlsq':
                # optimizer_threshold = parameters['optimizer_threshold']
                optimizer_threshold = self.threshold
                if self.alpha is None:
                    alpha = parameters['alpha_stlsq']
                else:
                    alpha = self.alpha
                optimizer = ps.STLSQ(threshold=optimizer_threshold, alpha=alpha, normalize_columns=True)
            elif optimizer_type == 'sr3':
                optimizer_threshold = parameters['optimizer_threshold']
                optimizer = ps.SR3(threshold=optimizer_threshold)
            elif optimizer_type == 'frols':
                alpha = parameters['alpha']
                optimizer = ps.FROLS(alpha=alpha)
            elif optimizer_type == 'ssr':
                alpha = parameters['alpha']
                optimizer = ps.SSR(alpha=alpha)
            elif optimizer_type == 'miosr':
                alpha = parameters['alpha_miosr']
                if self.sparsity == 0:
                    sparsity = parameters['sparsity']
                else:
                    sparsity = self.sparsity
                optimizer = ps.MIOSR(alpha=alpha,target_sparsity=None,group_sparsity=(sparsity,1),normalize_columns=True)

           
            differentiation_kind = parameters['differentiation_kind']
            if differentiation_kind == 'finite_difference':
                k = parameters['k']
                differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
            elif differentiation_kind == 'spline':
                s = parameters['s']
                differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
            elif differentiation_kind == 'trend_filtered':
                order = parameters['order']
                alpha = parameters['alpha']
                differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
            elif differentiation_kind == 'smoothed_finite_difference':
                window_length = parameters['window_length']
                differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})

            return ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, feature_library=library)
    
    def train(self, model, tuning=False):
        """Train model."""

        INF = 1.0e9

            
        # Y needs to be a list of numpy arrays each of shape (n_measurements, 1)

        ys_train = [np.stack([y,t], axis=1) for y,t in zip(self.Y_train,self.T_train)]
        ts_train = [t.flatten() for t in self.T_train]
        xs_train = [x.item() for x in self.X_train]


        ys_val = [np.expand_dims(y, axis=1) for y in self.Y_val]
        ts_val = [t.flatten() for t in self.T_val]
        xs_val = [x.item() for x in self.X_val]

        ys_test = [np.expand_dims(y, axis=1) for y in self.Y_test]
        ts_test = [t.flatten() for t in self.T_test]
        xs_test = [x.item() for x in self.X_test]

        # print("Fitting model")
        try:
            model.fit(ys_train, t=ts_train, multiple_trajectories=True)
        except Exception as e:
            print(e)
            return {'train_loss': INF, 'val_loss': INF, 'test_loss': INF}
        # print("Model fitted")

        
        # train_losses = []
        # for X, t, y in zip(self.X_train, self.ts_train, self.ys_train):
        #     u = get_control_function(X)
        #     y_pred = model.simulate(y[[0]], t=t, u=u)
        #     train_losses.append(mean_squared_error(y, y_pred))
        # train_loss = np.mean(train_losses)
        
        
        normalize = False

        val_loss = 0
        if tuning:
            val_losses = []
            for x0, t, y in zip(xs_val, ts_val, ys_val):
                try:
                    y_pred = clip_to_finite(model.simulate(np.array([x0,0]), t=t, integrator='odeint'))[:,0]
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                if normalize:
                    sample_range = np.max(y) - np.min(y)
                    val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
                else:
                    val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
                
            val_loss = np.mean(val_losses)
        
        
        test_loss = 0
        if not tuning:
            test_losses = []
            for x0, t, y in zip(xs_test, ts_test, ys_test):
                try:
                    y_pred = clip_to_finite(model.simulate(np.array([x0,0]), t=t, integrator='odeint'))[:,0]
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                if normalize:
                    sample_range = np.max(y) - np.min(y)
                    test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
                else:
                    test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            test_loss = np.mean(test_losses)

        return model, {'val_loss': val_loss, 'test_loss': test_loss}
    
    def evaluate(self, model):

        INF = 1.0e9

        def clip_to_finite(y):
            return np.clip(y,-INF,INF)
        
        ys_val = [np.expand_dims(y, axis=1) for y in self.Y_val]
        ts_val = [t.flatten() for t in self.T_val]
        xs_val = [x.item() for x in self.X_val]

        ys_test = [np.expand_dims(y, axis=1) for y in self.Y_test]
        ts_test = [t.flatten() for t in self.T_test]
        xs_test = [x.item() for x in self.X_test]

        
        normalize = False

        val_loss = 0
        val_losses = []
        for x0, t, y in zip(xs_val, ts_val, ys_val):
            try:
                y_pred = clip_to_finite(model.simulate(np.array([x0,0]), t=t, integrator='odeint'))[:,0]
            except Exception as e:
                print(e)
                y_pred = np.zeros_like(y)
            if normalize:
                sample_range = np.max(y) - np.min(y)
                val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
            else:
                val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            
        val_loss = np.mean(val_losses)
        val_loss_std = np.std(val_losses)
        
        test_loss = 0
       
        test_losses = []
        for x0, t, y in zip(xs_test, ts_test, ys_test):
            try:
                y_pred = clip_to_finite(model.simulate(np.array([x0,0]), t=t, integrator='odeint'))[:,0]
            except Exception as e:
                print(e)
                y_pred = np.zeros_like(y)
            if normalize:
                sample_range = np.max(y) - np.min(y)
                test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
            else:
                test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
        test_loss = np.mean(test_losses)
        test_loss_std = np.std(test_losses)

        return model, {'val_loss': val_loss, 'test_loss': test_loss, 'test_loss_std': test_loss_std, 'val_loss_std': val_loss_std}


    

class WeakSINDyBenchmark(BaseBenchmark):
    """WSINDy benchmark."""
    def __init__(self, t_grid, sparsity=2):
        self.fitted = False
        self.t_grid = t_grid
        self.sparsity = sparsity
        super().__init__()
    
    def get_name(self):
        return f'WeakSINDy-{self.sparsity}'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""

        # library_functions = [lambda x: x, lambda x: x * x, lambda x: np.exp(x), lambda x: np.log(1+np.abs(x))]
        # library_functions += [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.sin(2*x), lambda x: np.cos(2*x), lambda x: np.sin(3*x), lambda x: np.cos(3*x)]
        # library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x: f"exp({x})", lambda x: f"log(1+{x})"]
        # library_function_names += [lambda x: f"sin({x})", lambda x: f"cos({x})", lambda x: f"sin(2{x})", lambda x: f"cos(2{x})", lambda x: f"sin(3{x})", lambda x: f"cos(3{x})"]

        # library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x*y, lambda x: np.exp(x), lambda x: np.log(1+np.abs(x))]
        library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x*y, lambda x: np.exp(x)]
        library_functions += [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.sin(2*x), lambda x: np.cos(2*x), lambda x: np.sin(3*x), lambda x: np.cos(3*x)]
        # library_functions += [lambda x: np.sin(x), lambda x: np.sin(2*x), lambda x: np.sin(3*x)]


        # library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x,y: f"{x}{y}", lambda x: f"exp({x})", lambda x: f"log(1+{x})"]
        library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x,y: f"{x}{y}", lambda x: f"exp({x})"]
        library_function_names += [lambda x: f"sin({x})", lambda x: f"cos({x})", lambda x: f"sin(2{x})", lambda x: f"cos(2{x})", lambda x: f"sin(3{x})", lambda x: f"cos(3{x})"]
        # library_function_names += [lambda x: f"sin({x})", lambda x: f"sin(2{x})", lambda x: f"sin(3{x})"]

        library = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            spatiotemporal_grid=self.t_grid,
            is_uniform=True,
            K=200,
            include_bias=True
            )

        # optimizer_type = trial.suggest_categorical('optimizer_type', ['stlsq', 'sr3', 'frols', 'ssr'])        
        optimizer_type = 'miosr'

        if optimizer_type == 'stlsq':
            optimizer_threshold = trial.suggest_float('optimizer_threshold', 1e-3, 1, log=True)
            optimizer = ps.STLSQ(threshold=optimizer_threshold)
        elif optimizer_type == 'sr3':
            optimizer_threshold = trial.suggest_float('optimizer_threshold', 1e-3, 1, log=True)
            optimizer = ps.SR3(threshold=optimizer_threshold)
        elif optimizer_type == 'frols':
            alpha = trial.suggest_float('alpha', 1e-3, 1, log=True)
            optimizer = ps.FROLS(alpha=alpha)
        elif optimizer_type == 'ssr':
            alpha = trial.suggest_float('alpha', 1e-3, 1, log=True)
            optimizer = ps.SSR(alpha=alpha)
        elif optimizer_type == 'miosr':
            alpha = trial.suggest_float('alpha_miosr', 1e-3, 1, log=True)
            if self.sparsity == 0:
                sparsity = trial.suggest_int('sparsity', 1, 20)
            else:
                sparsity = self.sparsity
            optimizer = ps.MIOSR(alpha=alpha,target_sparsity=None,group_sparsity=(sparsity,1),normalize_columns=True)
        

        model = ps.SINDy(optimizer=optimizer, feature_library=library)

        return (model, seed)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""

        # library_functions = [lambda x: x, lambda x: x * x, lambda x: np.exp(x), lambda x: np.log(1+np.abs(x))]
        # library_functions += [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.sin(2*x), lambda x: np.cos(2*x), lambda x: np.sin(3*x), lambda x: np.cos(3*x)]
        # library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x: f"exp({x})", lambda x: f"log(1+{x})"]
        # library_function_names += [lambda x: f"sin({x})", lambda x: f"cos({x})", lambda x: f"sin(2{x})", lambda x: f"cos(2{x})", lambda x: f"sin(3{x})", lambda x: f"cos(3{x})"]

        # library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x*y, lambda x: np.exp(x), lambda x: np.log(1+np.abs(x))]
        library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x*y, lambda x: np.exp(x)]
        library_functions += [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.sin(2*x), lambda x: np.cos(2*x), lambda x: np.sin(3*x), lambda x: np.cos(3*x)]
        # library_functions += [lambda x: np.sin(x), lambda x: np.sin(2*x), lambda x: np.sin(3*x)]


        # library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x,y: f"{x}{y}", lambda x: f"exp({x})", lambda x: f"log(1+{x})"]
        library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x,y: f"{x}{y}", lambda x: f"exp({x})"]
        library_function_names += [lambda x: f"sin({x})", lambda x: f"cos({x})", lambda x: f"sin(2{x})", lambda x: f"cos(2{x})", lambda x: f"sin(3{x})", lambda x: f"cos(3{x})"]
        # library_function_names += [lambda x: f"sin({x})", lambda x: f"sin(2{x})", lambda x: f"sin(3{x})"]


        library = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            spatiotemporal_grid=self.t_grid,
            is_uniform=True,
            K=200,
            include_bias=True
            )

        if parameters is None:
            # optimizer = ps.STLSQ(threshold=1e-3)
            optimizer = ps.SR3(threshold=0.05, thresholder="l1", max_iter=1000, normalize_columns=False, tol=1e-1)
            return (ps.SINDy(optimizer=optimizer, feature_library=library), seed)
        else:

            # optimizer_type = parameters['optimizer_type']
            optimizer_type = 'miosr'
            if optimizer_type == 'stlsq':
                optimizer_threshold = parameters['optimizer_threshold']
                optimizer = ps.STLSQ(threshold=optimizer_threshold)
            elif optimizer_type == 'sr3':
                optimizer_threshold = parameters['optimizer_threshold']
                optimizer = ps.SR3(threshold=optimizer_threshold)
            elif optimizer_type == 'frols':
                alpha = parameters['alpha']
                optimizer = ps.FROLS(alpha=alpha)
            elif optimizer_type == 'ssr':
                alpha = parameters['alpha']
                optimizer = ps.SSR(alpha=alpha)
            elif optimizer_type == 'miosr':
                alpha = parameters['alpha_miosr']
                if self.sparsity == 0:
                    sparsity = parameters['sparsity']
                else:
                    sparsity = self.sparsity
                optimizer = ps.MIOSR(alpha=alpha,target_sparsity=None,group_sparsity=(sparsity,1),normalize_columns=True)

            return (ps.SINDy(optimizer=optimizer, feature_library=library), seed)
    
    def train(self, model, tuning=False):
        """Train model."""

        model, seed = model

        INF = 1.0e9
            
        # Y needs to be a list of numpy arrays each of shape (n_measurements, 1)

        ys_train = [np.stack([y,t], axis=1) for y,t in zip(self.Y_train,self.T_train)]
        ts_train = [t.flatten() for t in self.T_train]
        xs_train = [x.item() for x in self.X_train]


        ys_val = [np.expand_dims(y, axis=1) for y in self.Y_val]
        ts_val = [t.flatten() for t in self.T_val]
        xs_val = [x.item() for x in self.X_val]

        ys_test = [np.expand_dims(y, axis=1) for y in self.Y_test]
        ts_test = [t.flatten() for t in self.T_test]
        xs_test = [x.item() for x in self.X_test]

        library_functions = model.feature_library.functions

        actual_functions = []
        for i in range(len(library_functions)):
            if library_functions[i].__code__.co_argcount == 1:
                actual_functions.append((lambda val: lambda x, t: library_functions[val](x))(i))
                actual_functions.append((lambda val: lambda x, t: library_functions[val](t))(i))
            elif library_functions[i].__code__.co_argcount == 2:
                actual_functions.append((lambda val: lambda x, t: library_functions[val](x,t))(i))

        np.random.seed(seed)
        try:

            model.fit(ys_train, multiple_trajectories=True)
        except Exception as e:
            print(e)
            return {'train_loss': INF, 'val_loss': INF, 'test_loss': INF}

        
    
        def simulate(model, x0, t):
            functions = actual_functions
            coefficients = model.coefficients()[0]
            def derivative(x, t):
                res = 0
                for i, f in enumerate(functions):
                    res += coefficients[i+1] * f(x[0],t)
                res += coefficients[0]
                return [res]
            y, infodict = odeint(derivative, [x0], t, full_output=True)
            y = y.flatten()
            if infodict['message'] != 'Integration successful.':
                y = interpolate_nans(y)
            return y
                
        

        normalize = False

        val_loss = 0
        val_losses = []
        if tuning:
            val_losses = []
            for x0, t, y in zip(xs_val, ts_val, ys_val):
                try:
                    y_pred = clip_to_finite(simulate(model,x0, t))
           
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                if normalize:
                    sample_range = np.max(y) - np.min(y)
                    val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
                else:
                    val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            val_loss = np.mean(val_losses)
        
        test_loss = 0
        if not tuning:
            test_losses = []
            for x0, t, y in zip(xs_test, ts_test, ys_test):
                try:
                    y_pred = clip_to_finite(simulate(model,x0, t))
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                if normalize:
                    sample_range = np.max(y) - np.min(y)
                    test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
                else:
                    test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            test_loss = np.mean(test_losses)

           
        if not tuning:
            self.fitted = True
            self.results = {'val_loss': val_loss, 'test_loss': test_loss}
    
        return model, {'val_loss': val_loss, 'test_loss': test_loss}
    
    def evaluate(self, model):

        
        ys_val = [np.expand_dims(y, axis=1) for y in self.Y_val]
        ts_val = [t.flatten() for t in self.T_val]
        xs_val = [x.item() for x in self.X_val]

        ys_test = [np.expand_dims(y, axis=1) for y in self.Y_test]
        ts_test = [t.flatten() for t in self.T_test]
        xs_test = [x.item() for x in self.X_test]

        library_functions = model.feature_library.functions

        actual_functions = []
        for i in range(len(library_functions)):
            if library_functions[i].__code__.co_argcount == 1:
                actual_functions.append((lambda val: lambda x, t: library_functions[val](x))(i))
                actual_functions.append((lambda val: lambda x, t: library_functions[val](t))(i))
            elif library_functions[i].__code__.co_argcount == 2:
                actual_functions.append((lambda val: lambda x, t: library_functions[val](x,t))(i))

        def simulate(model, x0, t):
            functions = actual_functions
            coefficients = model.coefficients()[0]
            def derivative(x, t):
                res = 0
                for i, f in enumerate(functions):
                    res += coefficients[i+1] * f(x[0],t)
                res += coefficients[0]
                return [res]
            y, infodict = odeint(derivative, [x0], t, full_output=True)
            y = y.flatten()
            if infodict['message'] != 'Integration successful.':
                y = interpolate_nans(y)
            return y
        

        normalize = False

        val_loss = 0
        val_losses = []
        val_losses = []
        for x0, t, y in zip(xs_val, ts_val, ys_val):
            try:
                y_pred = clip_to_finite(simulate(model,x0, t))
            except Exception as e:
                print(e)
                y_pred = np.zeros_like(y)
            if normalize:
                sample_range = np.max(y) - np.min(y)
                val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
            else:
                val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
        val_loss = np.mean(val_losses)
        val_loss_std = np.std(val_losses)

        test_loss = 0
        test_losses = []
        for x0, t, y in zip(xs_test, ts_test, ys_test):
            try:
                y_pred = clip_to_finite(simulate(model,x0, t))
            except Exception as e:
                print(e)
                y_pred = np.zeros_like(y)
            if normalize:
                sample_range = np.max(y) - np.min(y)
                test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))/sample_range))
            else:
                test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
        
        test_loss = np.mean(test_losses)
        test_loss_std = np.std(test_losses)

        return model, {'val_loss': val_loss, 'test_loss': test_loss, 'test_loss_std': test_loss_std, 'val_loss_std': val_loss_std}
    

class PySRBenchmark(BaseBenchmark):
    """PySR benchmark."""
    def __init__(self, sparsity=2):
        self.fitted = False
        self.sparsity = sparsity
        super().__init__()
        global pysr
        pysr = importlib.import_module('pysr')
    
    def get_name(self):
        return f'PySR-{self.sparsity}'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""

        differentiation_kind = trial.suggest_categorical('differentiation_kind', ['finite_difference', 'spline', 'trend_filtered'])
        if differentiation_kind == 'finite_difference':
            k = trial.suggest_int('k', 1, 5)
            differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
        elif differentiation_kind == 'spline':
            s = trial.suggest_float('s', 1e-3, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
        elif differentiation_kind == 'trend_filtered':
            order = trial.suggest_int('order', 0, 2)
            alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
        elif differentiation_kind == 'smoothed_finite_difference':
            window_length = trial.suggest_int('window_length', 1, 5)
            differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})


        model = pysr.PySRRegressor(
            binary_operators=["+", "*", "/", "-"],
            unary_operators=[
                "sin",
                "exp",
                "logp(x) = log(abs(x)+1.0f-3)",
                # ^ Custom operator (julia syntax)
            ],
            timeout_in_seconds=15,
            niterations=1000,
            nested_constraints={"sin": {"sin": 0}},
            extra_sympy_mappings={"logp": lambda x: sympy.log(sympy.Abs(x) + 1e-3)},
            maxsize=self.sparsity,
            deterministic=True,
            random_state=seed,
            procs=0,
            multithreading=False,
            model_selection='accuracy'
        )
        return (model, differentiation_method)

    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is None:
            differentiation_kind = 'finite_difference'
            k = 2
            differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
        else:
            differentiation_kind = parameters['differentiation_kind']
            if differentiation_kind == 'finite_difference':
                k = parameters['k']
                differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
            elif differentiation_kind == 'spline':
                s = parameters['s']
                differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
            elif differentiation_kind == 'trend_filtered':
                order = parameters['order']
                alpha = parameters['alpha']
                differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
            elif differentiation_kind == 'smoothed_finite_difference':
                window_length = parameters['window_length']
                differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})

        model = pysr.PySRRegressor(
        binary_operators=["+", "*", "/", "-"],
        unary_operators=[
            "sin",
            "exp",
            "logp(x) = log(abs(x)+1.0f-3)",
            # ^ Custom operator (julia syntax)
        ],
        timeout_in_seconds=60,
        niterations=1000,
        nested_constraints={"sin": {"sin": 0}},
        extra_sympy_mappings={"logp": lambda x: sympy.log(sympy.Abs(x) + 1e-3)},
        maxsize=self.sparsity,
        deterministic=True,
        random_state=seed,
        procs=0,
        multithreading=False,
        model_selection='accuracy'
        )
        return (model, differentiation_method)

    def train(self, model, tuning=False):

        # Differentiate the dataset
        model, differentiation_method = model

        dYdt_train = []
        for (y, t) in zip(self.Y_train, self.T_train):
            dYdt_train.append(differentiation_method._differentiate(y.reshape(-1,1), t))
        
        dYdt_val = []
        for (y, t) in zip(self.Y_val, self.T_val):
            dYdt_val.append(differentiation_method._differentiate(y.reshape(-1,1), t))
        
        dYdt_test = []
        for (y, t) in zip(self.Y_test, self.T_test):
            dYdt_test.append(differentiation_method._differentiate(y.reshape(-1,1), t))

        X_sr_y = np.concatenate([self.Y_train[i] for i in range(len(self.Y_train))], axis=0)
        X_sr_t = np.concatenate([self.T_train[i] for i in range(len(self.T_train))], axis=0)
        X_sr = np.stack([X_sr_y, X_sr_t], axis=1)
        y_sr = np.concatenate([dYdt_train[i] for i in range(len(dYdt_train))], axis=0)
        
        # Fit the model
        model.fit(X_sr,y_sr)

        if tuning:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=False)
        else:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=True)

        return model, results


    def evaluate(self, model, compute_val_loss=True, compute_test_loss=True):

        def derivative(x,t):
            return model.predict(np.array([[x[0],t]]))

        if compute_val_loss:
            val_losses = []
            for x0, t, y in zip(self.X_val, self.T_val, self.Y_val):
                try:
                    y_pred = clip_to_finite(odeint(derivative,x0, t)[:,0])
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            val_loss = np.mean(val_losses)
            val_loss_std = np.std(val_losses)
        else:
            val_loss = 0
            val_loss_std = 0
        
        if compute_test_loss:
            test_losses = []
            for x0, t, y in zip(self.X_test, self.T_test, self.Y_test):
                try:
                    y_pred = clip_to_finite(odeint(derivative,x0, t)[:,0])
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            test_loss = np.mean(test_losses)
            test_loss_std = np.std(test_losses)
        else:
            test_loss = 0
            test_loss_std = 0
        
        return model, {'val_loss': val_loss, 'test_loss': test_loss, 'test_loss_std': test_loss_std, 'val_loss_std': val_loss_std}

    
class NeuralODEBenchmark(BaseBenchmark):
    
    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'NeuralODE'
    
    def get_model_for_tuning(self, trial, seed):

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        layer_sizes = []
        for i in range(num_layers):
            units = trial.suggest_int(f'layer_{i}_units', 16, 128, log=True)
            layer_sizes.append(units)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        activation_name = trial.suggest_categorical('activation', ['ELU', 'Sigmoid'])
        activation = getattr(nn, activation_name)

        # Create the model with suggested hyperparameters
        model = NeuralODERegressor(
            layer_sizes=layer_sizes,
            activation=activation,
            init_method=nn.init.kaiming_normal_ if activation_name == 'ReLU' else nn.init.xavier_normal_,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            solver='rk4',
            # solver_options={'step_size': 0.01},
            device=self.config['device'],
            seed=seed
        )

        return model
    
    def get_final_model(self, parameters, seed):
        """Get model for testing."""

        if parameters is None:
            # Return a default model
            return NeuralODERegressor(
                layer_sizes=[64, 64],
                activation=nn.ELU,
                init_method=nn.init.kaiming_normal_,
                learning_rate=self.config['lr'],
                weight_decay=1e-6,
                dropout_rate=0.1,
                solver='rk4',
                # solver='dopri5',
                # solver_options={'step_size': 0.01},
                device=self.config['device'],
                seed=seed
            )
        else:    
            # Get the layer sizes
            num_layers = parameters['num_layers']
            layer_sizes = []
            for i in range(num_layers):
                units = parameters[f'layer_{i}_units']
                layer_sizes.append(units)
            activation_name = parameters['activation']
            activation = getattr(nn, activation_name)
            return NeuralODERegressor(
                layer_sizes=layer_sizes,
                activation=activation,
                init_method=nn.init.kaiming_normal_ if parameters['activation'] == nn.ReLU else nn.init.xavier_normal_,
                learning_rate=parameters['learning_rate'],
                weight_decay=parameters['weight_decay'],
                dropout_rate=parameters['dropout_rate'],
                solver='rk4',
                # solver_options={'step_size': 0.1},
                device=self.config['device'],
                seed=seed
            )
        
    def train(self, model, tuning=False):

        t_shared = self.T_train[0]
        model.fit(self.X_train[:,0],t_shared,self.Y_train,self.X_val[:,0],t_shared,self.Y_val, batch_size=self.config['batch_size'], max_epochs=self.config['n_epochs'], tuning=tuning)

        if tuning:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=False)
        else:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=True)
        
        return model, results

    def evaluate(self, model, compute_val_loss=True, compute_test_loss=True):

        if compute_val_loss:
            val_losses = []
            for x0, t, y in zip(self.X_val, self.T_val, self.Y_val):
                try:
                    y_pred = model.predict(x0[0], t).flatten()
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                val_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            val_loss = np.mean(val_losses)
            val_loss_std = np.std(val_losses)
        else:
            val_loss = 0
            val_loss_std = 0
        
        if compute_test_loss:
            test_losses = []
            for x0, t, y in zip(self.X_test, self.T_test, self.Y_test):
                try:
                    y_pred = model.predict(x0[0], t).flatten()
                except Exception as e:
                    print(e)
                    y_pred = np.zeros_like(y)
                test_losses.append(min(INF,np.sqrt(mean_squared_error(y, y_pred))))
            test_loss = np.mean(test_losses)
            test_loss_std = np.std(test_losses)
        else:
            test_loss = 0
            test_loss_std = 0
        
        return model, {'val_loss': val_loss, 'test_loss': test_loss, 'test_loss_std': test_loss_std, 'val_loss_std': val_loss_std}

class DeepONetBenchmark(BaseBenchmark):
    
    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'DeepONet'
    
    def get_model_for_tuning(self, trial, seed):

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        num_hidden = trial.suggest_int('num_hidden', 10, 100)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

        # Create the model with suggested hyperparameters
        model = DeepONet(num_hidden=num_hidden, 
                         num_layers=num_layers, 
                         dropout_rate=dropout_rate, 
                         learning_rate=learning_rate, 
                         weight_decay=weight_decay, 
                         device=self.config['device'], 
                         seed=seed,
                         batch_size=batch_size)

        return model
    
    def get_final_model(self, parameters, seed):
        """Get model for testing."""

        if parameters is None:
            # Return a default model
            return DeepONet(num_hidden=50,
                            num_layers=2,
                            dropout_rate=0.1,
                            learning_rate=self.config['lr'],
                            weight_decay=1e-6,
                            device=self.config['device'],
                            seed=seed,
                            batch_size=self.config['batch_size'])
        else:    
            return DeepONet(num_hidden=parameters['num_hidden'],
                            num_layers=parameters['num_layers'],
                            dropout_rate=parameters['dropout_rate'],
                            learning_rate=parameters['learning_rate'],
                            weight_decay=parameters['weight_decay'],
                            device=self.config['device'],
                            seed=seed,
                            batch_size=parameters['batch_size'])
            
        
    def train(self, model, tuning=False):

        model.fit(self.X_train,self.T_train,self.Y_train,self.X_val,self.T_val,self.Y_val, max_epochs=self.config['max_epochs'], tuning=tuning)

        if tuning:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=False)
        else:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=True)
        
        return model, results

    def evaluate(self, model, compute_val_loss=True, compute_test_loss=True):

        val_loss = 0
        val_std = 0
        test_loss = 0
        test_std = 0

        if compute_val_loss:
            y_val_pred = model.predict(self.X_val, self.T_val)
            val_loss = mean_rmse(self.Y_val, y_val_pred)
            val_std = std_rmse(self.Y_val, y_val_pred)
        
        if compute_test_loss:
            y_test_pred = model.predict(self.X_test, self.T_test)
            test_loss = mean_rmse(self.Y_test, y_test_pred)
            test_std = std_rmse(self.Y_test, y_test_pred)
        
        return model, {'val_loss': val_loss, 'test_loss': test_loss, 'test_loss_std': test_std, 'val_loss_std': val_std}

class NeuralLaplaceBenchmark(BaseBenchmark):
    
    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'NeuralLaplace'
    
    def get_model_for_tuning(self, trial, seed):

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        num_hidden = trial.suggest_int('num_hidden', 10, 100)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        latent_dim = trial.suggest_int('latent_dim', 2, 10)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

        # Create the model with suggested hyperparameters
        model = NeuralLaplaceRegressor(latent_dim=latent_dim,
                                       hidden_units=num_hidden, 
                                       num_layers=num_layers, 
                                       dropout_rate=dropout_rate, 
                                       learning_rate=learning_rate, 
                                       weight_decay=weight_decay, 
                                       device=self.config['device'], 
                                       seed=seed,
                                       batch_size=batch_size)

        return model
    
    def get_final_model(self, parameters, seed):
        """Get model for testing."""

        if parameters is None:
            # Return a default model
            return NeuralLaplaceRegressor(latent_dim=4,
                                          hidden_units=50,
                                          num_layers=2,
                                          dropout_rate=0.1,
                                          learning_rate=self.config['lr'],
                                          weight_decay=1e-6,
                                          device=self.config['device'],
                                          seed=seed,
                                          batch_size=self.config['batch_size'])
        else:    
            return NeuralLaplaceRegressor(latent_dim=parameters['latent_dim'],
                                          hidden_units=parameters['num_hidden'],
                                          num_layers=parameters['num_layers'],
                                          dropout_rate=parameters['dropout_rate'],
                                          learning_rate=parameters['learning_rate'],
                                          weight_decay=parameters['weight_decay'],
                                          device=self.config['device'],
                                          seed=seed,
                                          batch_size=parameters['batch_size'])
            
        
    def train(self, model, tuning=False):

        model.fit(self.X_train,self.T_train,self.Y_train,self.X_val,self.T_val,self.Y_val, max_epochs=self.config['max_epochs'], tuning=tuning)

        if tuning:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=False)
        else:
            model, results = self.evaluate(model, compute_val_loss=True, compute_test_loss=True)
        
        return model, results

    def evaluate(self, model, compute_val_loss=True, compute_test_loss=True):

        val_loss = 0
        val_std = 0
        test_loss = 0
        test_std = 0

        if compute_val_loss:
            y_val_pred = model.predict(self.X_val, self.T_val)
            val_loss = mean_rmse(self.Y_val, y_val_pred)
            val_std = std_rmse(self.Y_val, y_val_pred)
        
        if compute_test_loss:
            y_test_pred = model.predict(self.X_test, self.T_test)
            test_loss = mean_rmse(self.Y_test, y_test_pred)
            test_std = std_rmse(self.Y_test, y_test_pred)
        
        return model, {'val_loss': val_loss, 'test_loss': test_loss, 'test_loss_std': test_std, 'val_loss_std': val_std}


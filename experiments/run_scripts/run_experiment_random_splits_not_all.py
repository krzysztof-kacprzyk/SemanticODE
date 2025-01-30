import numpy as np
import argparse
import sys
sys.path.append('../')
from experiments.benchmark import SINDyBenchmark, WeakSINDyBenchmark, run_benchmark, SemanticODEBenchmark, PySRBenchmark, run_benchmark_random_splits, NeuralODEBenchmark, DeepONetBenchmark, NeuralLaplaceBenchmark
from semantic_odes.datasets import create_dataset_from_ode_with_noise, get_general_dataset, get_pk_dataset, get_integral_dataset, get_mackey_glass_dataset, get_real_tumor_dataset, get_real_pharma_dataset
from semantic_odes.odes import SimpleLogisticGrowthODE, DuffingODE
from semantic_odes.api import create_full_composition_library

def main(method, dataset, n_samples, n_measurements, noise_std, global_seed, n_trials, n_tune, experiment_name, sparsity):

    if dataset == 'sigmoid':
        ode = SimpleLogisticGrowthODE()
        initial_range = (0.2,4)
        t_range = (0,5)
        dataset = create_dataset_from_ode_with_noise(ode, n_samples, n_measurements, initial_range, t_range, noise_std, global_seed, r=1, K=2)
        semantic_max_length = 2
    elif dataset == 'integral':
        initial_range = (-1,1)
        t_range = (0,5)
        dataset = get_integral_dataset(n_samples, n_measurements, noise_std, global_seed)
        semantic_max_length = 3
    elif dataset == 'pharma':
        initial_range = (0,1)
        t_range = (0,1)
        dataset = get_pk_dataset(n_samples, n_measurements, noise_std, global_seed)
        semantic_max_length = 3
    elif dataset == 'mackey_glass':
        t_range = (0,1)
        initial_range = (1/3,1)
        semantic_max_length = 3
        dataset = get_mackey_glass_dataset(n_samples, n_measurements, noise_std, global_seed)
    elif dataset == 'general':
        initial_range = (-1,1)
        t_range = (0,1)
        dataset = get_general_dataset(n_samples, n_measurements, noise_std, global_seed)
        semantic_max_length = 3
    elif dataset == 'tumor_real':
        initial_range = (1,2)
        t_range = (0,1)
        dataset = get_real_tumor_dataset()
        semantic_max_length = 2
    elif dataset == 'tacrolimus_real':
        initial_range = (0.12,0.95)
        t_range = (0,1)
        dataset = get_real_pharma_dataset()
        semantic_max_length = 3
    elif dataset == 'duffing':
        ode = DuffingODE()
        initial_range = (-1,1)
        t_range = (0,5)
        dataset = create_dataset_from_ode_with_noise(ode, n_samples, n_measurements, initial_range, t_range, noise_std, global_seed)
        semantic_max_length = 3
   
    if method == 'sindy':
        method = SINDyBenchmark(sparsity=sparsity)
        run_benchmark_random_splits(dataset, method, n_trials=n_trials, n_tune=n_tune, seed=global_seed, experiment_name=experiment_name)
    elif method == 'wsindy':
        print('sparsity',sparsity)
        t_grid = np.linspace(t_range[0], t_range[1], n_measurements)
        method = WeakSINDyBenchmark(t_grid=t_grid, sparsity=sparsity)
        print(method.sparsity)
        run_benchmark_random_splits(dataset, method, n_trials=n_trials, n_tune=n_tune, seed=global_seed, experiment_name=experiment_name)
    elif 'semantic' in method:
        if method == 'semantic_tumor':
            composition_library = [['-+c','++f']]
            subtype = 'tumor'
        elif method == 'semantic_pharma':
            composition_library = [['+-c','--c','-+h']]
            subtype = 'pharma'
        else:
            composition_library = create_full_composition_library(max_length=semantic_max_length,is_infinite=True)
            subtype = 'default'
        opt_config = {
            'lr': 0.1,
            'n_epochs': 200,
            'batch_size': 256,
            'weight_decay': 0.0,
            'device': 'cpu',
            'dis_loss_coeff_1': 1e-2,
            'dis_loss_coeff_2': 1e-6,
            'n_tune':n_tune
        }
        config = {
            't_range': t_range,
            'initial_range': initial_range,
            'basis_function_type': 'bspline',
            'num_basis_functions': 6,
            'max_branches': 3,
            'opt_config': opt_config,
            'composition_library': composition_library,
            'verbose': True,
            'subtype': subtype
        }
        method = SemanticODEBenchmark(config)
        run_benchmark_random_splits(dataset, method, n_trials=n_trials, n_tune=0, seed=global_seed, experiment_name=experiment_name)
    elif method == 'pysr':
        method = PySRBenchmark(sparsity=sparsity)
        run_benchmark_random_splits(dataset, method, n_trials=n_trials, n_tune=n_tune, seed=global_seed, experiment_name=experiment_name)
    elif method == 'neural_ode':
        config = {
            'n_epochs': 200,
            'batch_size': 32,
            'device': 'gpu',
            'lr': 1e-3
        }
        method = NeuralODEBenchmark(config)
        run_benchmark_random_splits(dataset, method, n_trials=n_trials, n_tune=n_tune, seed=global_seed, experiment_name=experiment_name)
    elif method == 'deeponet':
        config = {
            'max_epochs': 200,
            'batch_size': 32,
            'device': 'gpu',
            'lr': 1e-3
        }
        method = DeepONetBenchmark(config)
        run_benchmark_random_splits(dataset, method, n_trials=n_trials, n_tune=n_tune, seed=global_seed, experiment_name=experiment_name)
    elif method == 'neural_laplace':
        config = {
            'max_epochs': 200,
            'batch_size': 32,
            'device': 'gpu',
            'lr': 1e-3
        }
        method = NeuralLaplaceBenchmark(config)
        run_benchmark_random_splits(dataset, method, n_trials=n_trials, n_tune=n_tune, seed=global_seed, experiment_name=experiment_name)
   
    
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sindy', help='Method to use for benchmarking')
    parser.add_argument('--dataset', type=str, default='sigmoid', help='Dataset to use for benchmarking')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of samples in dataset')
    parser.add_argument('--n_measurements', type=int, default=20, help='Number of measurements in dataset')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Noise standard deviation')
    parser.add_argument('--global_seed', type=int, default=0, help='Global seed for reproducibility')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--n_tune', type=int, default=0, help='Number of tuning trials to run')
    parser.add_argument('--experiment_name', type=str, default='untitled', help='Name of the experiment')
    parser.add_argument('--sparsity', type=int, default=2, help='Sparsity of the model')
    args = parser.parse_args()

    main(args.method, args.dataset, args.n_samples, args.n_measurements, args.noise_std, args.global_seed, args.n_trials, args.n_tune, args.experiment_name, args.sparsity)


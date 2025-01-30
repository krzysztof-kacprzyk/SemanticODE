n_trials=5
n_tune=20
global_seed=0
n_samples=200
n_measurements=20

experiment_name="Main"

noise_std=0.01 # does not matter

dataset = tacrolimus_real
echo "Dataset: $dataset"

# Semantic
echo "Semantic"
python run_scripts/run_experiment_random_splits_not_all.py --method semantic --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# DeepONet
echo "DeepONet"
python run_scripts/run_experiment_random_splits_not_all.py --method deeponet --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# Neural Laplace
echo "Neural Laplace"
python run_scripts/run_experiment_random_splits_not_all.py --method neural_laplace --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# Neural ODE
echo "Neural ODE"
python run_scripts/run_experiment_random_splits_not_all.py --method neural_ode --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# SINDy - sparsity 5
echo "SINDy-5"
python run_scripts/run_experiment_random_splits_not_all.py --method sindy --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 5

# SINDy - sparsity 0
echo "SINDy-0"
python run_scripts/run_experiment_random_splits_not_all.py --method sindy --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 0

# PySR - sparsity 20
echo "PySR-20"
python run_scripts/run_experiment_random_splits_not_all.py --method pysr --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 20

dataset = tumor_real
echo "Dataset: $dataset"

# Semantic
echo "Semantic"
python run_scripts/run_experiment_random_splits_not_all.py --method semantic --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# DeepONet
echo "DeepONet"
python run_scripts/run_experiment_random_splits_not_all.py --method deeponet --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# Neural Laplace
echo "Neural Laplace"
python run_scripts/run_experiment_random_splits_not_all.py --method neural_laplace --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# Neural ODE
echo "Neural ODE"
python run_scripts/run_experiment_random_splits_not_all.py --method neural_ode --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name

# SINDy - sparsity 5
echo "SINDy-5"
python run_scripts/run_experiment_random_splits_not_all.py --method sindy --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 5

# SINDy - sparsity 0
echo "SINDy-0"
python run_scripts/run_experiment_random_splits_not_all.py --method sindy --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 0

# WSINDy - sparsity 5
echo "W-SINDy-5"
python run_scripts/run_experiment_random_splits_not_all.py --method wsindy --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 5

# WSINDy - sparsity 0
echo "W-SINDy-0"
python run_scripts/run_experiment_random_splits_not_all.py --method wsindy --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 0

# PySR - sparsity 20
echo "PySR-20"
python run_scripts/run_experiment_random_splits_not_all.py --method pysr --dataset $dataset --n_samples $n_samples --n_measurements $n_measurements --noise_std $noise_std --global_seed $global_seed --n_trials $n_trials --n_tune $n_tune --experiment_name $experiment_name --sparsity 20

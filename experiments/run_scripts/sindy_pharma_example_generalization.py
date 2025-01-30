import sys
sys.path.append('../')
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from semantic_odes.model_numpy import calculate_loss
from semantic_odes.datasets import get_pk_dataset, get_pk_dataset_extrapolation, get_pk_dataset_extrapolation_x0
from experiments.benchmark import run_benchmark, SemanticODEBenchmark, SINDyBenchmark, WeakSINDyBenchmark, generate_indices
import pandas as pd

def mean_rmse(y_true_list, y_pred_list):
    rmse_scores = [np.sqrt(np.mean((y_true_i - y_pred_i)**2)) for y_true_i, y_pred_i in zip(y_true_list, y_pred_list)]
    return np.mean(rmse_scores)

global_seed = 2

n_samples = 100
n_measurements = 20
noise_std = 0.01

pk_dataset = get_pk_dataset(n_samples=n_samples, n_measurements=n_measurements, noise_std=noise_std, seed=global_seed)
pk_dataset_extrapolation = get_pk_dataset_extrapolation(n_samples=n_samples, n_measurements=n_measurements*2, noise_std=noise_std, seed=global_seed)
pk_dataset_extrapolation_x0 = get_pk_dataset_extrapolation_x0(n_samples=n_samples, n_measurements=n_measurements, noise_std=noise_std, seed=global_seed)
train_indices, val_indices, test_indices = generate_indices(n_samples, 0.7, 0.15, seed=global_seed)

def create_equation(raw_eq, label):

    # Replace each x0 with x and x1 with t
    eq = raw_eq.replace("x0", "x(t)")
    eq = eq.replace("x1", "t")
    # replace sin with \sin, cos with \cos, exp with \exp, log with \log
    eq = eq.replace("sin", r"\sin")
    eq = eq.replace("cos", r"\cos")
    eq = eq.replace("exp", r"\exp")
    eq = eq.replace("log", r"\log")
    # replace + - with -
    eq = eq.replace(" + -", " -")
    # replace 1 with empty
    eq = eq.replace(" 1 ", "")
    eq = eq.replace("(1 ", "(")

    return r"\begin{dmath}" + "\n" + label + "\n" + r"\dot{x}(t) = " + eq + "\n" + r"\end{dmath}"



def create_df_row(sparsity):
    row_dict = {}
    sindy_benchmark = SINDyBenchmark(sparsity=sparsity)
    summary, sindy_model = run_benchmark(pk_dataset, sindy_benchmark, n_trials=1, n_tune=20, seed=global_seed)
    row_dict['syntactic_constraints'] = [r"$\dot{x}=\sum_{i=1}^n \alpha_i g_i, n \leq " + str(sparsity) + r"$"]
    row_dict['model'] = ['SINDy']
    row_dict['semantic_constraints'] = ['NA']
    row_dict['syntactic_representation'] = [r"\cref{eq:sindy_"+str(sparsity)+"}"]
    row_dict['semantic_representation'] = ['NA']
    _, results = sindy_benchmark.evaluate(sindy_model)
    row_dict['test_rmse'] = [f"${results['test_loss']:.3f}_"+r"{("+f"{results['test_loss_std']:.3f}"+r")}$"]
    # Extrapolation
    sindy_benchmark.prepare_data(pk_dataset_extrapolation_x0, train_indices, val_indices, test_indices)
    _, results = sindy_benchmark.evaluate(sindy_model)
    row_dict['extra_rmse'] = [f"${results['test_loss']:.3f}_"+r"{("+f"{results['test_loss_std']:.3f}"+r")}$"]
    return pd.DataFrame(row_dict), create_equation(sindy_model.equations(precision=2)[0], label="\\label{eq:sindy_"+str(sparsity)+"}")

                             
def create_df():
    sparsities = [1,2,5,10,15]
    rows = []
    eqs = []
    for sparsity in sparsities:
        row, eq = create_df_row(sparsity)
        eqs.append(eq)
        rows.append(row)
    return pd.concat(rows, ignore_index=True), "\n".join(eqs)

        
df, eqs = create_df()
df = df[['model', 'syntactic_constraints', 'semantic_constraints', 'syntactic_representation','semantic_representation', 'test_rmse', 'extra_rmse']]

df.to_latex("analysis/output/Pharma SINDy OOD (Tab 9).tex", index=False)
with open("analysis/output/Pharma SINDy OOD (Tab 9) eqs.tex", "w") as f:
    f.write(eqs)

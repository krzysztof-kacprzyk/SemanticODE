import numpy as np
from abc import ABC, abstractmethod
from scipy.integrate import odeint
from semantic_odes.odes import HeatODE
import pandas as pd
import os
from ddeint import ddeint
from scipy.interpolate import interp1d

class Dataset:
    def __init__(self, name, X0, T, Y):
        self.name = name
        self.X0 = X0
        self.T = T
        self.Y = Y

    def __repr__(self):
        return f"Dataset({self.name})"
    
    def get_X_T_Y(self):
        return self.X0, self.T, self.Y
    
    def get_name(self):
        return self.name
    
    def __len__(self):
        return self.X0.shape[0]
    
# Adapted from https://github.com/krzysztof-kacprzyk/TIMEVIEW
class TacrolimusPK():

    def __init__(self):
        self.feature_names = ['DOSE', 'DV_0', 'SEX', 'WEIGHT', 'HT', 'HB', 'CREAT', 'CYP', 'FORM']

        self.params = {
            'TVCL': 21.2, # Typical value of clearance (L/h)
            'TVV1': 486, # Typical central volume of distribution (L)
            'TVQ': 79, # Typical intercomp clearance 1 (L/h)
            'TVV2': 271, # Typical peripheral volume of distribution 1 (L)
            'TVKTR': 3.34, # Typical transfert rate constant(1/h)
            'HTCL': -1.14, # effect of HT on cl
            'CYPCL': 2.00, # effect of CYP on cl
            'STKTR': 1.53, # effect of study on ktr
            'STV1': 0.29 # effect of study on v1
        }

    def get_feature_names(self):
        return self.feature_names
    
    def get_feature_ranges(self):
        return {
            'DOSE': (1, 10),
            'DV_0': (0, 20),
            'SEX': [0, 1],
            'WEIGHT': (45, 110),
            'HT': (20, 47),
            'HB': (6, 16),
            'CREAT': (60, 830),
            'CYP': [0, 1],
            'FORM': [0, 1]
        }


    def predict(self, covariates, time_points):

        assert covariates.shape[0] == 1, "TacrolimusPK only supports one patient at a time"
        assert covariates.shape[1] == 9, "TacrolimusPK requires 9 covariates"

        # Covariates
        cov = {name: covariates[0,i] for i, name in enumerate(self.feature_names)}

        # ODE Parameters
        CL = self.params['TVCL'] * ((cov['HT'] / 35) ** self.params['HTCL']) * (self.params['CYPCL']) ** cov['CYP']
        V1 = self.params['TVV1'] * ((self.params['STV1']) ** cov['FORM'])
        Q = self.params['TVQ']
        V2 = self.params['TVV2']
        KTR = self.params['TVKTR'] * ((self.params['STKTR']) ** cov['FORM'])

        # Initial conditions
        DEPOT_0 = cov['DOSE']
        TRANS1_0 = 0
        TRANS2_0 = 0
        TRANS3_0 = 0
        CENT_0 = cov['DV_0'] * (V1 / 1000)
        PERI_0 = V2/V1 * CENT_0

        # print(f"CL: {CL}, V1: {V1}, Q: {Q}, V2: {V2}, KTR: {KTR}, DEPOT_0: {DEPOT_0}, CENT_0: {CENT_0}, PERI_0: {PERI_0}")

        # ODE
        def tacrolimus_ode(y, t):
            dDEPOTdt = -KTR * y[0]
            dTRANS1dt = KTR * y[0] - KTR * y[1]
            dTRANS2dt = KTR * y[1] - KTR * y[2]
            dTRANS3dt = KTR * y[2] - KTR * y[3]
            dCENTdt = KTR*y[3] - ((CL + Q) * y[4]/V1) + (Q * y[5] / V2)
            dPERIdt = (Q * y[4]/V1) - (Q * y[5]/V2)
            return [dDEPOTdt, dTRANS1dt, dTRANS2dt, dTRANS3dt, dCENTdt, dPERIdt]
        
        # Solve ODE
        y = odeint(tacrolimus_ode, [DEPOT_0, TRANS1_0, TRANS2_0, TRANS3_0, CENT_0, PERI_0], time_points)

        y = y[:, 4] # Only return central compartment
        y = y * 1000 / V1 # Convert back to micro g/mL

        return y
    
    def predict_from_parameters(self, parameters, time_points):

        # Covariates

        # ODE Parameters
        CL = parameters[0]
        V1 = parameters[1]
        Q = parameters[2]
        V2 = parameters[3]
        KTR = parameters[4]

        # Initial conditions
        DEPOT_0 = parameters[5]
        TRANS1_0 = 0
        TRANS2_0 = 0
        TRANS3_0 = 0
        CENT_0 = parameters[6] * (V1 / 1000)
        PERI_0 = V2/V1 * CENT_0

        # ODE
        def tacrolimus_ode(y, t):
            dDEPOTdt = -KTR * y[0]
            dTRANS1dt = KTR * y[0] - KTR * y[1]
            dTRANS2dt = KTR * y[1] - KTR * y[2]
            dTRANS3dt = KTR * y[2] - KTR * y[3]
            dCENTdt = KTR*y[3] - ((CL + Q) * y[4]/V1) + (Q * y[5] / V2)
            dPERIdt = (Q * y[4]/V1) - (Q * y[5]/V2)
            return [dDEPOTdt, dTRANS1dt, dTRANS2dt, dTRANS3dt, dCENTdt, dPERIdt]
        
        # Solve ODE
        y = odeint(tacrolimus_ode, [DEPOT_0, TRANS1_0, TRANS2_0, TRANS3_0, CENT_0, PERI_0], time_points)

        y = y[:, 4] # Only return central compartment
        y = y * 1000 / V1 # Convert back to micro g/mL

        return y
    
    def get_random_dataset(self,n_patients, t_range, seed=0):

        covariates = self.get_random_covariates(n_patients, seed=seed)
        trajectories = self(covariates, t_range=t_range)
        
        return covariates, trajectories
    
    def get_random_covariates(self, n_patients, seed=0):

        # Random number generator
        rng = np.random.default_rng(seed)

        dataset = []

        for i in range(n_patients):

            covariates = []
            for feature in self.get_feature_names():
                f_range = self.get_feature_ranges()[feature]
                if isinstance(f_range, list):
                    # choose a random value from the list
                    value = rng.choice(f_range)
                else:
                    # choose a random value from the range
                    value = rng.uniform(f_range[0], f_range[1])
                covariates.append(value)
            covariates = np.array(covariates).reshape(1, -1)
            dataset.append(covariates)
        
        return np.concatenate(dataset, axis=0)
    

def create_dataset_from_ode(ode, n_samples, n_measurements, initial_range, t_range, **kwargs):
    xs = np.linspace(initial_range[0],initial_range[1],n_samples).reshape(-1,1)       
    ts = np.stack([np.linspace(t_range[0],t_range[1],n_measurements) for _ in range(n_samples)], axis=0)

    ys = np.stack([ode.solve(xs[i][0], ts[i], **kwargs) for i in range(n_samples)], axis=0)

    dataset_name = f"{ode.name}_n={n_samples}_m={n_measurements}"
    # add the parameters to the dataset name
    for key, value in kwargs.items():
        dataset_name += f"_{key}={value}"

    return Dataset(dataset_name,xs, ts, ys)

def create_dataset_from_ode_with_noise(ode, n_samples, n_measurements, initial_range, t_range, noise_std, seed, noise_ratio=False, **kwargs):
    xs = np.linspace(initial_range[0],initial_range[1],n_samples).reshape(-1,1)       
    ts = np.stack([np.linspace(t_range[0],t_range[1],n_measurements) for _ in range(n_samples)], axis=0)

    gen = np.random.default_rng(seed)
    raw_ys = np.stack([ode.solve(xs[i][0], ts[i], **kwargs) for i in range(n_samples)], axis=0)

    # add noise
    if noise_ratio:
        ys = raw_ys + gen.normal(0, noise_std, size=raw_ys.shape) * (np.max(raw_ys,axis=1) - np.min(raw_ys,axis=1)).reshape(-1,1)
    else:
        ys = raw_ys + gen.normal(0, noise_std, size=raw_ys.shape)

    dataset_name = f"{ode.name}_n={n_samples}_m={n_measurements}_noise={noise_std}"
    # add the parameters to the dataset name
    for key, value in kwargs.items():
        dataset_name += f"_{key}={value}"

    return Dataset(dataset_name,xs, ts, ys)

def get_pk_dataset(n_samples, n_measurements, noise_std, seed):

    tac = TacrolimusPK()
    default_covariates = np.array([[10,10,1,80,20,16,60,1,0]])

    v0 = np.linspace(0,20,n_samples)
    t = np.linspace(0,24,n_measurements)

    covariates = np.tile(default_covariates,(len(v0),1))
    covariates[:,1] = v0

    ys = np.stack([tac.predict(covariates[[i],:],t) for i in range(len(v0))],axis=0)/20
    ts = np.tile(t,(len(v0),1))/24
    xs = v0.reshape(-1,1)/20

    # add noise
    gen = np.random.default_rng(seed)
    ys = ys + gen.normal(0, noise_std, size=ys.shape)

    dataset_name = f"tacrolimus_n={n_samples}_m={n_measurements}_noise={noise_std}"

    return Dataset(dataset_name,xs, ts, ys)

def get_pk_dataset_extrapolation(n_samples, n_measurements, noise_std, seed):

    tac = TacrolimusPK()
    default_covariates = np.array([[10,10,1,80,20,16,60,1,0]])

    v0 = np.linspace(0,20,n_samples)
    t = np.linspace(0,48,n_measurements)

    covariates = np.tile(default_covariates,(len(v0),1))
    covariates[:,1] = v0

    ys = np.stack([tac.predict(covariates[[i],:],t) for i in range(len(v0))],axis=0)/20
    ts = np.tile(t,(len(v0),1))/24
    xs = v0.reshape(-1,1)/20

    # add noise
    gen = np.random.default_rng(seed)
    ys = ys + gen.normal(0, noise_std, size=ys.shape)

    dataset_name = f"tacrolimus_extra_n={n_samples}_m={n_measurements}_noise={noise_std}"

    measurement_indices = [0] + list(np.arange(n_measurements//2,n_measurements))
    ys = ys[:,measurement_indices]
    ts = ts[:,measurement_indices]

    return Dataset(dataset_name,xs, ts, ys)


def get_temperature_dataset(n_samples, n_measurements, noise_std, seed):

    initial_range = (2,80)
    t_range = (0,1)

    ode = HeatODE()

    xs = np.linspace(initial_range[0],initial_range[1],n_samples).reshape(-1,1)       
    ts = np.stack([np.linspace(t_range[0],t_range[1],n_measurements) for _ in range(n_samples)], axis=0)

    gen = np.random.default_rng(seed)
    raw_ys = np.stack([ode.solve(xs[i][0], ts[i]) for i in range(n_samples)], axis=0)

    # add noise
    ys = raw_ys + gen.normal(0, noise_std, size=raw_ys.shape)

    dataset_name = f"{ode.name}_n={n_samples}_m={n_measurements}_noise={noise_std}"

    xs = xs/80
    ys = ys/80
    
    return Dataset(dataset_name,xs, ts, ys)

def get_mackey_glass_dataset(n_samples, n_measurements, noise_std, seed):

    initial_range = (1.0,3.0)
    t_range = (0,30)


    def mackey_glass(Y, t, beta, gamma, tau, n):
        return beta * Y(t - tau) / (1 + Y(t - tau)**n) - gamma * Y(t)
    
    xs = np.linspace(initial_range[0],initial_range[1],n_samples).reshape(-1,1)       
    ts = np.stack([np.linspace(t_range[0],t_range[1],n_measurements) for _ in range(n_samples)], axis=0)

    beta = 0.4
    gamma = 0.25
    tau = 4.0
    n = 4

    gen = np.random.default_rng(seed)
    # raw_ys = np.stack([ode.solve(xs[i][0], ts[i]) for i in range(n_samples)], axis=0)
    raw_ys = np.stack([ddeint(mackey_glass, lambda t: xs[i][0], ts[i], fargs=(beta, gamma, tau, n)).flatten() for i in range(n_samples)], axis=0)

    # add noise
    ys = raw_ys + gen.normal(0, noise_std, size=raw_ys.shape)

    dataset_name = f"mackey_glass_n={n_samples}_m={n_measurements}_noise={noise_std}_beta={beta}_gamma={gamma}_tau={tau}_n={n}"

    xs = xs/3.0
    ys = ys/3.0
    ts = ts/30.0

    return Dataset(dataset_name,xs, ts, ys)

import numpy as np
from scipy.stats import multivariate_normal

def gaussian_mixture_pdf(x, y, means, covariances, weights):
    """
    Evaluates the PDF of a 2D mixture of 3 Gaussians at point (x, y).
    
    Parameters:
    - x: x-coordinate of the point
    - y: y-coordinate of the point
    - means: A list of 2D means for each Gaussian component (3 x 2 array)
    - covariances: A list of 2D covariance matrices for each Gaussian component (3 x 2 x 2 array)
    - weights: A list of mixture weights for each Gaussian component (length 3)
    
    Returns:
    - The value of the mixture PDF at the point (x, y)
    """
    
    point = np.array([x, y])
    
    # Compute the PDF value of the mixture of 3 Gaussians
    pdf_value = 0.0
    for mean, cov, weight in zip(means, covariances, weights):
        pdf_value += weight * multivariate_normal.pdf(point, mean=mean, cov=cov)
    
    return pdf_value




def get_general_dataset(n_samples, n_measurements, noise_std, seed):

    initial_range = (-3,3)
    t_range = (0,5)

    # Example usage
    means = [np.array([0, 0]), np.array([3, 3]), np.array([-1, -2])]
    covariances = [np.array([[1, 0], [0, 1]]), np.array([[1, 0.0], [0.0, 1]]), np.array([[2, 0], [0, 2]])]
    weights = [0.4, -0.3, 0.3]

    def g(x,t):
        return gaussian_mixture_pdf(x[0], t, means, covariances, weights)

    def dfdt(x,t):
        return 50*g(x,t)


    xs = np.linspace(initial_range[0],initial_range[1],n_samples).reshape(-1,1)       
    ts = np.stack([np.linspace(t_range[0],t_range[1],n_measurements) for _ in range(n_samples)], axis=0)

    gen = np.random.default_rng(seed)
    raw_ys = np.stack([odeint(dfdt, xs[i][0], ts[i]).flatten() for i in range(n_samples)], axis=0)

    # add noise
    ys = raw_ys + gen.normal(0, noise_std, size=raw_ys.shape)

    xs = xs/3.0
    ys = ys/3.0
    ts = ts/5.0

    dataset_name = f"general_n={n_samples}_m={n_measurements}_noise={noise_std}"

    return Dataset(dataset_name,xs, ts, ys)


def get_integral_dataset(n_samples, n_measurements, noise_std, seed):

    initial_range = (-1,1)
    t_range = (0,5)

    from idesolver import IDESolver

    def solve_ide(x0):
        solver = IDESolver(
            x = np.linspace(0,5,200),
            y_0 = x0,
            c = lambda x,y: -2*y,
            d = lambda x: -5,
            k = lambda x,s: 1,
            f = lambda y: y,
            lower_bound = lambda x: 0,
            upper_bound = lambda x: x,
        )
        solver.solve()
   
        return solver.y[::10]

    xs = np.linspace(initial_range[0],initial_range[1],n_samples).reshape(-1,1)       
    ts = np.stack([np.linspace(t_range[0],t_range[1],n_measurements) for _ in range(n_samples)], axis=0)

    gen = np.random.default_rng(seed)
    raw_ys = np.stack([solve_ide(xs[i][0]).flatten() for i in range(n_samples)], axis=0)

    # add noise
    ys = raw_ys + gen.normal(0, noise_std, size=raw_ys.shape)

    dataset_name = f"integral_n={n_samples}_m={n_measurements}_noise={noise_std}"

    return Dataset(dataset_name,xs, ts, ys)




def get_real_tumor_dataset():

    import pickle

    with open('data/tumor.pkl', 'rb') as f:
        data = pickle.load(f)
    
    xs = data['xs']
    ts = data['ts']
    ys = data['ys']

    dataset_name = "tumor-real"

    return Dataset(dataset_name,xs, ts, ys)


def extract_data_from_one_dataframe(df):
        """
        This function extracts the data from one dataframe
        Args:
            df a pandas dataframe with columns ['id','x1','x2',...,'xM','t','y'] where the first M columns are the static features and the last two columns are the time and the observation
        Returns:
            X: a pandas dataframe of shape (D,M) where D is the number of samples and M is the number of static features
            ts: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
            ys: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
        """
        # TODO: Validate data

        ids = df['id'].unique()
        X = []
        ts = []
        ys = []
        for id in ids:
            df_id = df[df['id'] == id].copy()
            X.append(
                df_id.iloc[[0], 1:-2])
            # print(X)
            df_id.sort_values(by='t', inplace=True)
            ts.append(df_id['t'].values.reshape(-1))
            ys.append(df_id['y'].values.reshape(-1))
        X = pd.concat(X, axis=0, ignore_index=True)
        return X, ts, ys
    

def get_real_pharma_dataset(data_folder='data'):
    max_t = 12.5

    df = pd.read_csv(os.path.join(data_folder, "tacrolimus", "tac_pccp_mr4_250423.csv"))
    dosage_rows = df[df['DOSE'] != 0]
    assert dosage_rows['visit_id'].is_unique
    df.drop(columns=['DOSE', 'EVID','II', 'AGE'], inplace=True) # we drop age because many missing values. the other columns are not needed
    df.drop(index=dosage_rows.index, inplace=True) # drop dosage rows
    # Merge df with dosage rows on visit_id
    df = df.merge(dosage_rows[['visit_id', 'DOSE']], on='visit_id', how='left') # add dosage as a feature
    df.loc[df['TIME'] >= 168, 'TIME'] -= 168 # subtract 168 from time to get time since last dosage
    missing_24h = df[(df['TIME'] == 0) & (df['DV'] == 0)].index
    df.drop(index=missing_24h, inplace=True) # drop rows where DV is 0 and time is 0 - they correspond to missing 24h measurements

    dv_0 = df[df['TIME'] == 0][['visit_id', 'DV']]
    assert dv_0['visit_id'].is_unique
    df = df.merge(dv_0, on='visit_id', how='left', suffixes=('', '_0')) # add DV_0 as a feature

    more_than_t = df[df['TIME'] > max_t].index
    df.drop(index=more_than_t, inplace=True) # drop rows where time is greater than max_t

    df.dropna(inplace=True) # drop rows with missing values

    df = df[['visit_id'] + ['DOSE', 'DV_0', 'SEXE', 'POIDS', 'HT', 'HB', 'CREAT', 'CYP', 'FORMULATION'] + ['TIME', 'DV']]

    feature_names = ['DOSE', 'DV_0', 'SEX', 'WEIGHT', 'HT', 'HB', 'CREAT', 'CYP', 'FORM']

    df.columns = ['id'] + feature_names + ['t', 'y']

    X, ts, ys = extract_data_from_one_dataframe(df)

    global_t = np.array([0,0.33,0.67,1,1.5,2,3,4,6,9,12])

    new_ys = []
    new_ts = []

    for i in range(len(ts)):
        # interpolate the data
        t = ts[i]
        y = ys[i]
        y_interp = interp1d(t, y, kind='linear', fill_value='extrapolate')(global_t)
        new_ys.append(y_interp)
        new_ts.append(global_t)
    ts = np.stack(new_ts, axis=0)/12
    ys = np.stack(new_ys, axis=0)/20
    xs = ys[:,[0]]

    return Dataset('tacrolimus-real', xs, ts, ys)


def get_bifurcation_dataset(n_samples, n_measurements, noise_std, seed):

    initial_range = (-1, 2)
    t_range = (0,5)

    def dfdt(x,t, r=1):
        return r*x - x**2


    xs = np.linspace(initial_range[0],initial_range[1],n_samples).reshape(-1,1)       
    ts = np.stack([np.linspace(t_range[0],t_range[1],n_measurements) for _ in range(n_samples)], axis=0)

    gen = np.random.default_rng(seed)
    raw_ys = np.stack([odeint(dfdt, 1.0, ts[i], args=(xs[i][0],)).flatten() for i in range(n_samples)], axis=0)

    # add noise
    ys = raw_ys + gen.normal(0, noise_std, size=raw_ys.shape)

    dataset_name = f"bifurcation_n={n_samples}_m={n_measurements}_noise={noise_std}"

    return Dataset(dataset_name,xs, ts, ys)

def get_pk_dataset_extrapolation_x0(n_samples, n_measurements, noise_std, seed):

    tac = TacrolimusPK()
    default_covariates = np.array([[10,10,1,80,20,16,60,1,0]])

    v0 = np.linspace(21,30,n_samples)
    t = np.linspace(0,24,n_measurements)

    covariates = np.tile(default_covariates,(len(v0),1))
    covariates[:,1] = v0

    ys = np.stack([tac.predict(covariates[[i],:],t) for i in range(len(v0))],axis=0)/20
    ts = np.tile(t,(len(v0),1))/24
    xs = v0.reshape(-1,1)/20

    # add noise
    gen = np.random.default_rng(seed)
    ys = ys + gen.normal(0, noise_std, size=ys.shape)

    dataset_name = f"tacrolimus-extra-2_n={n_samples}_m={n_measurements}_noise={noise_std}"

    return Dataset(dataset_name,xs, ts, ys)


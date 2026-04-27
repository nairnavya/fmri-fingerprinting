import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression


# ----- LOADING MOTION REGRESSOR FILE FOR EACH .NII FILE -----
def load_motion_regressors(motion_path):
    if motion_path is None:
        return None

    return np.loadtxt(motion_path)


# ----- COLLECTING THREE CONFOUNDS FOR REGRESSION - MOTION, MEAN TIME COURSE, LINEAR TREND -----
def make_confounds(node_timeseries, motion_regressors=None):
    n_time = node_timeseries.shape[0]

    confounds = []

    if motion_regressors is not None:
        confounds.append(motion_regressors)
    # e.g. motion: (1200 × 12)

    global_signal = node_timeseries.mean(axis=1, keepdims=True) # "regression of the mean time course of global signal"
    confounds.append(global_signal)
    # e.g. global signal: (1200 × 1)

    linear_trend = np.linspace(-1, 1, n_time).reshape(-1, 1) # "removal of the linear trend"
    confounds.append(linear_trend)
    # e.g. trend: (1200 × 1)

    return np.hstack(confounds)
    # so np.hstack([motion, global, trend]) gives (1200 × 14)


# ----- REGRESSING THE THREE CONFOUNDS -----
def regress_confounds(node_timeseries, confounds):
    model = LinearRegression()
    model.fit(confounds, node_timeseries)
    predicted_noise = model.predict(confounds)
    cleaned = node_timeseries - predicted_noise

    return cleaned


# ----- PERFORMING LOWPASS FILTERING -----
def lowpass_filter(node_timeseries, tr=0.72, cutoff=0.1): # " low-pass filtering"
    # note: tr=0.72 means 0.72s between consecutive fMRI image acqusitions 
    fs = 1 / tr

    b, a = signal.butter(N=2, Wn=cutoff, btype="lowpass",fs=fs)
    filtered = signal.filtfilt(b, a, node_timeseries, axis=0)

    return filtered


# ----- CALLING PREPROCESSING FUNCTIONS -----
def preprocess_node_timeseries(node_timeseries, motion_regressors=None, tr=0.72):
    
    confounds = make_confounds(node_timeseries, motion_regressors)
    cleaned = regress_confounds(node_timeseries, confounds)
    filtered = lowpass_filter(cleaned, tr=tr)

    return filtered
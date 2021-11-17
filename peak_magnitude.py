#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Dict, List, Tuple
from xarray.core.dataarray import DataArray
from neuralhydrology.datautils import utils


# In[ ]:


# from https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py
def _validate_inputs(obs: DataArray, sim: DataArray):
    if obs.shape != sim.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(obs.shape) > 1) and (obs.shape[1] > 1):
        raise RuntimeError("Metrics only defined for time series (1d or 2d with second dimension 1)")
        
def _mask_valid(obs: DataArray, sim: DataArray) -> Tuple[DataArray, DataArray]:
    # mask of invalid entries. NaNs in simulations can happen during validation/testing
    idx = (~sim.isnull()) & (~obs.isnull())

    obs = obs[idx]
    sim = sim[idx]

    return obs, sim


# In[ ]:


# adjusted from "mean_peak_timing" function in 
# https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py
def mean_peak_magnitude(obs: DataArray,
                     sim: DataArray,
                     window: int = None,
                     resolution: str = '1D',
                     datetime_coord: str = None) -> float:

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations (scipy's find_peaks doesn't guarantee correctness with NaNs)
    obs, sim = _mask_valid(obs, sim)

    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(obs.values, distance=100, prominence=np.std(obs.values))

    # infer name of datetime index
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(obs)

    if window is None:
        # infer a reasonable window size
        window = max(int(utils.get_frequency_factor('12H', resolution)), 3)

    # evaluate timing
    abs_errors = []
    rel_errors = []
    for idx in peaks:
        # skip peaks at the start and end of the sequence and peaks around missing observations
        # (NaNs that were removed in obs & sim would result in windows that span too much time).
        if (idx - window < 0) or (idx + window >= len(obs)) or (pd.date_range(obs[idx - window][datetime_coord].values,
                                                                              obs[idx + window][datetime_coord].values,
                                                                              freq=resolution).size != 2 * window + 1):
            continue

        # check if the value at idx is a peak (both neighbors must be smaller)
        if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
            peak_sim = sim[idx]
        else:
            # define peak around idx as the max value inside of the window
            values = sim[idx - window:idx + window + 1]
            peak_sim = values[values.argmax()]

        # get xarray object of qobs peak, for getting the date and calculating the datetime offset
        peak_obs = obs[idx]

        # calculate the absolute difference between the peaks
        abs_diff = peak_obs - peak_sim

        rel_diff = abs_diff / peak_obs

        abs_errors.append(abs_diff)
        rel_errors.append(rel_diff)
        

    # count peaks
    peaks_sim_lower = len([x for x in abs_errors if x > 0])
    peaks_sim_higher = len([x for x in abs_errors if x < 0])

    if len(abs_errors) == 0:
        if len(rel_errors) == 0:
            results = [np.nan, np.nan]
        else: 
            results = [np.nan, np.mean(list(map(abs, rel_errors)))]
    elif len(rel_errors) == 0:
        results = [np.mean(list(map(abs, abs_errors))), np.nan]
    else:
        results = [np.mean(list(map(abs, abs_errors))), np.mean(list(map(abs, rel_errors)))]

    results_dict = {'abs_error': results[0],     # mean absolute error between simulated and observed peaks
                    'rel_error': results[1],     # mean relative error between simulated and observed peaks
                    'peaks': len(abs_errors),    # total number of identified peaks
                    'sim<obs': peaks_sim_lower,  # nr of peaks lower in simulation than in observed timeseries
                    'sim>obs': peaks_sim_higher} # nr of peaks higher in simulation than in observed timeseries
        
    return results_dict


import os
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import json

from processing import utility, quadrotorFM
from common import angleFuncs, solvers

def import_OB(rowIdx, logFile, g=9.81, doCGCorrection = True, useOBAttitude = False, acc_calib = 1/2048, quat_scale = 1/((127 << 6)-1), filterOBSpikes = True, **kwargs):
    filename = logFile.loc[rowIdx, 'Onboard Name']
    filedir = logFile.loc[rowIdx, 'OB Path']
    filepath = os.path.join(filedir, '{}.csv'.format(filename))
    if not os.path.exists(filepath):
        print('[ INFO ] Could not find .csv file for {}.'.format(filename))
        raise RuntimeError('File: {} not found.'.format(filename))
    print('[ INFO ] Importing On-Board data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))
    try:
        rawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', header=0)
    except pd.errors.ParserError:
        print('[ INFO ] I had trouble reading the .csv file, perhaps the header is unusually large, attempting to fix...')
        with open(os.path.join(filedir, '{}.csv'.format(filename)), 'r') as f:
            dd = f.readlines()
        headers = [d for d in dd if not d.split(',')[0].isnumeric()]
        usecols = len(headers[-1].split(','))
        headerLine = len(headers)
        rawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', header=headerLine - 1, usecols = np.arange(0, usecols, 1))

    colMapping = {
                'time':'t',                             # microseconds
                'pos[0]':'x_est',                       # mm
                'pos[1]':'y_est',                       # mm
                'pos[2]':'z_est',                       # mm
                'vel[0]':'u_est',                       # cm/s
                'vel[1]':'v_est',                       # cm/s
                'vel[2]':'w_est',                       # cm/s
                'extPos[0]':'x',                        # mm
                'extPos[1]':'y',                        # mm
                'extPos[2]':'z',                        # mm
                'extVel[0]':'u',                        # cm/s
                'extVel[1]':'v',                        # cm/s
                'extVel[2]':'w',                        # cm/s
                'quat[0]':'qw',                         # Need to scale by (127 << 6) - 1
                'quat[1]':'qx',                         # Need to scale by (127 << 6) - 1
                'quat[2]':'qy',                         # Need to scale by (127 << 6) - 1
                'quat[3]':'qz',                         # Need to scale by (127 << 6) - 1
                'heading[0]':'roll',                    # rad
                'heading[1]':'pitch',                   # rad
                'heading[2]':'yaw',                     # rad
                'gyroADC[0]':'p_filt',                  # deg/s
                'gyroADC[1]':'q_filt',                  # deg/s
                'gyroADC[2]':'r_filt',                  # deg/s
                'gyroADCafterRpm[0]':'p',               # deg/s
                'gyroADCafterRpm[1]':'q',               # deg/s
                'gyroADCafterRpm[2]':'r',               # deg/s
                'accSmooth[0]':'ax_filt',               # (1/2048) g 
                'accSmooth[1]':'ay_filt',               # (1/2048) g 
                'accSmooth[2]':'az_filt',               # (1/2048) g 
                'accADCafterRpm[0]':'ax',               # (1/2048) g 
                'accADCafterRpm[1]':'ay',               # (1/2048) g 
                'accADCafterRpm[2]':'az',               # (1/2048) g 
                'omega[0]':'w1_synced',                 # rad/s
                'omega[1]':'w2_synced',                 # rad/s
                'omega[2]':'w3_synced',                 # rad/s
                'omega[3]':'w4_synced',                 # rad/s
                'omega[4]':'w5_synced',                 # rad/s
                'omega[5]':'w6_synced',                 # rad/s
                'omega[6]':'w7_synced',                 # rad/s
                'omega[7]':'w8_synced',                 # rad/s
                'omegaUnfiltered[0]':'w1',              # rad/s
                'omegaUnfiltered[1]':'w2',              # rad/s
                'omegaUnfiltered[2]':'w3',              # rad/s
                'omegaUnfiltered[3]':'w4',              # rad/s
                'omegaUnfiltered[4]':'w5',              # rad/s
                'omegaUnfiltered[5]':'w6',              # rad/s
                'omegaUnfiltered[6]':'w7',              # rad/s
                'omegaUnfiltered[7]':'w8',              # rad/s
                'motor[0]':'w1_CMD',                    # -
                'motor[1]':'w2_CMD',                    # -
                'motor[2]':'w3_CMD',                    # -
                'motor[3]':'w4_CMD'                     # -
                }

    Data = rawData[colMapping.keys()].copy()
    Data.rename(columns=colMapping, inplace=True)

    # Remove any string-like information from loop iteration column. Not doing so causes some issues with pandas indexing
    try:
        indexes = list(Data['t'].index.get_level_values(0))
        badIndexes = [i + 1 for i, v in enumerate(indexes) if not v.replace(' ', '').isnumeric()] # i + 1 since we need to offset with the header
        if len(badIndexes):
            # We have some bad indexes which need to be removed
            usecols = np.arange(0, len(rawData.columns))
            rawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', header=0, skiprows=badIndexes, usecols=usecols)
            Data = rawData[colMapping.keys()].copy()
            Data.rename(columns=colMapping, inplace=True)
    except AttributeError:
        pass

    # Shift time back to 'zero'
    Data['t'] = (Data['t'] - Data.loc[0, 't']) * 1e-6 # Convert from microseconds to seconds

    # # Adjust for yaw offset
    # Data['yaw'] = Data['yaw'] - Data['yaw'][0]

    # Scale quat values from encoding
    Data[['qw', 'qx', 'qy', 'qz']] = Data[['qw', 'qx', 'qy', 'qz']] * quat_scale

    # Convert angles to rad
    if useOBAttitude:
        Data[['roll', 'pitch', 'yaw']] = Data[['roll', 'pitch', 'yaw']]*np.pi/180
        # Seems like betaflight 'yaw' is flipped; compare rates [p, q, r] with attitude [roll, pitch, yaw]
        # TODO: Check if Yaw is also flipped for INDI flight
        Data['yaw'] = -1*Data['yaw']    
    else:
        eul = angleFuncs.Quat2Eul(Data[['qw', 'qx', 'qy', 'qz']].to_numpy())
        Data[['roll', 'pitch', 'yaw']] = eul

    # Check for sudden spikes in euler angles
    if filterOBSpikes:
        attCols = ['roll', 'pitch', 'yaw']
        for col in attCols:
            Data[col] = filterSpikes(np.unwrap(Data[col].to_numpy().copy()), Data['t'].to_numpy(), sigma_dx = 6, window = 10).reshape(-1)

    # Correct acceleration values to m/s/s
    for acol in ['ax', 'ay', 'az', 'ax_filt', 'ay_filt', 'az_filt']:
        Data[acol] = Data[acol] * acc_calib * g

    # Correct velocity values to m/s (from cm/s)
    for vcol in ['u', 'v', 'w', 'u_est', 'v_est', 'w_est']:
        Data[vcol] = Data[vcol] * 1e-2

    # Correct position values to m (from mm)
    for vcol in ['x', 'y', 'z', 'x_est', 'y_est', 'z_est']:
        Data[vcol] = Data[vcol] * 1e-3

    # Correct gyro values to rad/s
    for rcol in ['p', 'q', 'r', 'p_filt', 'q_filt', 'r_filt']:
        Data[rcol] = Data[rcol] * np.pi/180

    # TODO: Account for time varying c.g. due to shifts in battery during (aggressive) flight. Use hover information throughout flight to
    #       get current best estimate of c.g.
    if doCGCorrection:
        configFile = logFile.loc[rowIdx, 'Drone Config File']
        configPath = logFile.loc[rowIdx, 'Drone Config Path']
        with open(os.path.join(configPath, configFile + '.json'), 'r') as f:
            droneParams = json.load(f)

        droneParams.update({'g':g})

        cg_xy = quadrotorFM.findCG_xy_omegaMethod(Data, droneParams, cutoff = 2)
        cg = np.hstack((cg_xy, 0))
        r = np.vstack((cg,)*len(Data))
        accIMU = Data[['ax', 'ay', 'az']].to_numpy()
        rates = Data[['p', 'q', 'r']].to_numpy()
        rates_dot = solvers.derivative(rates, Data['t'].to_numpy())
        accCG = accIMU - np.cross(rates, np.cross(rates, r)) - np.cross(rates_dot, r)

        Data[['ax', 'ay', 'az']] = accCG

    return Data


def filterSpikes(x, t, sigma_dx = 6, window = 25):
    # Cast x to 2-D array
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)
    for i in range(x.shape[1]):
        # First find outliers which are 6 standard deviations from the mean of dx
        dxi = solvers.derivative(x[:, i].reshape(-1, 1), t).reshape(-1)
        min_height = np.nanmean(np.abs(dxi)) + sigma_dx * np.nanstd(np.abs(dxi))
        # scipy.find_peaks returns index of peaks
        peaks, _ = find_peaks(np.abs(dxi), height=(min_height, None))
        # Remove 'window' samples on either side of the peak
        removeIdxs = []
        for p in peaks:
            idxs = np.arange(np.max([0, p - window]), np.min([len(x), p + window]), 1)
            removeIdxs += list(idxs)  
        removeIdxs = np.unique(removeIdxs)
        if len(removeIdxs):
            keepIdxs = np.delete(np.arange(0, len(x)), removeIdxs)
            func = interp1d(keepIdxs, x[keepIdxs, i], kind = 'cubic')
            # Check if removeIdxs are outside of keepIdxs bounds
            validRemoveIdxs = np.where((removeIdxs > keepIdxs[0]) & (removeIdxs < keepIdxs[-1]))[0]
            if len(validRemoveIdxs):
                x[removeIdxs[validRemoveIdxs], i] = func(removeIdxs[validRemoveIdxs])
            else:
                # Idxs to remove are outside of interpolation range, so apply zoh from nearest value
                closestIdx = np.argmin((keepIdxs - removeIdxs[0]))
                x[removeIdxs] = x[keepIdxs[closestIdx]]
    return x
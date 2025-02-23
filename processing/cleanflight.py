import os
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import json

from processing import utility, quadrotorFM
from common import angleFuncs, solvers


def V1_import_OB(rowIdx, logFile, g=9.81, rowBreak = 112, numCols = 55, BBLOptions = '[--raw --merge-gps --simulate-imu --debug --unit-acceleration m/s2 --unit-frame-time s --unit-rotation rad/s ]', **kwargs):
    filename = logFile.loc[rowIdx, 'Onboard Name']
    filedir = logFile.loc[rowIdx, 'OB Path']
    filepath = os.path.join(filedir, '{}.csv'.format(filename))
    if not os.path.exists(filepath):
        print('[ INFO ] Could not find .csv file for {}. Attempting to convert .bfl instead'.format(filename))
        try:
            utility.BBL2CSV(filedir, filename, index = -1, options = BBLOptions)
            print('[ INFO ] Successfully converted {}.bfl to .csv'.format(filename))
        except FileNotFoundError:
            raise RuntimeError('File: {} not found.')
    print('[ INFO ] Importing On-Board data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))
    configData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', nrows=int(rowBreak)-1, header=None).T
    configData.columns = configData.iloc[0]
    configData.drop(0, axis=0, inplace=True)
    rawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', header=int(rowBreak), usecols=np.arange(int(numCols)))

    colMapping = {
                'time':'t',               # ms
                'heading[0]':'roll',      # rad
                'heading[1]':'pitch',     # rad
                'heading[2]':'yaw',       # rad (with offset)
                'gyroADC[0]':'p',         # deg/s
                'gyroADC[1]':'q',         # deg/s
                'gyroADC[2]':'r',         # deg/s
                'accSmooth[0]':'ax',      # g
                'accSmooth[1]':'ay',      # g
                'accSmooth[2]':'az',      # g
                'debug[0]':'w1',          # rpm
                'debug[1]':'w2',          # rpm
                'debug[2]':'w3',          # rpm
                'debug[3]':'w4',           # rpm
                'motor[0]':'w1_CMD',                    # -
                'motor[1]':'w2_CMD',                    # -
                'motor[2]':'w3_CMD',                    # -
                'motor[3]':'w4_CMD'                     # -                
                }

    # minThrottle = int(configData.loc[1, 'minthrottle'])
    # maxThrottle = int(configData.loc[1, 'maxthrottle'])
    accCalib = int(configData.loc[1, 'acc_1G'])

    Data = rawData[colMapping.keys()].copy()
    Data.rename(columns=colMapping, inplace=True)

    # Convert time to seconds
    Data['t'] = Data.loc[:, 't'].apply(utility.milliseconds2seconds)
    # Shift time back to 'zero'
    Data['t'] = Data['t'] - Data.loc[0, 't']

    # Convert gyro rates to rad/s 
    # Note: rates still have bias term
    Data[['p', 'q', 'r']] = Data[['p', 'q', 'r']]*np.pi/180

    # Correct accelerations
    Data[['ax', 'ay', 'az']] = Data[['ax', 'ay', 'az']]/accCalib*g

    # # BetaFlight arbitrarily sets the yaw angle to around -4.4 radians when it turns on, regardless of quad orientation w.r.t earth
    # # Correct for yaw offset
    # Data['yaw'] = Data['yaw'] - Data['yaw'][0]

    # import matplotlib.pyplot as plt
    # from common import angleFuncs
    # # import code
    # # code.interact(local=locals())
    # fig = plt.figure()
    # ax = fig.add_subplot(311)
    # ax.plot(Data['t'], Data['p'], c = 'lightcoral', label = 'Roll rate')
    # ax.plot(Data['t'], Data['roll'], c = 'firebrick', label = 'Roll angle')
    # ax.set_ylabel(r'$\mathbf{Roll}$ [rad] (rate [rad/s])')
    # ax.legend()

    # ax = fig.add_subplot(312, sharex = ax)
    # ax.plot(Data['t'], Data['q'], c = 'bisque', label = 'Pitch rate')
    # ax.plot(Data['t'], Data['pitch'], c = 'darkorange', label = 'Pitch angle')
    # ax.set_ylabel(r'$\mathbf{Pitch}$ [rad] (rate [rad/s])')
    # ax.legend()

    # ax = fig.add_subplot(313, sharex = ax)
    # ax.plot(Data['t'], Data['r'], c = 'lightsteelblue', label = 'Yaw rate')
    # ax.plot(Data['t'], angleFuncs.unwrapPi(Data['yaw']), c = 'cornflowerblue', label = 'Yaw angle')
    # ax.set_ylabel(r'$\mathbf{Yaw}$ [rad] (rate [rad/s])')
    # ax.legend()
    # ax.set_xlabel(r'$\mathbf{Time}$ [s]')

    # plt.show()


    # Seems like betaflight 'yaw' is flipped; compare rates [p, q, r] with attitude [roll, pitch, yaw]
    Data['yaw'] = -1*Data['yaw']

    # Rotation from Betaflight axis system to conventional aerospace
    #   Betaflight: x-forward, y-left, z-up
    #   Aerospace: x-forward, y-right, z-down
    rotationX_Beta2Aero = angleFuncs.Eul2Quat(np.array([np.pi, 0, 0]))
    rotationY_Beta2Aero = angleFuncs.Eul2Quat(np.array([0, 0, 0]))
    rotationZ_Beta2Aero = angleFuncs.Eul2Quat(np.array([0, 0, 0]))
    quat_Beta2Aero = angleFuncs.quatMul(rotationZ_Beta2Aero, angleFuncs.quatMul(rotationY_Beta2Aero, rotationX_Beta2Aero))

    quat_att = angleFuncs.Eul2Quat(Data[['roll', 'pitch', 'yaw']].to_numpy())
    quat_EB2Aero = angleFuncs.quatMul(quat_Beta2Aero, quat_att)
    quat_EE2Aero = angleFuncs.QuatConj(angleFuncs.quatMul(quat_Beta2Aero, angleFuncs.QuatConj(quat_EB2Aero)))
    att_aero = angleFuncs.Quat2Eul(quat_EE2Aero)

    Data[['roll', 'pitch', 'yaw']] = att_aero

    toTransform = [
        ['p', 'q', 'r'],
        ['ax', 'ay', 'az']
        ]

    for vec in toTransform:
        _dBeta = Data[vec].to_numpy().reshape(-1, 3)
        _dAero = angleFuncs.QuatRot(quat_Beta2Aero, _dBeta)
        Data[vec] = _dAero

    return Data




def V2_import_OB(rowIdx, logFile, g=9.81, BBLOptions = '--merge-gps --simulate-imu --debug --unit-acceleration m/s2 --unit-frame-time s --unit-rotation rad/s --imu-ignore-mag', doCGCorrection = True, **kwargs):
    filename = logFile.loc[rowIdx, 'Onboard Name']
    filedir = logFile.loc[rowIdx, 'OB Path']
    filepath = os.path.join(filedir, '{}.csv'.format(filename))
    if not os.path.exists(filepath):
        print('[ INFO ] Could not find .csv file for {}. Attempting to convert .bfl instead'.format(filename))
        try:
            utility.BBL2CSV(filedir, filename, index = -1, options = BBLOptions)
            print('[ INFO ] Successfully converted {}.bfl to .csv'.format(filename))
        except FileNotFoundError:
            raise RuntimeError('File: {} not found.'.format(filename))
    print('[ INFO ] Importing On-Board data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))
    rawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', header=0)

    colMapping = {
                ' time (s)':'t',                         # s
                ' roll':'roll',                          # deg
                ' pitch':'pitch',                        # deg
                ' heading':'yaw',                        # deg (with offset)
                ' gyroADC[0] (rad/s)':'p',               # rad/s
                ' gyroADC[1] (rad/s)':'q',               # rad/s
                ' gyroADC[2] (rad/s)':'r',               # rad/s
                ' accSmooth[0] (m/s/s)':'ax',            # m/s/s
                ' accSmooth[1] (m/s/s)':'ay',            # m/s/s
                ' accSmooth[2] (m/s/s)':'az',            # m/s/s
                ' debug[0]':'w1',                        # erpm
                ' debug[1]':'w2',                        # erpm
                ' debug[2]':'w3',                        # erpm
                ' debug[3]':'w4',                        # erpm
                ' motor[0]':'w1_CMD',                    # -
                ' motor[1]':'w2_CMD',                    # -
                ' motor[2]':'w3_CMD',                    # -
                ' motor[3]':'w4_CMD'                     # -
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
    Data['t'] = Data['t'] - Data.loc[0, 't']

    # # Adjust for yaw offset
    # Data['yaw'] = Data['yaw'] - Data['yaw'][0]

    # Convert angles to rad
    Data[['roll', 'pitch', 'yaw']] = Data[['roll', 'pitch', 'yaw']]*np.pi/180

    # Seems like betaflight 'yaw' is flipped; compare rates [p, q, r] with attitude [roll, pitch, yaw]
    Data['yaw'] = -1*Data['yaw']    

    # Check for sudden spikes in euler angles
    attCols = ['roll', 'pitch', 'yaw']
    for col in attCols:
        Data[col] = filterSpikes(np.unwrap(Data[col].to_numpy().copy()), Data['t'].to_numpy(), sigma_dx = 6, window = 10).reshape(-1)

    # # Transform from betaflight axis system (x-forward, y-left, z-up) to conventional aerospace system (x-forward, y-right, z-down) 
    # # Rotate 180 degrees about x axis
    # flips = ['pitch', 'yaw', 'q', 'r', 'ay', 'az']
    # for f in flips:
    #     Data[f] = -1*Data[f]

    # Rotation from Betaflight axis system to conventional aerospace
    #   Betaflight: x-forward, y-left, z-up
    #   Aerospace: x-forward, y-right, z-down
    rotationX_Beta2Aero = angleFuncs.Eul2Quat(np.array([np.pi, 0, 0]))
    rotationY_Beta2Aero = angleFuncs.Eul2Quat(np.array([0, 0, 0]))
    rotationZ_Beta2Aero = angleFuncs.Eul2Quat(np.array([0, 0, 0]))
    quat_Beta2Aero = angleFuncs.quatMul(rotationZ_Beta2Aero, angleFuncs.quatMul(rotationY_Beta2Aero, rotationX_Beta2Aero))

    quat_att = angleFuncs.Eul2Quat(Data[['roll', 'pitch', 'yaw']].to_numpy())
    quat_EB2Aero = angleFuncs.quatMul(quat_Beta2Aero, quat_att)
    quat_EE2Aero = angleFuncs.QuatConj(angleFuncs.quatMul(quat_Beta2Aero, angleFuncs.QuatConj(quat_EB2Aero)))
    att_aero = angleFuncs.Quat2Eul(quat_EE2Aero)

    Data[['roll', 'pitch', 'yaw']] = att_aero

    toTransform = [
        ['p', 'q', 'r'],
        ['ax', 'ay', 'az']
        ]

    for vec in toTransform:
        _dBeta = Data[vec].to_numpy().reshape(-1, 3)
        _dAero = angleFuncs.QuatRot(quat_Beta2Aero, _dBeta)
        Data[vec] = _dAero

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
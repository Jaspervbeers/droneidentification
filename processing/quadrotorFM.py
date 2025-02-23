from common import solvers, angleFuncs
import numpy as np
from pandas import DataFrame

from scipy.signal import butter, filtfilt

def _ButterFilter(y, fs, order, cutoff_freq, filt_type, analog=False):
    '''Utility function to apply a butterworth filter to input data
    
    :param y: Data for which filter should be applied
    :param fs: Sampling frequency, in Hz
    :param order: Order of butterworth filter, as int
    :param cutoff_freq: Cutoff frequency, in Hz
    :param filt_type: Boolean, indicating if filter is low-pass ('low') or high-pass ('high')
    :param analog: Boolean, indiciating if analog (True) or digital (False) filter should be returned. Default is False.

    :return: Pass filtered data
    '''
    b, a = butter(order, cutoff_freq, btype=filt_type, analog=analog, fs=fs, output='ba')
    filt_X = filtfilt(b, a, y)
    return filt_X


def calcF(mass, acceleration):
    '''Function to calculate force based on Newton's second law
    
    :param mass: Mass, as float
    :param acceleration: Array-like data containing acceleration samples

    :param: Array of corresponding Forces
    '''
    return mass*acceleration



def calcM(Iv, rates, t, derivativeMethod = None, derivativeKwargs = None):
    '''Function to calculate moments of an object using first principles
    
    :param Iv: [3x3] Matrix of the moment of inertias of the object
    :param rates: Array-like structure containing the samples of the object's rotational rate. Shape is Nx3, N = number of observations
    :param t: 1-D array of time associated with body rate measurements
    :param derivativeMethod: Function used to differentiate the rotational rates to obtain the rotational acceleration. Function must take the rates as the only positional argument. All other parameters must be specified as keyword values in derivativeKwargs. Default is finite difference (Default = None).
    :param derivativeKwargs: Keyword arguments necessary for the chosen differentiation function (i.e. derivativeMethod)

    :return: Moments associated with measured rates
    '''
    if derivativeMethod is None:
        derivativeMethod = solvers._finiteDifference
        derivativeKwargs = {'t':t}
    d_rates = rates.copy()*0
    for r in range(rates.shape[1]):
        d_rates[:, r] = derivativeMethod(rates[:, r], **derivativeKwargs)
    # I_alpha = Iv * d_rates.T
    # I_omega = Iv * rates.T
    I_alpha = np.matmul(Iv, d_rates.T)
    I_omega = np.matmul(Iv, rates.T)
    M_gyro = np.cross(rates, I_omega, axisa=-1, axisb=0)
    return np.array(I_alpha.T + M_gyro)



def addMoments(droneParams, filteredData):
    moments = calcM(droneParams['Iv'], filteredData[['p', 'q', 'r']].to_numpy(), t = filteredData['t'].to_numpy())
    filteredData['Mx'] = moments[:, 0]
    filteredData['My'] = moments[:, 1]
    filteredData['Mz'] = moments[:, 2] 
    return filteredData



def getInducedVelocity(filteredData, logFile, rowIdx, droneParams):
    def checkIfReal(val):
        imPart = float(val.imag)
        if np.isclose(imPart, 0):
            rePart = float(val.real)
        else:
            rePart = np.nan
        return rePart
    
    rotorRadius = float(droneParams['R'])
    airDensity = float(droneParams['rho'])
    g = float(droneParams['g'])
    mass = float(logFile.loc[rowIdx, 'Mass'])
    numRotors = 4

    thurstHover_est = mass*g

    inducedVelocityHover = np.sqrt(thurstHover_est/(2*airDensity*numRotors*(2*np.pi*rotorRadius**2)))

    u = filteredData['u'].to_numpy()
    v = filteredData['v'].to_numpy()
    w = filteredData['w'].to_numpy()
    V = np.sqrt(np.square(u) + np.square(v) + np.square(w))
    v_in_vals = np.zeros(len(u))
    for i in range(len(v_in_vals)):
        # coeff = [4, -8*w[i], 4*V[i]**2, 0, -1*inducedVelocityHover**2]
        coeff = [1, -2*w[i], V[i]**2, 0, -1*inducedVelocityHover**4]
        try:
            roots = np.roots(coeff)
        except np.linalg.LinAlgError:
            roots = []
        if len(roots):
            if len(roots) > 1:
                val = np.nan
                diff = 100000
                for j in roots:
                    _val = checkIfReal(j)
                    if not np.isnan(_val):
                        if i > 0:
                            _diff = np.abs(_val - v_in_vals[i-1])
                        else:
                            # _diff = np.abs(_val - V[i])
                            _diff = np.abs(_val - inducedVelocityHover)
                        if _diff < diff:
                            val = _val
                            diff = _diff
                v_in_vals[i] = val
            else:
                v_in_vals[i] = checkIfReal(roots[0])
        else:
            v_in_vals[i] = np.nan
    return v_in_vals



def addControlMoments(filteredDataList, rotorConfig, r_sign, rotorDir, N_rot = 4):
    DataList = []
    for filteredData in filteredDataList:
        _Data = filteredData.copy(deep=True)
        omega = filteredData[['w1', 'w2', 'w3', 'w4']].to_numpy()*2*np.pi/60
        omegaCMD = filteredData[['w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD']].to_numpy()*2*np.pi/60
        omega_tot = np.sum(omega, axis=1)
        omega2 = filteredData[['w2_1', 'w2_2', 'w2_3', 'w2_4']].to_numpy()*(2*np.pi/60)**2
        w_avg = np.array(np.sqrt(np.sum(omega2, axis = 1)/N_rot)).reshape(-1, 1)
        _Data['w_tot'] = omega_tot
        _Data['w_avg'] = w_avg
        _Data['w1'] = omega[:, 0]
        _Data['w2'] = omega[:, 1]
        _Data['w3'] = omega[:, 2]
        _Data['w4'] = omega[:, 3]

        _Data['w1_CMD'] = omegaCMD[:, 0]
        _Data['w2_CMD'] = omegaCMD[:, 1]
        _Data['w3_CMD'] = omegaCMD[:, 2]
        _Data['w4_CMD'] = omegaCMD[:, 3]        

        _Data['w2_1'] = omega2[:, 0]
        _Data['w2_2'] = omega2[:, 1]
        _Data['w2_3'] = omega2[:, 2]
        _Data['w2_4'] = omega2[:, 3]

        # Define control moments
        # -> u_p,q,r = linear difference in rotor speeds: e.g. w1 + w2 - w3 - w4
        # -> U_p,q,r = quadratic difference in rotor speeds: e.g. w1^2 + w2^2 - w3^2 - w4^2
        u_p = (omega[:, rotorConfig['front left'] - 1] + omega[:, rotorConfig['aft left'] - 1]) - (omega[:, rotorConfig['front right'] - 1] + omega[:, rotorConfig['aft right'] - 1])
        u_q = (omega[:, rotorConfig['front left'] - 1] + omega[:, rotorConfig['front right'] - 1]) - (omega[:, rotorConfig['aft left'] - 1] + omega[:, rotorConfig['aft right'] - 1])
        u_r = r_sign[rotorDir]*((omega[:, rotorConfig['front left'] - 1] + omega[:, rotorConfig['aft right']- 1]) - (omega[:, rotorConfig['front right'] - 1] + omega[:, rotorConfig['aft left'] - 1]))

        _Data[['u_p', 'u_q', 'u_r']] = DataFrame(np.vstack((u_p, u_q, u_r)).T, columns=['u_p', 'u_q', 'u_r'])

        U_p = (omega2[:, rotorConfig['front left'] - 1] + omega2[:, rotorConfig['aft left'] - 1]) - (omega2[:, rotorConfig['front right'] - 1] + omega2[:, rotorConfig['aft right'] - 1])
        U_q = (omega2[:, rotorConfig['front left'] - 1] + omega2[:, rotorConfig['front right'] - 1]) - (omega2[:, rotorConfig['aft left'] - 1] + omega2[:, rotorConfig['aft right'] - 1])
        U_r = r_sign[rotorDir]*((omega2[:, rotorConfig['front left'] - 1] + omega2[:, rotorConfig['aft right']- 1]) - (omega2[:, rotorConfig['front right'] - 1] + omega2[:, rotorConfig['aft left'] - 1]))

        _Data[['U_p', 'U_q', 'U_r']] = DataFrame(np.vstack((U_p, U_q, U_r)).T, columns=['U_p', 'U_q', 'U_r'])
        DataList.append(_Data)
    
    return DataList




def findCG_xyz_accMethod(data, cutoff = 2):
    # Assume accelerometer is only measuring gravity in the manoeuvre
    # Assume rigid body
    # Assume thrust acts only along body z axis
    # Assume IMU is flush w.r.t x-y plane
    t = data['t'].to_numpy()
    dt = t[1] - t[0]
    fs = int(1/dt)
    rates = data[['p', 'q', 'r']]
    accIMU = data[['ax', 'ay', 'az']]
    att = data[['roll', 'pitch', 'yaw']]
    rates_filt = np.zeros(rates.shape)
    accIMU_filt = np.zeros(accIMU.shape)
    att_filt = np.zeros(att.shape)
    for (_rate, _acc, _att, i) in zip(rates.columns, accIMU.columns, att.columns, range(len(rates_filt))):
        rates_filt[:, i] = _ButterFilter(rates[_rate].to_numpy(), fs, 4, cutoff, 'low')
        accIMU_filt[:, i] = _ButterFilter(accIMU[_acc].to_numpy(), fs, 4, cutoff, 'low')
        att_filt[:, i] = _ButterFilter(att[_att].to_numpy(), fs, 4, cutoff, 'low')


    rates_dot = solvers.derivative(rates_filt, t)
    quat = angleFuncs.Eul2Quat(att_filt)

    gVector = np.zeros((len(data), 3))
    gVector[:, 2] = g
    gB = angleFuncs.QuatRot(quat, gVector, rot='E2B')

    # Find hovering flight, assume where all angles approx 0 is hovering
    hoverIdxs = np.where(np.isclose(gB[:, 2], g, rtol = 0.0001))
    wi = data[['w1', 'w2', 'w3', 'w4']].to_numpy()[hoverIdxs, :]
    azHover = accIMU_filt[hoverIdxs, 2]
    wi2 = np.sum(np.square(wi).reshape(-1, 4), axis = 1).reshape(-1, 1)
    kappa0 = np.matmul(np.matmul(np.linalg.inv(np.matmul(wi2.T, wi2)), wi2.T), azHover.reshape(-1, 1)).__array__()[0][0]
    # wHover = np.sqrt((-g)/(4*kappa0))

    omega = data[['w1', 'w2', 'w3', 'w4']]
    omega_filt = np.zeros(omega.shape)
    for i, o in enumerate(omega.columns):
        omega_filt[:, i] = _ButterFilter(omega[o].to_numpy(), fs, 4, cutoff, 'low')

    azT = kappa0 * np.sum(np.square(omega_filt), axis = 1)
    acc = gB.copy()
    acc[:, 2] = -azT
    deltaAcc = acc + accIMU_filt

    A11 = np.array(
        [
        rates_filt[:, 1]**2 + rates_filt[:, 2]**2, 
        rates_dot[:, 2] - rates_filt[:, 1]*rates_filt[:, 0], 
        -1*(rates_filt[:, 2]*rates_filt[:, 0] + rates_dot[:, 1])
        ]
        )
    A22 = np.array(
        [
        -1*(rates_filt[:, 0]*rates_filt[:, 1] + rates_dot[:, 2]),
        rates_filt[:, 0]**2 + rates_filt[:, 2]**2,
        rates_dot[:, 0] - rates_filt[:, 2]*rates_filt[:, 1]
        ]
        )
    A33 = np.array(
        [
        rates_dot[:, 1] - rates_filt[:, 0]*rates_filt[:, 2],
        -1*(rates_filt[:, 1]*rates_filt[:, 2] + rates_dot[:, 0]),
        rates_filt[:, 0]**2 + rates_filt[:, 1]**2
        ]
        )

    A_stack = np.zeros((3*len(A11.T), 3))
    A_stack[0:(3*len(A11.T)):3] = A11.T
    A_stack[1:(3*len(A11.T)):3] = A22.T
    A_stack[2:(3*len(A11.T)):3] = A33.T
    A = A_stack.reshape((len(A11.T), 3, 3))

    # +ve means c.g. in +ve axis direction 
    r = np.matmul(np.linalg.inv(A), deltaAcc.reshape(-1, 3, 1)).reshape(-1, 3)

    return np.nanmedian(r, axis = 0)


def findCG_xy_omegaMethod(data, droneParams, cutoff = 2):
    t = data['t'].to_numpy()
    dt = t[1] - t[0]
    fs = int(1/dt)
    rates = data[['p', 'q', 'r']]
    rates_filt = np.zeros(rates.shape)
    att = data[['roll', 'pitch', 'yaw']]
    rates_filt = np.zeros(rates.shape)
    att_filt = np.zeros(att.shape)
    for (_rate, _att, i) in zip(rates.columns, att.columns, range(len(rates_filt))):
        rates_filt[:, i] = _ButterFilter(rates[_rate].to_numpy(), fs, 4, cutoff, 'low')
        att_filt[:, i] = _ButterFilter(att[_att].to_numpy(), fs, 4, cutoff, 'low')

    rates_dot = solvers.derivative(rates_filt, t)

    quat = angleFuncs.Eul2Quat(att_filt)
    gVector = np.zeros((len(data), 3))
    gVector[:, 2] = droneParams['g']
    gB = angleFuncs.QuatRot(quat, gVector, rot='E2B')

    # Find hovering flight, assume where all angles approx 0 is hovering
    hoverIdxs = np.where(np.isclose(gB[:, 2], droneParams['g'], rtol = 0.0001))

    omega = data[['w1', 'w2', 'w3', 'w4']]
    omega_filt = np.zeros(omega.shape)
    for i, o in enumerate(omega.columns):
        omega_filt[:, i] = _ButterFilter(omega[o].to_numpy(), fs, 4, cutoff, 'low')

    # Assume c.g. along z direction is at centroid
    w1 = omega_filt[:, droneParams['rotor config']['aft right']-1]
    w2 = omega_filt[:, droneParams['rotor config']['front right']-1]
    w3 = omega_filt[:, droneParams['rotor config']['aft left']-1]
    w4 = omega_filt[:, droneParams['rotor config']['front left']-1]

    ell = float(droneParams['b'])

    idxsX = np.intersect1d(np.where(np.isclose(rates_dot[:, 1], 0, atol = 0.01)), hoverIdxs)
    idxsY = np.intersect1d(np.where(np.isclose(rates_dot[:, 0], 0, atol = 0.01)), hoverIdxs)
    
    # -1 here to align with axis direction definition s.t. +ve deltaCG means cg is in +ve x (or y) direction
    deltaCG_Y = -1*((w1 + w2 - w3 - w4)*ell/(w4 + w3 + w1 + w2)).reshape(-1)[idxsY]
    deltaCG_X = -1*((w1 + w3 - w2 - w4)*ell/(w4 + w2 + w1 + w3)).reshape(-1)[idxsX]
    if not len(deltaCG_Y):
        deltaCG_Y = 0
        print('[ WARNING ] Could not approximate c.g. location (y-dir)')
    if not len(deltaCG_X):
        deltaCG_X = 0
        print('[ WARNING ] Could not approximate c.g. location (x-dir)')

    return np.array(([np.nanmedian(deltaCG_X), np.nanmedian(deltaCG_Y)]))


def scaleFromTo(x, fromMin, fromMax, toMin, toMax):
    return (x-fromMin)/(fromMax - fromMin) * (toMax - toMin) + toMin


# def estimateActuatorDynamics(data, threshold = (2*np.pi/60) * 50):
#     t = data['t'].to_numpy()
#     dt = np.nanmean(t[1:] - t[:-1])
#     rotors = ['w1', 'w2', 'w3', 'w4']
#     taus = {}
#     for r in rotors:
#         idxs = np.where(np.abs(data[r] - data[r + '_CMD']) <= threshold)[0]
#         w = data[r].to_numpy()[idxs]
#         wCMD = data[r + '_CMD'].to_numpy()[idxs]
#         # Assume the form y = alpha * x
#         #   where:
#         #       y = w_i+1 - w_i
#         #       x = wCMD_i - w_i
#         #       alpha = 1 - e^(-tau * dt)
#         y = np.matrix(w[1:] - w[:-1]).T
#         x = np.matrix(wCMD[:-1] - w[:-1]).T
#         try:
#             alpha = np.linalg.inv(x.T*x)*x.T*y
#             tau = (-1*np.log(1-alpha)/dt).__array__()[0][0]
#         except np.linalg.LinAlgError:
#             tau = np.nan
#         taus.update({r:tau})
#     return taus



def estimateActuatorDynamics(data, threshold = (2*np.pi/60) * 50):
    t = data['t'].to_numpy()
    dt = t[1] - t[0]
    fs = int(1/dt)
    rotors = ['w1', 'w2', 'w3', 'w4']
    taus = {}
    for r in rotors:
        idxs = np.where(np.abs(data[r] - data[r + '_CMD']) <= threshold)[0]
        w = _ButterFilter(data[r].to_numpy(), fs, 4, 10, 'low')
        wCMD = _ButterFilter(data[r + '_CMD'].to_numpy(), fs, 4, 10, 'low')
        w_dot = _ButterFilter(solvers.derivative(data[r].to_numpy().reshape(-1, 1), t).reshape(-1), fs, 4, 10, 'low')
        try:
            wCMD = scaleFromTo(wCMD, np.nanmin(wCMD), np.nanmax(wCMD), np.nanmin(w), np.nanmax(w))
            y = np.matrix(w_dot[idxs]).T
            x = np.matrix(wCMD[idxs] - w[idxs]).T
            theta = np.linalg.inv(x.T*x)*x.T*y
            tau = (1/theta).__array__()[0][0]
        except (np.linalg.LinAlgError, ValueError) as e:
            tau = np.nan
        taus.update({r:tau})
    return taus
import numpy as np
from tqdm import tqdm
import scipy.signal as signal
import pandas as pd

from common import solvers, angleFuncs
from processing import quadrotorFM

def Continuous2DiscreteAB(A, B, dt, approximation_terms = 5):
    '''Function to convert continuous model to discrete time
    :param A: State space matrix
    :param B: Input matrix
    :param dt: Sampling time, in seconds
    :param approximation_terms: Number of terms, as int, in Taylor series expansion used to compute discrete variant. Default = 5. 
    :return: Tuple of discrete (A, B)
    '''
    # Due to the nature of np.linalg, A needs to be a square matrix for operations to work
    # Here, we artificially make A a square matrix - if it is not - to perform necessary calculations
    # Note that this augmentation does not affect the result
    checkA = True
    counter = 0
    while checkA:
        if A.shape[0] == A.shape[1]:
            checkA = False
        else:
            A = np.vstack((A, np.zeros(A.shape[1])))
            B = np.vstack((B, np.zeros(B.shape[1])))
            counter += 1
    # Define discrete counterparts
    n = A.shape[0]
    Ak = np.eye(n)
    Bk = B*dt
    # Compute Taylor up to approximation_terms
    for i in range(approximation_terms):
        Ak += 1/np.math.factorial(i + 1) * A ** (i + 1) * dt ** (i + 1)
        Bk += np.matmul(1/np.math.factorial(i + 2) * A ** (i + 1), B * dt ** (i + 2))
    # Revert A back to its original shape
    if counter > 0:
        Ak = Ak[:-counter]
    return np.matrix(Ak), np.matrix(Bk)



def ExtendedKalmanFilter(Data, IC, Noise, dt, Funcs, Discretize = True, ignoreDivergence = False):
    '''Extended Kalman Filter used to estimate the state vector from noisy measurement data

    :param Data: Dictionary containing the measurement data. The keys are:

    - 'Z' = Measured state(s) of interest for filtering with shape NxM where N = number of observations and M the number of states
    - 'U' = Measured state derivative(s) with shape NxM
    - 't' = Associated time array

    :param IC: Dictionary of initial conditions with keys:

    - 'X0' = Complete initial state vector with shape 1xM where M is the number of states
    - 't0' = Initial time, as float
    - 'P00' = Initial covariance matrix of state estimation error with shape MxM

    :param Noise: Dictionary of noise statistics information with keys:

    - 'Q' = Process noise matrix, shape MxM
    - 'R' = Measurement noise matrix, shape NxN where N corresponds to the size of 'U'

    :param dt: Time step, as float
    :param Funcs: Dictionary containing the necessary process-specific functions, with keys:

    - 'Pred_X' = State equation
    - 'Pred_X_Args' = Dictionary of keyword arguments required by state equation, if any
    - 'Pred_Z' = Measurement equation 
    - 'Pred_Z_Args' = Dictionary of keyword arguments required by measurement equation, if any
    - 'J_Fx' = Function to obtain Jacobian of state equation
    - 'J_Fx_Args' = Dictionary of keyword arguments required by Jacobian of state equation, if any
    - 'J_Hx' = Function to obtain Jacobian of measurement equation
    - 'J_Hx_Args' = Dictionary of keyword arguments required by Jacobian of measurement equation, if any
    - 'G' = System noise input matrix function with shape MxP where M is the number of states and P are the noise sources
    - 'G_Args' = Dictionary of keyword arguments required by System noise input matrix function, if any

    :param Discretize: Boolean to indicate if data is discrete, default is True
    :param ignoreDivergence: Boolean to ignore sudden divergence of filter for cases where process equations are insufficient to describe dynamics. WARNING only use this if you are confident in the results and review these properly. Default is False. Setting this to true essentially negates the filter and may lead to erroneous results.  

    :return: State estimate, Measurement estimate, Error covariance, and standard deviation of estimated states
    '''
    # Initialize filter parameters
    K = len(Data['U'])

    #   Initial conditions
    x_k1_k1 = IC['X0']
    t_k1_k1 = IC['t0']
    P_k1_k1 = IC['P00']

    #   Noise characteristics
    # G = Noise['G']
    Q = np.matrix(Noise['Q'])
    R = np.matrix(Noise['R'])

    # Create empty arrays to store data
    State_Estimate = np.zeros((K, len(x_k1_k1)))
    Measurement_Estimate = np.copy(Data['Z'])*0
    Error_Covariance = np.zeros((K, len(x_k1_k1), len(x_k1_k1)))
    StateStandardDeviation = np.copy(State_Estimate)*0

    # print('\n')
    # print('#'*65)
    # print('{:<65}'.format('[ INFO ] Running Kalman Filter'))

    for k in tqdm(range(K)):
        # One step ahead prediction
        #   State prediction
        x_k1_k, t_k1_k1 = solvers.rk4(Funcs['Pred_X'], x_k1_k1, Data['U'][k, :], t_k1_k1, dt, Funcs['Pred_X_Args'])
        #   Measurement prediction
        z_k1_k = Funcs['Pred_Z'](x_k1_k, Data['U'][k, :], t_k1_k1, **Funcs['Pred_Z_Args'])

        # Jacobian calculations
        Fx = Funcs['J_Fx'](x_k1_k, Data['U'][k, :], t_k1_k1, **Funcs['J_Fx_Args'])
        Hx = Funcs['J_Hx'](x_k1_k, Data['U'][k, :], t_k1_k1, **Funcs['J_Hx_Args'])

        # Compute input matrix
        G = Funcs['G'](x_k1_k, Data['U'][k, :], t_k1_k1, **Funcs['G_Args'])

        # Discretization
        Phi, Gamma = Continuous2DiscreteAB(Fx, G, dt, approximation_terms=5)

        # Covariance matrix of state prediction error
        P_k1_k = Phi*P_k1_k1*Phi.T + Gamma*Q*Gamma.T

        # Kalman gain computation
        K_k1 = P_k1_k*Hx.T*((Hx*P_k1_k*Hx.T + R)**(-1))

        # Measurement Update
        # x_k1_k1 = x_k1_k + K_k1*(np.matrix(Data['Z'][k, :]).T - np.matrix(z_k1_k).T)
        x_k1_k1 = x_k1_k.reshape(-1, 1) + K_k1*(np.matrix(Data['Z'][k, :]).T - np.matrix(z_k1_k).T)

        # Covariance matrix of state estimation error
        P_k1_k1 = [np.eye(len(x_k1_k)) - K_k1*Hx]*P_k1_k

        # Record results
        State_Estimate[k, :] = np.array(x_k1_k1).reshape((len(x_k1_k1),))
        Measurement_Estimate[k, :] = np.array(z_k1_k)
        Error_Covariance[k, :] = np.matrix(P_k1_k1)
        invalidCov = np.where(np.diag(Error_Covariance[k, :]) < 0)
        if len(invalidCov[0]):
            if not ignoreDivergence:
                raise ValueError('[ ERROR ] Kalman filter has diverged (negative covariance detected). Results may be inaccurate. Set ignoreDivergence = True if you wish to ignore this.')
            else:
                print('[ WARNING ] Kalman filter has DIVERGED. Results may be inaccurate.')
                Error_Covariance[k, :] = Error_Covariance[k-1, :]
                P_k1_k1 = np.array(Error_Covariance[k-1, :]).reshape(P_k1_k1.shape)
        StateStandardDeviation[k, :] = np.array(np.sqrt(np.diag(Error_Covariance[k, :])))

    # print('#'*65)

    return State_Estimate, Measurement_Estimate, Error_Covariance, StateStandardDeviation



def PassFilter(Data, fs, filtParams, inplace=False):
    '''Function to apply a pass filter to data
    
    :param Data: Data to apply pass filter to, as a Pandas DataFrame
    :param fs: Sampling frequency, as float in Hz
    :param filtParams: Dictionary describing filter, keys are:

    - 'order' = Order of the butterworth filter
    - 'cutoff' = Cutoff frequency, as float in Hz
    - 'type' = Specify 'low' for low-pass filter or 'high' for high-pass filter

    :param inplace: Boolean indicating if filtered Data should replace entries of Data. Default = False, so Data is unmodified and function returns a copy of the filtered DataFrame

    :return: Pass filtered data, as pandas DataFrame with the same structure as inputted Data. 
    '''
    # If inplace, then modify Data directly, otherwise modify copy. 
    if not inplace:
        filtData = Data.copy(deep = True)
    else:
        filtData = Data

    for vrs in filtParams.keys():
        filtArgs = (fs, filtParams[vrs]['order'], filtParams[vrs]['cutoff'], filtParams[vrs]['type'])
        filtData[list(vrs)] = filtData[list(vrs)].apply(_ButterFilter, args=filtArgs)

    return filtData



def _ButterFilter(X, fs, order, cutoff_freq, filt_type, analog=False):
    '''Utility function to apply a butterworth filter to input data
    
    :param X: Data for which filter should be applied
    :param fs: Sampling frequency, in Hz
    :param order: Order of butterworth filter, as int
    :param cutoff_freq: Cutoff frequency, in Hz
    :param filt_type: Boolean, indicating if filter is low-pass ('low') or high-pass ('high')
    :param analog: Boolean, indiciating if analog (True) or digital (False) filter should be returned. Default is False.

    :return: Pass filtered data
    '''
    b, a = signal.butter(order, cutoff_freq, btype=filt_type, analog=analog, fs=fs, output='ba')
    filt_X = signal.filtfilt(b, a, X)
    return filt_X



def _NotchFilter(X, w0, Q, fs):
    '''Utility function to apply a notch filter to data
    
    :param X: Data for which filter should be applied
    :param w0: Notch frequency, in Hz
    :param Q: Quality factor, as float. Describes width of notch filter.
    :param fs: Sampling frequency

    :return: Notch filtered data
    '''
    b, a = signal.iirnotch(w0, Q, fs = fs)
    filt_X = signal.filtfilt(b, a, X)
    return filt_X




def findStationaryPeriod(RotorSpeeds, threshold = 100):
    max_idx = 0
    for i in range(RotorSpeeds.shape[1]):
        idx = np.where(RotorSpeeds[:, i] >= threshold)[0][0]
        if idx > max_idx:
            max_idx = idx
    return idx



def _getNoiseStats(noiseStats, key):
    mean = float(noiseStats[key]['mean'])
    std = float(noiseStats[key]['std'])
    return mean, std



def runFilter(rowIdx, rawData, droneMass, droneParams, logFile, removeGravityComponent = True, defaultNoise = False, showPlots = False, distrustFactor = 1, fuseOBOTAccelerations = True):
    g = droneParams['g']
    # Un-bind angles for filtering
    rawData['roll'] = angleFuncs.unwrapPi(rawData['roll'].to_numpy())
    rawData['pitch'] = angleFuncs.unwrapPi(rawData['pitch'].to_numpy())
    rawData['yaw'] = angleFuncs.unwrapPi(rawData['yaw'].to_numpy())

    idxStationary = findStationaryPeriod(rawData[['w1', 'w2', 'w3', 'w4']].to_numpy(), threshold = 300)
    # Apply EKF to estimate attitudes and biases
    # Define parameters needed for Kalman Filtering
    t = rawData['t'].to_numpy()
    U = rawData[['p', 'q', 'r', 'ax', 'ay', 'az']].to_numpy()
    Z = rawData[['roll', 'pitch', 'yaw', 'u', 'v', 'w']].to_numpy()
    KalmanData = {'t':t, 'U':U, 'Z':Z}

    # Define initial conditions and parameters
    X0 = np.hstack((Z[0, :], Z[0, :]*0)).reshape(-1, 1)
    t0 = t[0]
    stdX = 1
    P00 = np.eye(len(X0))*stdX
    IC = {'X0':X0, 't0':t0, 'P00':P00}

    # Use default noise statistics. will be changed if true statistics are provided later
    # Note, by default, the process noise is set up to distrust the model and trust the measurements more. 
    # This is reasonable when using OptiTrack, which is assumed to be near ground truth.
    # Also, the state equation is not of a high enough fidelity for our applications 
    # Process noise; OptiTrack [p, q, r, ax, ay, az, bias_dp, bias_dq, bias_dr, bias_dax, bias_day, bias_daz]
    Q = np.diag((3.06*(np.pi/180)**2, 7.20*(np.pi/180)**2, 7.12*(np.pi/180)**2, 44.21, 14.80, 24.20))

    # Measurement noise; OT [roll, pitch, yaw, u, v, w]
    R = np.square(np.diag((9.94e-4*(np.pi/180), 1.07e-3*(np.pi/180), 8.33e-4*(np.pi/180), 9.80e-3, 2.47e-3, 6.82e-3)))


    # If users opt not to use default noise characteristics, then obtain them from droneConfig file
    # NOTE: It is strongly recommended to use the actual noise statistics of the drone. Hence, defaultNoise = False
    if not defaultNoise:
        if 'imu noise statistics' in droneParams.keys():
            print('[ INFO ] Found drone specific noise statistics in config file, using these over default.')
            noiseStats = droneParams['imu noise statistics']
            # Process noise; OptiTrack [p, q, r, ax, ay, az, bias_dp, bias_dq, bias_dr, bias_dax, bias_day, bias_daz]
            # Can even make OptiTrack 0 
            Q = np.diag(
                (_getNoiseStats(noiseStats, 'p')[1]**2*distrustFactor, _getNoiseStats(noiseStats, 'q')[1]**2*distrustFactor, _getNoiseStats(noiseStats, 'r')[1]**2*distrustFactor,
                _getNoiseStats(noiseStats, 'ax')[1]**2*distrustFactor, _getNoiseStats(noiseStats, 'ay')[1]**2*distrustFactor, _getNoiseStats(noiseStats, 'az')[1]**2*distrustFactor)
                )

            # Measurement noise; OT [roll, pitch, yaw, u, v, w]
            R = np.square(np.diag((9.94e-4*(np.pi/180), 1.07e-3*(np.pi/180), 8.33e-4*(np.pi/180), 9.80e-3, 2.47e-3, 6.82e-3)))
        else:
            print('[ WARNING ] Could not find drone-specific noise statistics. Using default values. Kalman filtering results will likely not estimate noise well.')


    Noise = {'Q':Q, 'R':R}

    Funcs = {'Pred_X':state_eq, 'Pred_X_Args':{'g':g},
            'Pred_Z':measurement_eq, 'Pred_Z_Args':{'v':np.diag(R)}, 
            'J_Fx':Jx_state_eq, 'J_Fx_Args':{},
            'J_Hx':Jx_measurement_eq, 'J_Hx_Args':{'v':np.diag(R)},
            'G':input_matrix, 'G_Args':{}}

    # Run Kalman filter
    dt = t[1] - t[0]
    X_est, Z_est, Cov, Std = ExtendedKalmanFilter(KalmanData, IC, Noise, dt, Funcs, ignoreDivergence=True)

    filtResults = {'X_est':X_est, 'Z_est':Z_est, 'Cov':Cov, 'Std':Std}

    # Note due to incomplete process model (i.e. unsuitable for high velocities) we can only make estimates
    # on the bias based on hovering or pre-flight measurements. It is assumed that the bias remains constant
    # for the duration of the flight.
    # Check if there are sufficient points for estimating biases
    if int(0.90*idxStationary):
        if int(0.90*idxStationary) < 100:
            print('[ WARNING ] There are too few points in the stationary period. Cannot reliably estimate biases.')
            U_corrected = U
        else:
            biases = X_est[:int(0.90*idxStationary), 6:12]
            correction = np.nanmean(biases[int(0.8*len(biases)):], axis = 0)
            # + Correction because biases are implemented as -ve in the state equations
            U_corrected = U + correction
    else:
        print('[ WARNING ] No stationary period was found. Cannot reliably estimate biases.')
        # biases = X_est[-1*int(0.1*len(X_est)):, 6:12]
        # correction = np.nanmean(biases[int(0.8*len(biases)):], axis = 0)      
        U_corrected = U
    

    # # Apply filter to IMU (unbiased) measurements
    # U_colNames = ['p', 'q', 'r', 'ax', 'ay', 'az']
    # U_corrected = pd.DataFrame(U_corrected, columns=U_colNames)

    # filtParams = {
    #             ('p', 'q', 'r'):{'type':'low',
    #                             'cutoff':100,
    #                             'order':4},
    #             ('ax', 'ay', 'az'):{'type':'low',
    #                                 'cutoff':100,
    #                                 'order':4}
    #                 }
    # fs = 1/dt

    # U_filt = PassFilter(U_corrected, fs, filtParams)

    # Apply filter to IMU (unbiased) measurements
    U_colNames = ['p', 'q', 'r', 'ax', 'ay', 'az']
    U_corrected = pd.DataFrame(U_corrected, columns=U_colNames)

    filtParams = {
                ('p', 'q', 'r'):{'type':'low',
                                'cutoff':100,
                                'order':4},
                ('ax', 'ay', 'az'):{'type':'low',
                                    'cutoff':100,
                                    'order':4}
                    }
    fs = 1/dt

    U_filt = PassFilter(U_corrected, fs, filtParams)


    rColNames = ['w1', 'w2', 'w3', 'w4', 'w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD']
    rFilt = rawData[rColNames].copy(deep=True)
    rfiltParams = {
        ('w1', 'w2', 'w3', 'w4'):{
            'type':'low',
            'cutoff':100,
            'order':4
        },
        ('w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD'):{
            'type':'low',
            'cutoff':100,
            'order':4
        }
    }

    rFilt = PassFilter(rFilt, fs, rfiltParams)


    # # Sensor fusion of accelerometer data
    # if fuseOBOTAccelerations:
    #     K = len(t)
    #     accOT = solvers.derivative(X_est[:, 3:6], t)
    #     accOB = U_filt[['ax', 'ay', 'az']].to_numpy()
    #     gComponent = np.vstack((np.array([0, 0, g]),)*K)
    #     quatE2B = angleFuncs.QuatConj(angleFuncs.Eul2Quat(X_est[:, 0:3]))
    #     gB = angleFuncs.QuatRot(quatE2B, gComponent, rot = 'E2B')
    #     accOB = accOB + gB
    #     _Q = Q[3:, 3:]
    #     _R = R[3:, 3:]

    #     x_k1_k1 = accOB[0, :]*0
    #     z_k1_k1 = accOT[0, :]*0
    #     P_k1_k1 = np.eye(len(x_k1_k1))

    #     _accEst = np.zeros(accOB.shape)
    #     _accErrorCovariance = np.zeros((K, len(x_k1_k1), len(x_k1_k1)))
    #     for k in tqdm(range(K)):
    #         # One step ahead prediction
    #         #   State prediction
    #         x_k1_k = accOB[k]

    #         # Jacobian calculations
    #         Fx = np.eye(len(x_k1_k))
    #         Hx = np.eye(len(x_k1_k))

    #         z_k1_k = np.matmul(Hx, x_k1_k)

    #         # Compute input matrix <- assume no cross-noise influence between x and z
    #         _G = np.eye(len(x_k1_k1))

    #         # Discretization
    #         Phi, Gamma = Continuous2DiscreteAB(Fx, _G, dt, approximation_terms=5)

    #         # Covariance matrix of state prediction error
    #         P_k1_k = Phi*P_k1_k1*Phi.T + Gamma*_Q*Gamma.T

    #         # Kalman gain computation
    #         K_k1 = P_k1_k*Hx.T*((Hx*P_k1_k*Hx.T + _R)**(-1))

    #         # Measurement Update
    #         x_k1_k1 = x_k1_k.reshape(-1, 1) + K_k1*(np.matrix(accOT[k, :]).T - np.matrix(z_k1_k).T)

    #         # Covariance matrix of state estimation error
    #         P_k1_k1 = [np.eye(len(x_k1_k)) - K_k1*Hx]*P_k1_k

    #         # Record results
    #         _accEst[k, :] = np.array(x_k1_k1).reshape((len(x_k1_k1),))
    #         _accErrorCovariance[k, :] = np.matrix(P_k1_k1)


    # Create filtered DataFrame
    filteredData = rawData.copy(deep=True)

    # Rotor speeds
    filteredData['w1'], filteredData['w2'], filteredData['w3'], filteredData['w4'] = rFilt['w1'], rFilt['w2'], rFilt['w3'], rFilt['w4']
    # Rotor speeds CMD
    filteredData['w1_CMD'], filteredData['w2_CMD'], filteredData['w3_CMD'], filteredData['w4_CMD'] = rFilt['w1_CMD'], rFilt['w2_CMD'], rFilt['w3_CMD'], rFilt['w4_CMD']
    # Add rotorspeeds squared
    filteredData['w2_1'], filteredData['w2_2'], filteredData['w2_3'], filteredData['w2_4'] = filteredData['w1'].apply(np.square), filteredData['w2'].apply(np.square), filteredData['w3'].apply(np.square), filteredData['w4'].apply(np.square)
    filteredData.reset_index(inplace=True)
    # accelerations
    filteredData['ax'], filteredData['ay'], filteredData['az'] = U_filt['ax'].to_numpy(), U_filt['ay'].to_numpy(), U_filt['az'].to_numpy()
    # rates
    filteredData['p'], filteredData['q'], filteredData['r'] = U_filt['p'].to_numpy(), U_filt['q'].to_numpy(), U_filt['r'].to_numpy()
    # velocities
    filteredData['u'], filteredData['v'], filteredData['w'] = X_est[:, 3], X_est[:, 4], X_est[:, 5]
    # attitude
    filteredData['roll'], filteredData['pitch'], filteredData['yaw'] = X_est[:, 0], X_est[:, 1], X_est[:, 2]

    # Derive forces
    F = quadrotorFM.calcF(droneMass, filteredData[['ax', 'ay', 'az']].to_numpy())
    if removeGravityComponent:
        # Remove gravity from forces

        # Get angles from E-frame to B-frame, in quaternions. Note, euler angles give B2E, so we take conjugate for E2B
        quatE2B = angleFuncs.QuatConj(angleFuncs.Eul2Quat(X_est[:, 0:3]))

        # Define Fg vector in E-frame (x-forward, y-right, and z-down)
        Fg = np.vstack((np.array([0, 0, g])*droneMass,)*len(filteredData))
        # Rotate into B-frame
        Fg_B = angleFuncs.QuatRot(quatE2B, Fg)

        # Elimintate effects of Fg from data (note, is +ve because of the way accelerometer works: facing g (and not moving) gives opposite acceleration)
        F_B = F + Fg_B

    else:
        F_B = F.copy()

    # Add forces to filtered data
    filteredData['Fx'], filteredData['Fy'], filteredData['Fz'] = F_B[:, 0], F_B[:, 1], F_B[:, 2]

    # Sanity check & plot results
    if showPlots:
        import matplotlib.pyplot as plt

        # Attitude
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(filteredData['t'], filteredData['roll'], color = 'firebrick', label = 'Roll (estm)')
        ax.plot(filteredData['t'], Z[:, 0], color = 'firebrick', linestyle = '--', alpha = 0.3, label = 'Roll (meas)')
        ax.plot(filteredData['t'], filteredData['pitch'], color = 'darkorange', label = 'Pitch (estm)')
        ax.plot(filteredData['t'], Z[:, 1], color = 'darkorange', linestyle = '--', alpha = 0.3, label = 'Pitch (meas)')
        ax.plot(filteredData['t'], filteredData['yaw'], color = 'royalblue', label = 'Yaw (estm)')
        ax.plot(filteredData['t'], Z[:, 2], color = 'royalblue', linestyle = '--', alpha = 0.3, label = 'Yaw (meas)')
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Attitude [rad]')

        # Velocity
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(filteredData['t'], filteredData['u'], color = 'firebrick', label = 'u (estm)')
        ax.plot(filteredData['t'], Z[:, 3], color = 'firebrick', linestyle = '--', alpha = 0.3, label = 'u (meas)')
        ax.plot(filteredData['t'], filteredData['v'], color = 'darkorange', label = 'v (estm)')
        ax.plot(filteredData['t'], Z[:, 4], color = 'darkorange', linestyle = '--', alpha = 0.3, label = 'v (meas)')
        ax.plot(filteredData['t'], filteredData['w'], color = 'royalblue', label = 'w (estm)')
        ax.plot(filteredData['t'], Z[:, 5], color = 'royalblue', linestyle = '--', alpha = 0.3, label = 'w (meas)')
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [m/s]')

        # Gyro biases
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(filteredData['t'], X_est[:, 6], color = 'firebrick', label = 'bias p (estm)')
        ax.plot(filteredData['t'], Std[:, 6], color = 'firebrick', label = 'bias p (conv)', linestyle = '--', alpha = 0.3)
        ax.plot(filteredData['t'], X_est[:, 7], color = 'darkorange', label = 'bias q (estm)')
        ax.plot(filteredData['t'], Std[:, 7], color = 'darkorange', label = 'bias q (conv)', linestyle = '--', alpha = 0.3)
        ax.plot(filteredData['t'], X_est[:, 8], color = 'royalblue', label = 'bias r (estm)')
        ax.plot(filteredData['t'], Std[:, 8], color = 'royalblue', label = 'bias r (conv)', linestyle = '--', alpha = 0.3)
        ylims = ax.get_ylim()
        ax.vlines(filteredData['t'][idxStationary], -100, 100, color = 'k', linestyle = '--', label = 'Bias estimation cutoff')
        ax.set_ylim(ylims)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Rate [rad/s]')

        # Accelerometer biases
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(filteredData['t'], X_est[:, 9], color = 'firebrick', label = 'bias ax (estm)')
        ax.plot(filteredData['t'], Std[:, 9], color = 'firebrick', label = 'bias ax (conv)', linestyle = '--', alpha = 0.3)
        ax.plot(filteredData['t'], X_est[:, 10], color = 'darkorange', label = 'bias ay (estm)')
        ax.plot(filteredData['t'], Std[:, 10], color = 'darkorange', label = 'bias ay (conv)', linestyle = '--', alpha = 0.3)
        ax.plot(filteredData['t'], X_est[:, 11], color = 'royalblue', label = 'bias az (estm)')
        ax.plot(filteredData['t'], Std[:, 11], color = 'royalblue', label = 'bias az (conv)', linestyle = '--', alpha = 0.3)
        ylims = ax.get_ylim()
        ax.vlines(filteredData['t'][idxStationary], -100, 100, color = 'k', linestyle = '--', label = 'Bias estimation cutoff')
        ax.set_ylim(ylims)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [m/s/s]')

        # Forces
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(filteredData['t'], filteredData['Fx'], color = 'firebrick', label = 'Fx (without g = {})'.format(removeGravityComponent))
        ax.plot(filteredData['t'], filteredData['Fy'], color = 'darkorange', label = 'Fy (without g = {})'.format(removeGravityComponent))
        ax.plot(filteredData['t'], filteredData['Fz'], color = 'royalblue', label = 'Fz (without g = {})'.format(removeGravityComponent))
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Force [N]')

        plt.show()

    return filteredData, filtResults



# Convert string to matrix. If x is a num, then return num. 
def str2mat(x):
    try:
        if len(x):
            x = np.matrix(x)
    except TypeError:
        pass
    return x



def state_eq(_x, _u, t, g = 9.81):
    # Reshape to speed up computations
    shape = _x.shape
    x = _x.__array__().reshape(-1)
    x_dot = np.zeros(x.shape)
    u = _u.reshape(-1)

    # Angular accelerations 
    # Phi dot
    x_dot[0] = (u[0] - x[6]) + (u[1] - x[7])*np.sin(x[0])*np.tan(x[1]) + (u[2]-x[8])*np.cos(x[0])*np.tan(x[1])
    # Theta dot
    x_dot[1] = (u[1]-x[7])*np.cos(x[0]) - (u[2]-x[8])*np.sin(x[0])
    # Psi dot
    x_dot[2] = (u[1]-x[7])*np.sin(x[0])*(1/np.cos(x[1])) + (u[2]-x[8])*np.cos(x[0])*(1/np.cos(x[1]))
    
    # Linear accelerations
    # u dot
    x_dot[3] = (u[2] - x[8])*x[4] - (u[1]-x[7])*x[5] - g*np.sin(x[1]) + u[3] - x[9]
    # v dot
    x_dot[4] = -(u[2] - x[8])*x[3] + (u[0]-x[6])*x[5] + g*np.sin(x[0])*np.cos(x[1]) + u[4] - x[10]
    # w dot
    x_dot[5] = (u[1]-x[7])*x[3] - (u[0]-x[6])*x[4] + g*np.cos(x[0])*np.cos(x[1]) + u[5] - x[11]

    return x_dot.reshape(shape)



def input_matrix(_x, _u, t, g = 9.81):
    x = _x.__array__().reshape(-1)
    u = _u.reshape(-1)
    G = np.zeros((len(x), len(u)))

    G = np.zeros((len(x), len(u)))

    # phi_dot
    #   w.r.t noise in p
    G[0, 0] = -1
    #   w.r.t noise in q
    G[0, 1] = -np.sin(x[0])*np.tan(x[1])
    #   w.r.t noise in r
    G[0, 2] = -np.cos(x[0])*np.tan(x[1])

    # theta_dot
    #   w.r.t noise in q
    G[1, 1] = -np.cos(x[0])
    #   w.r.t noise in r
    G[1, 2] = -np.sin(x[0])

    # psi_dot
    #   w.r.t noise in q
    G[2, 1] = -np.sin(x[0])*(1/np.cos(x[1]))
    #   w.r.t noise in r
    G[2, 2] = -np.cos(x[0])*(1/np.cos(x[1]))

    # u_dot
    #   w.r.t noise in q
    G[3, 1] = x[5]
    #   w.r.t noise in r
    G[3, 2] = -x[4]
    #   w.r.t noise in ax
    G[3, 3] = -1

    # v_dot
    #   w.r.t noise in p
    G[4, 0] = -x[5]
    #   w.r.t noise in r
    G[4, 2] = x[3]
    #   w.r.t noise in ay
    G[4, 4] = -1

    # w_dot
    #   w.r.t noise in p
    G[5, 0] = x[4]
    #   w.r.t noise in q
    G[5, 1] = -x[3]
    #   w.r.t noise in az
    G[5, 5] = -1

    return G



def Jx_state_eq(_x, _u, t, g = 9.81):
    x = _x.__array__().reshape(-1)
    u = _u.reshape(-1)
    Fx = np.zeros((len(x), len(x)))
    Fx = np.zeros((len(x), len(x)))

    # d(phi_dot)/d(phi)
    Fx[0, 0] = (u[1] - x[7])*np.cos(x[0])*np.tan(x[1]) - (u[2]-x[8])*np.sin(x[0])*np.tan(x[1])
    # d(phi_dot)/d(theta)
    Fx[0, 1] = (u[1] - x[7])*np.sin(x[0])*(1/np.cos(x[1]))**2 + (u[2]-x[8])*np.cos(x[0])*(1/np.cos(x[1]))**2
    # d(phi_dot)/d(bp)
    Fx[0, 6] = -1
    # d(phi_dot)/d(bq)
    Fx[0, 7] = -1*np.sin(x[0])*np.tan(x[1])
    # d(phi_dot)/d(br)
    Fx[0, 8] = -1*np.cos(x[0])*np.tan(x[1])


    # d(theta_dot)/d(phi)
    Fx[1, 0] = -1*(u[1]-x[7])*np.sin(x[0]) - (u[2]-x[8])*np.cos(x[0])
    # d(theta_dot)/d(bq)
    Fx[1, 7] = -np.cos(x[0])
    # d(theta_dot)/d(br)
    Fx[1, 8] = np.sin(x[0])


    # d(psi_dot)/d(phi)
    Fx[2, 0] = (u[1] - x[7])*np.cos(x[0])*(1/np.cos(x[1])) - (u[2] - x[8])*np.sin(x[0])*(1/np.cos(x[1]))
    # d(psi_dot)/d(theta)
    Fx[2, 1] = np.tan(x[1])*((u[1]-x[7])*np.sin(x[0])*(1/np.cos(x[1])) + (u[2]-x[8])*np.cos(x[0])*(1/np.cos(x[1])))
    # d(psi_dot)/d(bq)
    Fx[2, 7] = -np.sin(x[0])*(1/np.cos(x[1]))
    # d(psi_dot)/d(br)
    Fx[2, 8] = -np.cos(x[0])*(1/np.cos(x[1]))


    # d(u_dot)/d(theta)
    # Fx[3, 1] = g*np.cos(x[1])
    Fx[3, 1] = -1*g*np.cos(x[1])
    # d(u_dot)/d(v)
    Fx[3, 4] = u[2] - x[8]
    # d(u_dot)/d(w)
    Fx[3, 5] = -1*(u[1] - x[7])
    # d(u_dot)/d(bq)
    Fx[3, 7] = x[5]
    # d(u_dot)/d(br)
    Fx[3, 8] = -1*x[4]
    # d(u_dot)d(bax)
    Fx[3, 9] = -1


    # d(v_dot)/d(phi)
    Fx[4, 0] = g*np.cos(x[0])*np.cos(x[1])
    # Fx[4, 0] = -1*g*np.cos(x[0])*np.cos(x[1])
    # d(v_dot)/d(theta)
    Fx[4, 1] = -1*g*np.sin(x[0])*np.sin(x[1])
    # Fx[4, 1] = g*np.sin(x[0])*np.sin(x[1])
    # d(v_dot)/d(u)
    Fx[4, 3] = -1*(u[2] - x[8])
    # d(v_dot)/d(w)
    Fx[4, 5] = u[0] - x[6]
    # d(v_dot)/d(bp)
    Fx[4, 6] = -x[5]
    # d(v_dot)/d(br)
    Fx[4, 8] = x[3]
    # d(v_dot)/d(bay)
    Fx[4, 10] = -1


    # # d(w_dot)/d(phi)
    # Fx[5, 0] = g*np.sin(x[0])*np.cos(x[1])
    # # d(w_dot)/d(theta)
    # Fx[5, 1] = g*np.cos(x[0])*np.sin(x[1])
    # d(w_dot)/d(phi)
    Fx[5, 0] = -1*g*np.sin(x[0])*np.cos(x[1])
    # d(w_dot)/d(theta)
    Fx[5, 1] = -1*g*np.cos(x[0])*np.sin(x[1])
    # d(w_dot)/d(u)
    Fx[5, 3] = u[1] - x[7]
    # d(w_dot)/d(v)
    Fx[5, 4] = -1*(u[0] - x[6])
    # d(w_dot)/d(bp)
    Fx[5, 6] = x[4]
    # d(w_dot)/d(bq)
    Fx[5, 7] = -1*x[3]
    # d(w_dot)/d(baz)
    Fx[5, 11] = -1

    return Fx



def measurement_eq(x, u, t, v = None):
    z_pred = v.copy()*0

    # Angle measurement predictions
    z_pred[0] = x[0] + v[0]
    z_pred[1] = x[1] + v[1]
    z_pred[2] = x[2] + v[2]

    # Velocity measurement predictions
    z_pred[3] = x[3] + v[3]
    z_pred[4] = x[4] + v[4]
    z_pred[5] = x[5] + v[5]    

    return z_pred



def Jx_measurement_eq(x, u, t, v = None):
    Hx = np.zeros((len(v), len(x)))

    # d(phi_m)/d(phi)
    Hx[0, 0] = 1
    # d(theta_m)/d(theta)
    Hx[1, 1] = 1
    # d(psi_m)/d(psi)
    Hx[2, 2] = 1
    # d(u_m)/du
    Hx[3, 3] = 1
    # d(v_m)/dv
    Hx[4, 4] = 1
    # d(w_m)/dw
    Hx[5, 5] = 1

    return Hx

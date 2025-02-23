import numpy as np
import pandas as pd

from common import angleFuncs

def normalizeQuadData(filteredData, droneParams, rotorConfig, r_sign, rotorDir, N_rot = 4, minRPM = 1000):
    # Extract drone params
    R = droneParams['R']
    b = droneParams['b']
    rho = droneParams['rho']

    # Derive normalized rotor speed
    omega = filteredData[['w1', 'w2', 'w3', 'w4']].to_numpy()*2*np.pi/60
    omega_tot = np.sum(omega, axis=1)
    omega2 = filteredData[['w2_1', 'w2_2', 'w2_3', 'w2_4']].to_numpy()*(2*np.pi/60)**2
    # TODO: Scale w_avg by w_max like w_tot is being scaled. 
    w_avg = np.array(np.sqrt(np.sum(omega2, axis = 1)/N_rot)).reshape(-1, 1)
    # Replace 0 with np.nan
    w_avg = np.where(w_avg < minRPM*(2*np.pi/60), np.nan, w_avg)

    # Normalize rotor speed
    n_omega = np.divide(omega, w_avg)
    # n_omega_tot = np.divide(omega_tot.reshape(-1, 1), w_avg)
    # print('[ WARNING ] Prototyping, w_tot is now scaled by droneParams["max RPM"]')
    n_omega_tot = np.divide(omega_tot.reshape(-1, 1), w_avg) * (w_avg/(droneParams['max RPM']*2*np.pi/60))
    n_omega2 = np.divide(omega2, np.square(w_avg))
    n_omega_CMD = np.divide(filteredData[['w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD']].to_numpy()*2*np.pi/60, w_avg)

    # Normalize forces and moments
    va = np.sqrt(np.sum(np.square(filteredData[['u', 'v', 'w']].to_numpy()), axis = 1))

    # F_den = 2/(rho*(N_rot*np.pi*R**2)*(R*w_avg)**2)
    # print('[ INFO ] RUNNING WITH NEW NORMALIZATION SCHEME.')
    # F_den = 2/(rho*(N_rot*np.pi*R**2)*(R**2*w_avg*droneParams['max RPM']*2*np.pi/60))
    # F_den = 2/(rho*(N_rot*np.pi*R**2)*(R*(droneParams['max RPM']*(2*np.pi/60))**2/(droneParams['max RPM']*(2*np.pi/60)+w_avg))**2)
    F_den = (0.5*rho*(N_rot*np.pi*R**2)*(R*(droneParams['max RPM']*(2*np.pi/60))**2/(droneParams['max RPM']*(2*np.pi/60)+w_avg))**2)

    # import code
    # code.interact(local=locals())

    # F_den = 2/(rho*(N_rot*np.pi*R**2)*(va)**2).reshape(-1, 1)
    # F_den = F_den + np.nanmean(F_den)
    # F_den = np.where(F_den < zeroLim, np.nan, F_den)
    Cx = np.divide(filteredData['Fx'].to_numpy().reshape(-1, 1), F_den)
    Cy = np.divide(filteredData['Fy'].to_numpy().reshape(-1, 1), F_den)
    Cz = np.divide(filteredData['Fz'].to_numpy().reshape(-1, 1), F_den)

    # Normalize rates 
    n_p = np.divide(filteredData['p'].to_numpy().reshape(-1, 1)*b, w_avg*R)
    n_q = np.divide(filteredData['q'].to_numpy().reshape(-1, 1)*b, w_avg*R)
    n_r = np.divide(filteredData['r'].to_numpy().reshape(-1, 1)*b, w_avg*R)

    # Define control moments 
    u_p = (n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['aft left'] - 1]) - (n_omega[:, rotorConfig['front right'] - 1] + n_omega[:, rotorConfig['aft right'] - 1])
    u_q = (n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['front right'] - 1]) - (n_omega[:, rotorConfig['aft left'] - 1] + n_omega[:, rotorConfig['aft right'] - 1])
    u_r = r_sign[rotorDir]*((n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['aft right']- 1]) - (n_omega[:, rotorConfig['front right'] - 1] + n_omega[:, rotorConfig['aft left'] - 1]))

    U_p = (n_omega2[:, rotorConfig['front left'] - 1] + n_omega2[:, rotorConfig['aft left'] - 1]) - (n_omega2[:, rotorConfig['front right'] - 1] + n_omega2[:, rotorConfig['aft right'] - 1])
    U_q = (n_omega2[:, rotorConfig['front left'] - 1] + n_omega2[:, rotorConfig['front right'] - 1]) - (n_omega2[:, rotorConfig['aft left'] - 1] + n_omega2[:, rotorConfig['aft right'] - 1])
    U_r = r_sign[rotorDir]*((n_omega2[:, rotorConfig['front left'] - 1] + n_omega2[:, rotorConfig['aft right']- 1]) - (n_omega2[:, rotorConfig['front right'] - 1] + n_omega2[:, rotorConfig['aft left'] - 1]))


    # Normalize velocities
    va = np.sqrt(np.sum(np.square(filteredData[['u', 'v', 'w']].to_numpy()), axis = 1))
    slow_va_idx = np.where(va < 0.01)[0]

    u_bar = np.divide(filteredData['u'], va)
    u_bar[slow_va_idx] = 0
    v_bar = np.divide(filteredData['v'], va)
    v_bar[slow_va_idx] = 0
    w_bar = np.divide(filteredData['w'], va)
    w_bar[slow_va_idx] = 0

    # Normalize the induced velocity
    vi_bar = np.divide(filteredData['v_in'], va)
    vi_bar[slow_va_idx] = 0


    # Normalize velocities
    mux_bar = np.divide(filteredData['u'].to_numpy().reshape(-1, 1), w_avg*R)
    muy_bar = np.divide(filteredData['v'].to_numpy().reshape(-1, 1), w_avg*R)
    muz_bar = np.divide(filteredData['w'].to_numpy().reshape(-1, 1), w_avg*R)

    # Normalize the induced velocity
    mu_vi_bar = np.divide(filteredData['v_in'].to_numpy().reshape(-1, 1), w_avg*R)


    # Check if moments are present
    hasMoments = False
    if all(M in filteredData.columns for M in ('Mx', 'My', 'Mz')):
        hasMoments = True
        M_den = F_den * (1/b)
        CL = np.divide(filteredData['Mx'].to_numpy().reshape(-1, 1), M_den)
        CM = np.divide(filteredData['My'].to_numpy().reshape(-1, 1), M_den)
        CN = np.divide(filteredData['Mz'].to_numpy().reshape(-1, 1), M_den)
        

    # Replace NaNs
    F_den_NaNs = np.where(np.isnan(F_den))[0]
    F_den[F_den_NaNs] = 0

    # Define normalized DataFrame
    NormalizedData = pd.DataFrame()
    NormalizedData['t'] = filteredData['t']
    NormalizedData['w_tot'] = n_omega_tot.reshape(-1)
    NormalizedData['w_avg'] = w_avg.reshape(-1)
    NormalizedData[['w1', 'w2', 'w3', 'w4']] = pd.DataFrame(n_omega, columns=['w1', 'w2', 'w3', 'w4'])
    NormalizedData[['w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD']] = pd.DataFrame(n_omega_CMD, columns=['w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD'])
    NormalizedData[['w2_1', 'w2_2', 'w2_3', 'w2_4']] = pd.DataFrame(n_omega2, columns=['w2_1', 'w2_2', 'w2_3', 'w2_4'])
    NormalizedData[['Fx', 'Fy', 'Fz']] = pd.DataFrame(np.hstack((Cx, Cy, Cz)), columns=['Fx', 'Fy', 'Fz'])
    NormalizedData[['p', 'q', 'r']] = pd.DataFrame(np.hstack((n_p, n_q, n_r)), columns=['p', 'q', 'r'])
    NormalizedData[['u_p', 'u_q', 'u_r']] = pd.DataFrame(np.vstack((u_p, u_q, u_r)).T, columns=['u_p', 'u_q', 'u_r'])
    NormalizedData[['U_p', 'U_q', 'U_r']] = pd.DataFrame(np.vstack((U_p, U_q, U_r)).T, columns=['U_p', 'U_q', 'U_r'])
    NormalizedData[['roll', 'pitch', 'yaw']] = filteredData[['roll', 'pitch', 'yaw']]
    NormalizedData[['u', 'v', 'w']] = pd.DataFrame(np.vstack((u_bar, v_bar, w_bar)).T, columns=['u', 'v', 'w'])
    NormalizedData['v_in'] = vi_bar
    NormalizedData[['mu_x', 'mu_y', 'mu_z']] = pd.DataFrame(np.hstack((mux_bar, muy_bar, muz_bar)), columns=['mu_x', 'mu_y', 'mu_z'])
    NormalizedData['mu_vin'] = mu_vi_bar
    NormalizedData['F_den'] = F_den.reshape(-1)

    if hasMoments:
        M_den[F_den_NaNs] = 0
        NormalizedData['Mx'] = CL
        NormalizedData['My'] = CM
        NormalizedData['Mz'] = CN
        NormalizedData['M_den'] = M_den


    # Replace NaNs
    NormalizedData.fillna(0, inplace=True)

    return NormalizedData





def addExtraColsAndTrim(DataList, usableDataRatio):
    '''
    Add extra columns, useful for identification, to DataFrames
    '''
    for i, d in enumerate(DataList.copy()):
        # Add abs(V) to columns of d
        d['|u|'] = np.abs(d['u'])
        d['|v|'] = np.abs(d['v'])
        d['|w|'] = np.abs(d['w'])

        if 'mu_x' in d.columns:
            # Add abs(mu_V) to columns of d
            d['|mu_x|'] = np.abs(d['mu_x'])
            d['|mu_y|'] = np.abs(d['mu_y'])
            d['|mu_z|'] = np.abs(d['mu_z'])

        # Add abs(rates) to columns of d
        d['|p|'] = np.abs(d['p'])
        d['|q|'] = np.abs(d['q'])
        d['|r|'] = np.abs(d['r'])

        # Add abs(control_moments) to columns of d
        d['|u_p|'] = np.abs(d['u_p'])
        d['|u_q|'] = np.abs(d['u_q'])
        d['|u_r|'] = np.abs(d['u_r'])

        d['|U_p|'] = np.abs(d['U_p'])
        d['|U_q|'] = np.abs(d['U_q'])
        d['|U_r|'] = np.abs(d['U_r'])        

        # Add np.sin(attitude) to columns of d
        d['sin[roll]'] = np.sin(d['roll'])
        d['sin[pitch]'] = np.sin(d['pitch'])
        d['sin[yaw]'] = np.sin(d['yaw'])

        # Add np.sin(attitude) to columns of d
        d['cos[roll]'] = np.cos(d['roll'])
        d['cos[pitch]'] = np.cos(d['pitch'])
        d['cos[yaw]'] = np.cos(d['yaw'])

        # Bind angles
        d['roll'] = angleFuncs.wrapPi(d['roll'].copy())
        d['pitch'] = angleFuncs.wrapPi(d['pitch'].copy())
        d['yaw'] = angleFuncs.wrapPi(d['yaw'].copy())

        # idxEnd = int(usableDataRatio*len(d))
        # DataList[i] = d.iloc[:idxEnd+1, :]
        buffer = int((len(d) - int(usableDataRatio*len(d)))/2)
        DataList[i] = d.iloc[buffer:-buffer, :]

    return DataList



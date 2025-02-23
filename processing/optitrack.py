import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import datetime
import json

from common import angleFuncs, solvers
from processing import utility, filtering


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
    return x



def V1_import_OT_withWind(rowIdx, logFile, rigidBodyName, applyFreqFilter = True, filterOTSpikes = True, velocityCutoffHz = 40):
    filename = logFile.loc[rowIdx, 'Raw OptiTrack Name']
    filedir = logFile.loc[rowIdx, 'Raw OT Path']
    print('[ INFO ] Importing OptiTrack data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))

    configData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', nrows=1, header=None)
    # Extract save parameters
    saveParams = {}
    k, v = None, None
    for i in configData:
        # Even indices correspond to keys, odd to values
        if not i % 2:
            k = configData[i][0]
        else:
            v = configData[i][0]
            saveParams.update({k:v})

    fullRawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', 
                                        header=None, skiprows=3, low_memory=False)
    idxQuadrotor = np.where(fullRawData.loc[0, :] == rigidBodyName)[0]
    if len(idxQuadrotor) < 1:
        raise ValueError('Provided rigidBodyName, "{}", not found in {}'.format(rigidBodyName, os.path.join(filedir, '{}.csv'.format(filename))))
    else:
        print('[ INFO ] Found {} as rigidBodyName'.format(rigidBodyName))
        # Create new DataFrame with specified quadrotor data
        colNames = ['Frame', 'Time']
        for i in idxQuadrotor:
            colNames.append('{}_{}'.format(fullRawData.loc[2, i], fullRawData.loc[3, i]))
        rawBodyData = fullRawData.loc[4:, idxQuadrotor].to_numpy()
        frame = fullRawData.loc[4:, 0].to_numpy()
        time = fullRawData.loc[4:, 1].to_numpy()
        rawBodyData = np.hstack((frame.reshape(-1, 1), time.reshape(-1, 1), rawBodyData)).astype(float)

        bodyData = pd.DataFrame(data=rawBodyData, columns=colNames)

        idxStart_old = 0
        idxEnd_old = len(bodyData)
        # If there are NaNs in bodyData, attempt to interpolate
        if bodyData.isnull().values.any():
            # For each OptiTrack signal
            for i, col in enumerate(bodyData.columns):
                # Find NaN locations, if any
                locNaNs = bodyData[col].isnull()
                idxNaNs = np.where(locNaNs)[0]
                # If there are NaNs in signal
                if len(idxNaNs) > 0:
                    # If NaNs are at the start of end of signal, then we cannot interpolate on these regions
                    # Find where data starts and ends
                    idxsWithData = np.where(~locNaNs)[0]
                    if not len(idxsWithData):
                        raise ValueError('Inputted OptiTrack data is full of NaNs for column {}, cannot interpolate. Please verify that the OpitTrack data is correct.'.format(col))
                    idxStart = idxsWithData[0]
                    # idxEnd = idxsWithData[-11]
                    idxEnd = idxsWithData[-1]
                    # Replace current idxStart and idxEnd if necessary
                    if idxStart > idxStart_old:
                        idxStart_old = idxStart
                    if idxEnd < idxEnd_old:
                        idxEnd_old = idxEnd
                    usefulNaNIdxs = np.where((idxNaNs > idxStart) & (idxNaNs < idxEnd))[0]
                    if len(usefulNaNIdxs):
                        idxNaNs = idxNaNs[usefulNaNIdxs]
                        # Create a function which linearly interpolates the signal, based on known data, which takes
                        # index as input -> i.e. y = f(index) -< can be thought of as a proxy for time
                        func = interp1d(np.where(~locNaNs)[0], bodyData.to_numpy()[~locNaNs, i], kind = 'slinear')
                        # Interpolate NaN indexes 
                        bodyData.loc[idxNaNs, col] = func(idxNaNs)

        # Clip BodyData to regions where data is available
        bodyData = bodyData.iloc[idxStart_old:idxEnd_old, :]
        
        time = bodyData['Time'].to_numpy()
        dt = time[1] - time[0]
        fs = float(configData.iloc[0, 7])


        # Add wind velocity, if present. 
        hasOJF = logFile.loc[rowIdx, 'Has OJF data']

        OT_startIdx = 0
        OT_endIdx = len(time)
        velWind_OJF = np.zeros(time.shape)

        if hasOJF.lower() == 'y':
            # Check if there is a wind file, or static wind should be used instead
            windFilePath = logFile.loc[rowIdx, 'OJF path']
            windFilename = logFile.loc[rowIdx, 'OJF name']
            if not str(windFilename) == 'nan':
                windHeader = pd.read_csv(os.path.join(windFilePath, windFilename + '.lvm'), nrows = 1, skiprows = 10, delimiter = ',', header = None)
                windData = pd.read_csv(os.path.join(windFilePath, windFilename + '.lvm'), header = None, skiprows=22, delimiter = ',', names = ['t', 'V_air'])
                # Resample windData to match OT data sampling
                windData_resampled = utility.resample(windData, fs, interpolation_kind='linear')

                # Get timestamp info 
                timeStampOT = configData.iloc[0, 11].split(' ')[1]
                # OptiTrack info does not attach AM/PM to times, so need to infer this from creation of file. Note, on Linux, we cant get creation time, only modification
                timeStampOT_MT = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(filedir, '{}.csv'.format(filename))))
                if timeStampOT_MT.hour > 12:
                    timeStampOT[:2] = str(timeStampOT_MT)
                timeOT_start_seconds = utility.timeOfDay2Seconds(timeStampOT) - float(logFile.loc[rowIdx, 'OptiTrack time offset (s)']) # Assume OptiTrack is ahead of OJF (so need to subtract delay)
                timeOJ_start_seconds = utility.timeOfDay2Seconds(str(windHeader.iloc[0, 1]))
                # Add starting times (with datum at 00:00:00 of the day) to respective time arrays 
                timeOT = time + timeOT_start_seconds
                timeOJ = windData_resampled['t'].to_numpy() + timeOJ_start_seconds

                timePrecision = 6
                minT = np.nanmax([timeOT[0], timeOJ[0]])
                maxT = np.nanmin([timeOT[-1], timeOJ[-1]])

                # Find intersection of time, in terms of indices
                OJ_startIdx = np.where(np.around(timeOJ, decimals=timePrecision) >= minT)[0][0]
                OJ_endIdx = np.where(np.around(timeOJ, decimals=timePrecision) <= maxT)[0][-1]

                # Sometimes, the resampling causes some floating point errors (e.g. x.xxxE-16 instead of 0)
                # This therefore offsets the startIdx/endIdx by one.
                OT_startIdx = np.where(np.around(timeOT, decimals=timePrecision) >= minT)[0][0]
                OT_endIdx = np.where(np.around(timeOT, decimals=timePrecision) <= maxT)[0][-1]

                velWind_OJF = windData_resampled['V_air'].to_numpy()[OJ_startIdx:OJ_endIdx]

            else:
                # Take static wind
                staticWind = float(logFile.loc[rowIdx, 'Static wind (m/s)'])
                # In OJF wind is along OptiTrack +z direction, so in E-frame this is along -x 
                velWind_OJF = velWind_OJF + staticWind
                #TODO: For static wind, maybe ramp up wind speed along with intial pitch changes
        else:
            print('[ WARNING ] File \n\t\t{}\n\t has no OJF data. Adjust log files if this is incorrect or ignore this message if correct.\n[ INFO ] Processing can still continue.'.format(filename))

        # Trim bodyData to region for which there is wind data, if used
        bodyData = bodyData.iloc[OT_startIdx:OT_endIdx, :]
        time = time[OT_startIdx:OT_endIdx] - time[OT_startIdx]


        # Remove 'spikes' (i.e. artifacts from loss of tracking) in position from OptiTrack data before processing. 
        # Spikes in rotation are removed later after conversion to euler angles
        sigma = 6
        interpolation_window = 10
        if filterOTSpikes:
            spikeCols = ['Position_X', 'Position_Y', 'Position_Z']
            for col in spikeCols:
                bodyData[col] = filterSpikes(bodyData[col].to_numpy().copy(), time, sigma_dx = sigma, window = interpolation_window).reshape(-1)

        # Quaternions in OptiTrack Frame. OptiTrack z, x, y -> x, y, z
        qx = bodyData['Rotation_X'].to_numpy()
        qy = bodyData['Rotation_Y'].to_numpy()
        qz = bodyData['Rotation_Z'].to_numpy()
        qw = bodyData['Rotation_W'].to_numpy()

        # G = OptiTrack Ground frame
        x_G = bodyData['Position_X'].to_numpy()
        y_G = bodyData['Position_Y'].to_numpy()
        z_G = bodyData['Position_Z'].to_numpy()
        posCO_G = np.vstack((x_G, y_G, z_G)).T

        # Open drone-config file 
        configFile = '{}.json'.format(logFile.loc[rowIdx, 'Drone Config File'])
        configPath = logFile.loc[rowIdx, 'Drone Config Path']
        path2File = os.path.join(configPath, configFile)

        with open(path2File, 'r') as f:
            droneConfigData = json.load(f)
            f.close()
        
        # ####################################################################
        # Rotate optitrack axis system to match that of Sihao
        rots = {'x':angleFuncs.EulRotX, 'y':angleFuncs.EulRotY, 'z':angleFuncs.EulRotZ}
        axisGCorrection = droneConfigData['optitrack ground axes correction']
        axisRotOrder = axisGCorrection['order'].lower()
        Rot1 = rots[axisRotOrder[0]](float(axisGCorrection[axisRotOrder[0]])*np.pi/180)
        Rot2 = rots[axisRotOrder[1]](float(axisGCorrection[axisRotOrder[1]])*np.pi/180)
        Rot3 = rots[axisRotOrder[2]](float(axisGCorrection[axisRotOrder[2]])*np.pi/180)
        R_axisCorrectionRotation = np.array(np.matmul(Rot1, np.matmul(Rot2, Rot3)))       

        posG = np.vstack((x_G, y_G, z_G)).T
        posC = np.matmul(np.vstack((R_axisCorrectionRotation,)*len(time)).reshape(-1, 3, 3), posG.reshape(-1, 3, 1)).reshape(-1, 3)

        x_G, y_G, z_G = posC[:, 0], posC[:, 1], posC[:, 2]
        posCO_G = np.vstack((x_G, y_G, z_G)).T

        eulGCorrection = np.array((float(axisGCorrection['x'])*np.pi/180, float(axisGCorrection['y'])*np.pi/180, float(axisGCorrection['z'])*np.pi/180))
        quatGCorrection = angleFuncs.Eul2Quat(eulGCorrection)
        quatG = np.vstack((qw, qx, qy, qz)).T
        quatC = angleFuncs.quatMul(np.vstack((quatGCorrection, )*len(quatG)), quatG)
        qw, qx, qy, qz = quatC[:, 0], quatC[:, 1], quatC[:, 2], quatC[:, 3]

        # Define rotation matrices for each axis 
        R_1 = np.array([(1 - 2*(qy*qy + qz*qz)), (2*(qx*qy - qw*qz)), (2*(qw*qy + qx*qz))])
        R_2 = np.array([(2*(qx*qy + qw*qz)), (1 - 2*(qx*qx + qz*qz)), (2*(qy*qz - qw*qx))])
        R_3 = np.array([2*(qx*qz - qw*qy), (2*(qw*qx + qy*qz)), (1 - 2*(qx*qx + qy*qy))])

        # Manipulate the indices of the rotation matrices above to get a vector of form
        # N x [3 x 3] such that each element corresponds to the rotation matrix for that
        # specific sample
        R_1 = R_1.T
        R_2 = R_2.T
        R_3 = R_3.T
        R_stack = np.zeros((3*len(R_1), 3))
        R_stack[0:(3*len(R_1)):3] = R_1
        R_stack[1:(3*len(R_1)):3] = R_2
        R_stack[2:(3*len(R_1)):3] = R_3
        R = R_stack.reshape((len(R_1), 3, 3))

        # G = OptiTrack ground, O = OptiTrack body
        # Default signs for angles in Sihao's frame
        # axisOCorrection = logFile.loc[rowIdx, 'Initial Yaw Offset']*np.pi/180 - 90*np.pi/180
        axisOCorrection = 0
        # Axis map for roll pitch yaw
        defaultMap = np.array([0, 1, 2])
        # defaultSigns = np.array([1, -1, -1])
        defaultSigns = np.array([1, -1, 1])
        axisMap = np.array(np.matmul(angleFuncs.EulRotY(axisOCorrection), defaultMap))
        signs = np.array(np.matmul(angleFuncs.EulRotY(axisOCorrection), defaultSigns))
        # signs = defaultSigns.reshape(1, -1)
        # Roll around OptiTrack Z 
        _roll_G2O = np.arctan2(R[:, 1, 0], R[:, 1, 1])
        # Pitch around OptiTrack X
        _pitch_G2O = np.arctan2(-R[:, 1, 2], np.real(np.sqrt(1 - R[:, 1, 2]**2)))
        # Yaw around OptiTrack Y
        _yaw_G2O = np.arctan2(R[:, 0, 2], R[:, 2, 2])

        anglesG2O = [_roll_G2O, _pitch_G2O, _yaw_G2O]

        roll_G2O = anglesG2O[int(abs(axisMap[0][0]))]*np.around(signs[0][0], 10)
        pitch_G2O = anglesG2O[int(abs(axisMap[0][1]))]*np.around(signs[0][1], 10)
        yaw_G2O = anglesG2O[int(abs(axisMap[0][2]))]*np.around(signs[0][2], 10)

        # unwrap yaw_G2O from 0, 2pi
        yaw_G2O = angleFuncs.unwrapPi(yaw_G2O)

        yaw_E2B_bias = yaw_G2O 
        roll_E2B_bias = roll_G2O
        pitch_E2B_bias = pitch_G2O

        # Get rotation matrix from body to earth frame to extract euler angles
        R_B2E = np.matmul(angleFuncs.EulRotZ_arr(yaw_E2B_bias), np.matmul(angleFuncs.EulRotY_arr(pitch_E2B_bias), angleFuncs.EulRotX_arr(roll_E2B_bias)))

        roll_E2B = np.arctan2(R_B2E[:, 2, 1], R_B2E[:, 2, 2])
        pitch_E2B = np.arctan2(-R_B2E[:, 2, 0], np.real(np.sqrt(1 - np.square(R_B2E[:, 2, 0]))))
        yaw_E2B = np.arctan2(R_B2E[:, 1, 0], R_B2E[:, 0, 0])

        # Unwrap yaw_E2B from 0, 2pi
        yaw_E2B = angleFuncs.unwrapPi(yaw_E2B)

        # Apply filter to attitude to remove large peaks in its derivatives
        if applyFreqFilter:
            roll_E2B = filtering._ButterFilter(roll_E2B, fs, 4, 10, 'low')
            pitch_E2B = filtering._ButterFilter(pitch_E2B, fs, 4, 10, 'low')
            yaw_E2B = filtering._ButterFilter(yaw_E2B, fs, 4, 10, 'low')


        axisCorrection = droneConfigData['axis direction correction']

        rots = {'x':angleFuncs.EulRotX, 'y':angleFuncs.EulRotY, 'z':angleFuncs.EulRotZ}
        axisRotOrder = axisCorrection['order'].lower()
        Rot1 = rots[axisRotOrder[0]](float(axisCorrection[axisRotOrder[0]])*np.pi/180)
        Rot2 = rots[axisRotOrder[1]](float(axisCorrection[axisRotOrder[1]])*np.pi/180)
        Rot3 = rots[axisRotOrder[2]](float(axisCorrection[axisRotOrder[2]])*np.pi/180)
        R_G2E = np.array(np.matmul(Rot1, np.matmul(Rot2, Rot3)))

        # Derive the position, velocity and acceleration information of the c.g. and IMU

        # Transform position from OptiTrack ground frame to E-Frame
        # Convert matrix to array so that we can apply element-wise multiplication
        R_G2E = np.array(R_G2E)
        # posCO_E = np.matmul(posCO_G.reshape(-1, 1, 3), np.vstack((R_G2E.T,)*posCO_G.shape[0]).reshape(-1, 3, 3)).reshape(-1, 3)
        posCO_E = np.matmul(np.vstack((R_G2E,)*posCO_G.shape[0]).reshape(-1, 3, 3), posCO_G.reshape(-1, 3, 1)).reshape(-1, 3)
        # posCO_E = np.matmul(posCO_G.reshape(-1, 1, 3), np.vstack((R_G2E.T,)*posCO_G.shape[0]).reshape(-1, 3, 3)).reshape(-1, 3)*np.sin(logFile.loc[rowIdx, 'Initial Yaw Offset']*np.pi/180)
        # Position c.g. w.r.t drone body frame, assume it is more-or-less at origin
        posCG_B = np.array([0, 0, 0])
        # Assume IMU is at c.g. position
        posIMU_B = np.array([0, 0, 0])
        # Position of cg of OptiTrack Markers on drone 
        posCO_B_dict = droneConfigData['optitrack marker cg offset']
        for k, v in posCO_B_dict.items():
            # Convert string input to floats
            posCO_B_dict.update({k:float(v)})


        # # Need to rotate by np.pi to get coordinates in body frame
        # posCO_B = np.array(np.array([posCO_B_dict['x'], posCO_B_dict['y'], posCO_B_dict['z']])*angleFuncs.EulRotX(-1*np.pi))
        posCO_B = np.array([posCO_B_dict['x'], posCO_B_dict['y'], posCO_B_dict['z']])

        # Derive position in E-frame from B-frame
        Rot_B2E = np.matmul(angleFuncs.EulRotZ_arr(yaw_E2B), np.matmul(angleFuncs.EulRotY_arr(pitch_E2B), angleFuncs.EulRotX_arr(roll_E2B)))
        posCG_E = posCO_E + np.matmul(Rot_B2E, np.vstack(((posCG_B - posCO_B),)*len(Rot_B2E)).reshape(-1, 3, 1)).reshape(-1, 3)
        posIMU_E = posCO_E + np.matmul(Rot_B2E, np.vstack(((posIMU_B - posCO_B),)*len(Rot_B2E)).reshape(-1, 3, 1)).reshape(-1, 3)


        # Derive velocity from position data, in E-frame
        velCO_E = solvers.derivative(posCO_E, time)
        velCG_E = solvers.derivative(posCG_E, time)
        velIMU_E = solvers.derivative(posIMU_E, time)


        # Add wind data
        velWind_E = np.vstack((velWind_OJF, velWind_OJF*0, velWind_OJF*0)).T
        # velCO_E[:, 0] = velCO_E[:, 0] - velWind_OJF
        # velCG_E[:, 0] = velCG_E[:, 0] - velWind_OJF
        # velIMU_E[:, 0] = velIMU_E[:, 0] - velWind_OJF
        velCO_E[:, 0] = velCO_E[:, 0] + velWind_OJF
        velCG_E[:, 0] = velCG_E[:, 0] + velWind_OJF
        velIMU_E[:, 0] = velIMU_E[:, 0] + velWind_OJF



        # Apply filter to velocities to remove large spikes of OptiTrack (e.g. due to loss of tracking)
        # Filtering not applied direct to position since outliers are harder to detect (i.e. box-like signal forms
        # instead of large, instantenous, peaks)
        if applyFreqFilter:
            # plt.psd(velCO_E[:, 0]) # Cut-off at 4 Hz
            for i in range(velCO_E.shape[1]):
                velCO_E[:, i] = filtering._ButterFilter(velCO_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velCG_E[:, i] = filtering._ButterFilter(velCG_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velIMU_E[:, i] = filtering._ButterFilter(velIMU_E[:, i], fs, 4, velocityCutoffHz, 'low')


        # Derive accelerations from velocity data, in E-frame
        accCO_E = solvers.derivative(velCO_E, time)
        accCG_E = solvers.derivative(velCG_E, time)
        accIMU_E = solvers.derivative(velIMU_E, time)

        # Convert positions and velocities in E-frame to B-frame
        Rot_E2B = np.matmul(angleFuncs.EulRotX_arr(-1*roll_E2B), np.matmul(angleFuncs.EulRotY_arr(-1*pitch_E2B), angleFuncs.EulRotZ_arr(-1*yaw_E2B)))

        velCO_B = np.matmul(Rot_E2B, velCO_E.reshape(-1, 3, 1)).reshape(-1, 3)
        velCG_B = np.matmul(Rot_E2B, velCG_E.reshape(-1, 3, 1)).reshape(-1, 3)
        velIMU_B = np.matmul(Rot_E2B, velIMU_E.reshape(-1, 3, 1)).reshape(-1, 3)
        velWind_B = np.matmul(Rot_E2B, velWind_E.reshape(-1, 3, 1)).reshape(-1, 3)

        accCO_B = np.matmul(Rot_E2B, accCO_E.reshape(-1, 3, 1)).reshape(-1, 3)
        accCG_B = np.matmul(Rot_E2B, accCG_E.reshape(-1, 3, 1)).reshape(-1, 3)
        accIMU_B = np.matmul(Rot_E2B, accIMU_E.reshape(-1, 3, 1)).reshape(-1, 3)


        # Derive angular rates and accelerations
        dRoll_E2B = solvers.derivative(roll_E2B.reshape(-1, 1), time)
        dPitch_E2B = solvers.derivative(pitch_E2B.reshape(-1, 1), time)
        dYaw_E2B = solvers.derivative(yaw_E2B.reshape(-1, 1), time)

        _AAZeros = np.zeros(roll_E2B.shape)
        _AAOnes = np.ones(roll_E2B.shape)
        AA = np.array([[_AAOnes, _AAZeros, -1*np.sin(pitch_E2B)],
                        [_AAZeros, np.cos(roll_E2B), np.sin(roll_E2B)*np.cos(pitch_E2B)],
                        [_AAZeros, -1*np.sin(roll_E2B), np.cos(roll_E2B)*np.cos(pitch_E2B)]]).reshape(-1, 3, 3)

        omega_B_OT = np.matmul(AA, np.hstack((dRoll_E2B, dPitch_E2B, dYaw_E2B)).reshape(-1, 3, 1)).reshape(-1, 3)

        alpha_B_OT = solvers.derivative(omega_B_OT, time)


        # Write data to new DataFrame
        flips = {False:1, True:-1}

        OT_Data = pd.DataFrame()
        OT_Data['t'] = time

        OT_Data['x'] = posCG_E[:, 0]
        OT_Data['y'] = posCG_E[:, 1]
        OT_Data['z'] = posCG_E[:, 2]
        OT_Data['x_g'] = posCO_G[:, 0]
        OT_Data['y_g'] = posCO_G[:, 1]
        OT_Data['z_g'] = posCO_G[:, 2]

        # OT_Data['posCO_E'] = posCO_E

        OT_Data['roll'] = roll_E2B * flips[droneConfigData['flip attitude sign']['roll']] 
        OT_Data['pitch'] = pitch_E2B * flips[droneConfigData['flip attitude sign']['pitch']] 
        OT_Data['yaw'] = yaw_E2B * flips[droneConfigData['flip attitude sign']['yaw']] 

        # OT_Data['velCO_E'] = velCO_E

        OT_Data['u_E'] = velCG_E[:, 0] 
        OT_Data['v_E'] = velCG_E[:, 1]
        OT_Data['w_E'] = velCG_E[:, 2]
        OT_Data['u'] = velCG_B[:, 0] * flips[droneConfigData['flip accelerometer sign']['ax']] 
        OT_Data['v'] = velCG_B[:, 1] * flips[droneConfigData['flip accelerometer sign']['ay']] 
        OT_Data['w'] = velCG_B[:, 2] * flips[droneConfigData['flip accelerometer sign']['az']]
        OT_Data['u_air'] = velWind_B[:, 0] * flips[droneConfigData['flip accelerometer sign']['ax']]
        OT_Data['v_air'] = velWind_B[:, 1] * flips[droneConfigData['flip accelerometer sign']['ay']]
        OT_Data['w_air'] = velWind_B[:, 2] * flips[droneConfigData['flip accelerometer sign']['az']]

        # OT_Data['accCO_B'] = accCO_B

        OT_Data['ax_E'] = accCG_E[:, 0]
        OT_Data['ay_E'] = accCG_E[:, 1]
        OT_Data['az_E'] = accCG_E[:, 2]
        OT_Data['ax'] = accCG_B[:, 0] * flips[droneConfigData['flip accelerometer sign']['ax']] 
        OT_Data['ay'] = accCG_B[:, 1] * flips[droneConfigData['flip accelerometer sign']['ay']] 
        OT_Data['az'] = accCG_B[:, 2] * flips[droneConfigData['flip accelerometer sign']['az']] 

        OT_Data['p'] = omega_B_OT[:, 0] * flips[droneConfigData['flip attitude sign']['roll']] 
        OT_Data['q'] = omega_B_OT[:, 1] * flips[droneConfigData['flip attitude sign']['pitch']] 
        OT_Data['r'] = omega_B_OT[:, 2] * flips[droneConfigData['flip attitude sign']['yaw']] 

        OT_Data['dp'] = alpha_B_OT[:, 0] * flips[droneConfigData['flip attitude sign']['roll']] 
        OT_Data['dq'] = alpha_B_OT[:, 1] * flips[droneConfigData['flip attitude sign']['pitch']] 
        OT_Data['dr'] = alpha_B_OT[:, 2] * flips[droneConfigData['flip attitude sign']['yaw']] 

    return OT_Data



def V2_import_OT_withWind(rowIdx, logFile, rigidBodyName, applyFreqFilter = True, filterOTSpikes = True, velocityCutoffHz = 7):
    filename = logFile.loc[rowIdx, 'Raw OptiTrack Name']
    filedir = logFile.loc[rowIdx, 'Raw OT Path']
    print('[ INFO ] Importing OptiTrack data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))

    configData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', nrows=1, header=None)
    # Extract save parameters
    saveParams = {}
    k, v = None, None
    for i in configData:
        # Even indices correspond to keys, odd to values
        if not i % 2:
            k = configData[i][0]
        else:
            v = configData[i][0]
            saveParams.update({k:v})

    fullRawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', 
                                        header=None, skiprows=3, low_memory=False)
    idxQuadrotor = np.where(fullRawData.loc[0, :] == rigidBodyName)[0]
    if len(idxQuadrotor) < 1:
        raise ValueError('Provided rigidBodyName, "{}", not found in {}'.format(rigidBodyName, os.path.join(filedir, '{}.csv'.format(filename))))
    else:
        print('[ INFO ] Found {} as rigidBodyName'.format(rigidBodyName))
        # Create new DataFrame with specified quadrotor data
        colNames = ['Frame', 'Time']
        for i in idxQuadrotor:
            colNames.append('{}_{}'.format(fullRawData.loc[2, i], fullRawData.loc[3, i]))
        rawBodyData = fullRawData.loc[4:, idxQuadrotor].to_numpy()
        frame = fullRawData.loc[4:, 0].to_numpy()
        time = fullRawData.loc[4:, 1].to_numpy()
        rawBodyData = np.hstack((frame.reshape(-1, 1), time.reshape(-1, 1), rawBodyData)).astype(float)

        bodyData = pd.DataFrame(data=rawBodyData, columns=colNames)

        idxStart_old = 0
        idxEnd_old = len(bodyData)
        # If there are NaNs in bodyData, attempt to interpolate
        if bodyData.isnull().values.any():
            # For each OptiTrack signal
            for i, col in enumerate(bodyData.columns):
                # Find NaN locations, if any
                locNaNs = bodyData[col].isnull()
                idxNaNs = np.where(locNaNs)[0]
                # If there are NaNs in signal
                if len(idxNaNs) > 0:
                    # If NaNs are at the start of end of signal, then we cannot interpolate on these regions
                    # Find where data starts and ends
                    idxsWithData = np.where(~locNaNs)[0]
                    if not len(idxsWithData):
                        raise ValueError('Inputted OptiTrack data is full of NaNs for column {}, cannot interpolate. Please verify that the OpitTrack data is correct.'.format(col))
                    idxStart = idxsWithData[0]
                    # idxEnd = idxsWithData[-11]
                    idxEnd = idxsWithData[-1]
                    # Replace current idxStart and idxEnd if necessary
                    if idxStart > idxStart_old:
                        idxStart_old = idxStart
                    if idxEnd < idxEnd_old:
                        idxEnd_old = idxEnd
                    usefulNaNIdxs = np.where((idxNaNs > idxStart) & (idxNaNs < idxEnd))[0]
                    if len(usefulNaNIdxs):
                        idxNaNs = idxNaNs[usefulNaNIdxs]
                        # Create a function which linearly interpolates the signal, based on known data, which takes
                        # index as input -> i.e. y = f(index) -< can be thought of as a proxy for time
                        func = interp1d(np.where(~locNaNs)[0], bodyData.to_numpy()[~locNaNs, i], kind = 'slinear')
                        # Interpolate NaN indexes 
                        bodyData.loc[idxNaNs, col] = func(idxNaNs)

        # Clip BodyData to regions where data is available
        bodyData = bodyData.iloc[idxStart_old:idxEnd_old, :]
        
        time = bodyData['Time'].to_numpy()
        dt = time[1] - time[0]
        fs = float(configData.iloc[0, 7])


        # Add wind velocity, if present. 
        hasOJF = logFile.loc[rowIdx, 'Has OJF data']

        OT_startIdx = 0
        OT_endIdx = len(time)
        velWind_OJF = np.zeros(time.shape)

        if hasOJF.lower() == 'y':
            # Check if there is a wind file, or static wind should be used instead
            windFilePath = logFile.loc[rowIdx, 'OJF path']
            windFilename = logFile.loc[rowIdx, 'OJF name']
            if not str(windFilename) == 'nan':
                windHeader = pd.read_csv(os.path.join(windFilePath, windFilename + '.lvm'), nrows = 1, skiprows = 10, delimiter = ',', header = None)
                windData = pd.read_csv(os.path.join(windFilePath, windFilename + '.lvm'), header = None, skiprows=22, delimiter = ',', names = ['t', 'V_air'])
                # Resample windData to match OT data sampling
                windData_resampled = utility.resample(windData, fs, interpolation_kind='linear')

                # Get timestamp info 
                timeStampOT = configData.iloc[0, 11].split(' ')[1]
                # OptiTrack info does not attach AM/PM to times, so need to infer this from creation of file. Note, on Linux, we cant get creation time, only modification
                timeStampOT_MT = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(filedir, '{}.csv'.format(filename))))
                if timeStampOT_MT.hour > 12:
                    TOT_MM, TOT_SS = timeStampOT[3:5], timeStampOT[6:]
                    TOT_HH = str(timeStampOT_MT.hour)
                    timeStampOT = ':'.join([TOT_HH, TOT_MM, TOT_SS])
                timeOT_start_seconds = utility.timeOfDay2Seconds(timeStampOT) - float(logFile.loc[rowIdx, 'OptiTrack time offset (s)']) # Assume OptiTrack is ahead of OJF (so need to subtract delay)
                timeOJ_start_seconds = utility.timeOfDay2Seconds(str(windHeader.iloc[0, 1]))
                # Add starting times (with datum at 00:00:00 of the day) to respective time arrays 
                timeOT = time + timeOT_start_seconds
                timeOJ = windData_resampled['t'].to_numpy() + timeOJ_start_seconds

                timePrecision = 6
                minT = np.nanmax([timeOT[0], timeOJ[0]])
                maxT = np.nanmin([timeOT[-1], timeOJ[-1]])

                # Find intersection of time, in terms of indices
                OJ_startIdx = np.where(np.around(timeOJ, decimals=timePrecision) >= minT)[0][0]
                OJ_endIdx = np.where(np.around(timeOJ, decimals=timePrecision) <= maxT)[0][-1]

                # Sometimes, the resampling causes some floating point errors (e.g. x.xxxE-16 instead of 0)
                # This therefore offsets the startIdx/endIdx by one.
                OT_startIdx = np.where(np.around(timeOT, decimals=timePrecision) >= minT)[0][0]
                OT_endIdx = np.where(np.around(timeOT, decimals=timePrecision) <= maxT)[0][-1]

                # velWind_OJF = windData_resampled['V_air'].to_numpy()[OJ_startIdx:OJ_endIdx]
                velWind_OJF = windData_resampled['V_air'].to_numpy()[OJ_startIdx:(OJ_startIdx + (OT_endIdx - OT_startIdx))]

            else:
                # Take static wind
                staticWind = float(logFile.loc[rowIdx, 'Static wind (m/s)'])
                # In OJF wind is along OptiTrack +z direction, so in E-frame this is along -x 
                velWind_OJF = velWind_OJF + staticWind
                #TODO: For static wind, maybe ramp up wind speed along with intial pitch changes
        else:
            print('[ WARNING ] File \n\t\t{}\n\t has no OJF data. Adjust log files if this is incorrect or ignore this message if correct.\n[ INFO ] Processing can still continue.'.format(filename))

        # Trim bodyData to region for which there is wind data, if used
        bodyData = bodyData.iloc[OT_startIdx:OT_endIdx, :]
        time = time[OT_startIdx:OT_endIdx] - time[OT_startIdx]


        # Remove 'spikes' (i.e. artifacts from loss of tracking) in position from OptiTrack data before processing. 
        # Spikes in rotation are removed later after conversion to euler angles
        sigma = 6
        interpolation_window = 10
        if filterOTSpikes:
            spikeCols = ['Position_X', 'Position_Y', 'Position_Z']
            for col in spikeCols:
                bodyData[col] = filterSpikes(bodyData[col].to_numpy().copy(), time, sigma_dx = sigma, window = interpolation_window).reshape(-1)


        # Quaternions in OptiTrack Frame. OptiTrack z, x, y -> x, y, z
        qx = bodyData['Rotation_X'].to_numpy()
        qy = bodyData['Rotation_Y'].to_numpy()
        qz = bodyData['Rotation_Z'].to_numpy()
        qw = bodyData['Rotation_W'].to_numpy()
        quatOT = np.vstack((qw, qx, qy, qz)).T

        # G = OptiTrack Ground frame
        x_G = bodyData['Position_X'].to_numpy()
        y_G = bodyData['Position_Y'].to_numpy()
        z_G = bodyData['Position_Z'].to_numpy()
        posCO_G = np.vstack((x_G, y_G, z_G)).T

        # Open drone-config file 
        configFile = '{}.json'.format(logFile.loc[rowIdx, 'Drone Config File'])
        configPath = logFile.loc[rowIdx, 'Drone Config Path']
        path2File = os.path.join(configPath, configFile)

        with open(path2File, 'r') as f:
            droneConfigData = json.load(f)
            f.close()
        
        # ===============================================================================================
        # Coordinate system transformation
        # ===============================================================================================
        # Converion of optitrack ground system (G) to earth system (E)
        # OptiTrack axes system (G) - from perspective of computer - x forward, y up, z right
        # y   x
        # |  /
        # | /
        # |/_____z
        #
        # Earth axes system (E) - from perspective of computer - x right, y backwards (towards computer), z down
        #      |----x
        #     /|
        #    / |
        #   /  |
        #  y   z
        #
        # Define individual rotations for G to E in quaternions
        G2E_quat_x = angleFuncs.Eul2Quat(np.array([90*np.pi/180, 0, 0]))
        G2E_quat_y = angleFuncs.Eul2Quat(np.array([0, 0, 0]))
        G2E_quat_z = angleFuncs.Eul2Quat(np.array([0, 0, 90*np.pi/180]))
        # Apply rotations, in quaternions
        G2E_quat = angleFuncs.QuatConj(angleFuncs.quatMul(G2E_quat_x, angleFuncs.quatMul(G2E_quat_y, G2E_quat_z)))

        # Get position in E frame (from G frame)
        posCO_E = angleFuncs.QuatRot(G2E_quat.reshape(1, -1), posCO_G)

        # quatOT describes angles transforming G frame into Optitrack Body (O) frame. 
        # Here we first go from the E-frame to the G-frame, then we apply quatOT to transform into the O-frame
        #   Thus, we have a rotation from E-frame to O-frame
        quatE2O = angleFuncs.quatMul(quatOT, angleFuncs.QuatConj(G2E_quat.reshape(-1, 4)))
        # Finally, we can apply a correction from O-frame to B-frame (quadrotor body, which is coincident with E-frame @ origin)
        #   This gives us our rotation from E-Frame to B-frame (i.e. quadrotor attitude w.r.t E-frame)
        quatE2B = angleFuncs.QuatConj(angleFuncs.quatMul(G2E_quat.reshape(-1, 4), quatE2O))
        # Take the conjugate for angles describing transformation form B-frame to E-frame
        quatB2E = angleFuncs.QuatConj(quatE2B)
        # Extract euler equivalents
        eulB2E = angleFuncs.Quat2Eul(quatB2E) 

        # ===============================================================================================
        # Velocity derivation
        # ===============================================================================================
        posCG_B = np.array([0, 0, 0])
        # Assume IMU is at c.g. position
        posIMU_B = np.array([0, 0, 0])
        # Position of cg of OptiTrack Markers on drone 
        posCO_B_dict = droneConfigData['optitrack marker cg offset']
        for k, v in posCO_B_dict.items():
            # Convert string input to floats
            posCO_B_dict.update({k:float(v)})

        posCO_B = np.array([posCO_B_dict['x'], posCO_B_dict['y'], posCO_B_dict['z']])
        # import code
        # code.interact(local=locals())
        posCG_E = posCO_E + angleFuncs.QuatRot(quatB2E, np.vstack((posCG_B - posCO_B,)*len(quatE2B)))
        posIMU_E = posCO_E + angleFuncs.QuatRot(quatB2E, np.vstack((posIMU_B - posCO_B,)*len(quatE2B)))

        # Derive velocity from position data, in E-frame
        velCO_E = solvers.derivative(posCO_E, time)
        velCG_E = solvers.derivative(posCG_E, time)
        velIMU_E = solvers.derivative(posIMU_E, time)

        # Add wind data
        velWind_E = np.vstack((velWind_OJF, velWind_OJF*0, velWind_OJF*0)).T
        # velCO_E[:, 0] = velCO_E[:, 0] + velWind_OJF
        # velCG_E[:, 0] = velCG_E[:, 0] + velWind_OJF
        # velIMU_E[:, 0] = velIMU_E[:, 0] + velWind_OJF
        velCO_E[:, 0] = velCO_E[:, 0] - velWind_OJF
        velCG_E[:, 0] = velCG_E[:, 0] - velWind_OJF
        velIMU_E[:, 0] = velIMU_E[:, 0] - velWind_OJF

        # Apply filter to velocities to remove large spikes of OptiTrack (e.g. due to loss of tracking)
        # Filtering not applied direct to position since outliers are harder to detect (i.e. box-like signal forms
        # instead of large, instantenous, peaks)
        if applyFreqFilter:
            # plt.psd(velCO_E[:, 0]) # Cut-off at 4 Hz
            for i in range(velCO_E.shape[1]):
                velCO_E[:, i] = filtering._ButterFilter(velCO_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velCG_E[:, i] = filtering._ButterFilter(velCG_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velIMU_E[:, i] = filtering._ButterFilter(velIMU_E[:, i], fs, 4, velocityCutoffHz, 'low')

        # Derive accelerations from velocity data, in E-frame
        accCO_E = solvers.derivative(velCO_E, time)
        accCG_E = solvers.derivative(velCG_E, time)
        accIMU_E = solvers.derivative(velIMU_E, time)

        # Convert positions and velocities in E-frame to B-frame
        velCO_B = angleFuncs.QuatRot(quatE2B, velCO_E)
        velCG_B = angleFuncs.QuatRot(quatE2B, velCG_E)
        velIMU_B = angleFuncs.QuatRot(quatE2B, velIMU_E)
        velWind_B = angleFuncs.QuatRot(quatE2B, velWind_E)

        accCO_B = angleFuncs.QuatRot(quatE2B, accCO_E)
        accCG_B = angleFuncs.QuatRot(quatE2B, accCG_E)
        accIMU_B = angleFuncs.QuatRot(quatE2B, accIMU_E)

        # Derive p, q, r from euler angles
        p = solvers.derivative(eulB2E[:, 0].reshape(-1, 1), time)
        q = solvers.derivative(eulB2E[:, 1].reshape(-1, 1), time)
        r = solvers.derivative(eulB2E[:, 2].reshape(-1, 1), time)

        # Write data to new DataFrame
        flips = {False:1, True:-1}

        OT_Data = pd.DataFrame()
        OT_Data['t'] = time

        OT_Data['x'] = posCG_E[:, 0]
        OT_Data['y'] = posCG_E[:, 1]
        OT_Data['z'] = posCG_E[:, 2]
        OT_Data['x_g'] = posCO_G[:, 0]
        OT_Data['y_g'] = posCO_G[:, 1]
        OT_Data['z_g'] = posCO_G[:, 2]

        OT_Data['roll'] = angleFuncs.unwrapPi(eulB2E[:, 0])
        OT_Data['pitch'] = angleFuncs.unwrapPi(eulB2E[:, 1])
        OT_Data['yaw'] = angleFuncs.unwrapPi(eulB2E[:, 2])

        OT_Data['u_E'] = velCG_E[:, 0] 
        OT_Data['v_E'] = velCG_E[:, 1]
        OT_Data['w_E'] = velCG_E[:, 2]
        OT_Data['u'] = velCG_B[:, 0] 
        OT_Data['v'] = velCG_B[:, 1] 
        OT_Data['w'] = velCG_B[:, 2] 
        OT_Data['u_air'] = velWind_B[:, 0] 
        OT_Data['v_air'] = velWind_B[:, 1] 
        OT_Data['w_air'] = velWind_B[:, 2] 

        OT_Data['ax_E'] = accCG_E[:, 0]
        OT_Data['ay_E'] = accCG_E[:, 1]
        OT_Data['az_E'] = accCG_E[:, 2]
        OT_Data['ax'] = accCG_B[:, 0] 
        OT_Data['ay'] = accCG_B[:, 1] 
        OT_Data['az'] = accCG_B[:, 2] 

        OT_Data['p'] = p
        OT_Data['q'] = q
        OT_Data['r'] = r

    return OT_Data



def V1_import_OT(rowIdx, logFile, rigidBodyName, applyFreqFilter = True, filterOTSpikes = True, velocityCutoffHz = 40):
    filename = logFile.loc[rowIdx, 'Raw OptiTrack Name']
    filedir = logFile.loc[rowIdx, 'Raw OT Path']
    print('[ INFO ] Importing OptiTrack data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))

    configData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', nrows=1, header=None)
    # Extract save parameters
    saveParams = {}
    k, v = None, None
    for i in configData:
        # Even indices correspond to keys, odd to values
        if not i % 2:
            k = configData[i][0]
        else:
            v = configData[i][0]
            saveParams.update({k:v})

    fullRawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', 
                                        header=None, skiprows=3, low_memory=False)
    idxQuadrotor = np.where(fullRawData.loc[0, :] == rigidBodyName)[0]
    if len(idxQuadrotor) < 1:
        raise ValueError('Provided rigidBodyName, "{}", not found in {}'.format(rigidBodyName, os.path.join(filedir, '{}.csv'.format(filename))))
    else:
        print('[ INFO ] Found {} as rigidBodyName'.format(rigidBodyName))
        # Create new DataFrame with specified quadrotor data
        colNames = ['Frame', 'Time']
        for i in idxQuadrotor:
            colNames.append('{}_{}'.format(fullRawData.loc[2, i], fullRawData.loc[3, i]))
        rawBodyData = fullRawData.loc[4:, idxQuadrotor].to_numpy()
        frame = fullRawData.loc[4:, 0].to_numpy()
        time = fullRawData.loc[4:, 1].to_numpy()
        rawBodyData = np.hstack((frame.reshape(-1, 1), time.reshape(-1, 1), rawBodyData)).astype(float)

        bodyData = pd.DataFrame(data=rawBodyData, columns=colNames)

        idxStart_old = 0
        idxEnd_old = len(bodyData)
        # If there are NaNs in bodyData, attempt to interpolate
        if bodyData.isnull().values.any():
            # For each OptiTrack signal
            for i, col in enumerate(bodyData.columns):
                # Find NaN locations, if any
                locNaNs = bodyData[col].isnull()
                idxNaNs = np.where(locNaNs)[0]
                # If there are NaNs in signal
                if len(idxNaNs) > 0:
                    # If NaNs are at the start of end of signal, then we cannot interpolate on these regions
                    # Find where data starts and ends
                    idxsWithData = np.where(~locNaNs)[0]
                    if not len(idxsWithData):
                        raise ValueError('Inputted OptiTrack data is full of NaNs for column {}, cannot interpolate. Please verify that the OpitTrack data is correct.'.format(col))
                    idxStart = idxsWithData[0]
                    # idxEnd = idxsWithData[-11]
                    idxEnd = idxsWithData[-1]
                    # Replace current idxStart and idxEnd if necessary
                    if idxStart > idxStart_old:
                        idxStart_old = idxStart
                    if idxEnd < idxEnd_old:
                        idxEnd_old = idxEnd
                    usefulNaNIdxs = np.where((idxNaNs > idxStart) & (idxNaNs < idxEnd))[0]
                    if len(usefulNaNIdxs):
                        idxNaNs = idxNaNs[usefulNaNIdxs]
                        # Create a function which linearly interpolates the signal, based on known data, which takes
                        # index as input -> i.e. y = f(index) -< can be thought of as a proxy for time
                        func = interp1d(np.where(~locNaNs)[0], bodyData.to_numpy()[~locNaNs, i], kind = 'slinear')
                        # Interpolate NaN indexes 
                        bodyData.loc[idxNaNs, col] = func(idxNaNs)

        # Clip BodyData to regions where data is available
        bodyData = bodyData.iloc[idxStart_old:idxEnd_old, :]

        time = bodyData['Time'].to_numpy()
        dt = time[1] - time[0]
        fs = int(1/dt)

        # Remove 'spikes' (i.e. artifacts from loss of tracking) in position from OptiTrack data before processing. 
        # Spikes in rotation are removed later after conversion to euler angles
        sigma = 6
        interpolation_window = 10
        if filterOTSpikes:
            spikeCols = ['Position_X', 'Position_Y', 'Position_Z']
            for col in spikeCols:
                bodyData[col] = filterSpikes(bodyData[col].to_numpy().copy(), time, sigma_dx = sigma, window = interpolation_window).reshape(-1)


        # Quaternions in OptiTrack Frame. OptiTrack z, x, y -> x, y, z
        qx = bodyData['Rotation_X'].to_numpy()
        qy = bodyData['Rotation_Y'].to_numpy()
        qz = bodyData['Rotation_Z'].to_numpy()
        qw = bodyData['Rotation_W'].to_numpy()

        # G = OptiTrack Ground frame
        x_G = bodyData['Position_X'].to_numpy()
        y_G = bodyData['Position_Y'].to_numpy()
        z_G = bodyData['Position_Z'].to_numpy()
        posCO_G = np.vstack((x_G, y_G, z_G)).T


        # Open drone-config file 
        configFile = '{}.json'.format(logFile.loc[rowIdx, 'Drone Config File'])
        configPath = logFile.loc[rowIdx, 'Drone Config Path']
        path2File = os.path.join(configPath, configFile)

        with open(path2File, 'r') as f:
            droneConfigData = json.load(f)
            f.close()
        
        # ####################################################################
        # Rotate optitrack axis system to match that of Sihao
        rots = {'x':angleFuncs.EulRotX, 'y':angleFuncs.EulRotY, 'z':angleFuncs.EulRotZ}
        axisGCorrection = droneConfigData['optitrack ground axes correction']
        axisRotOrder = axisGCorrection['order'].lower()
        Rot1 = rots[axisRotOrder[0]](float(axisGCorrection[axisRotOrder[0]])*np.pi/180)
        Rot2 = rots[axisRotOrder[1]](float(axisGCorrection[axisRotOrder[1]])*np.pi/180)
        Rot3 = rots[axisRotOrder[2]](float(axisGCorrection[axisRotOrder[2]])*np.pi/180)
        R_axisCorrectionRotation = np.array(np.matmul(Rot1, np.matmul(Rot2, Rot3)))       

        posG = np.vstack((x_G, y_G, z_G)).T
        posC = np.matmul(np.vstack((R_axisCorrectionRotation,)*len(time)).reshape(-1, 3, 3), posG.reshape(-1, 3, 1)).reshape(-1, 3)

        x_G, y_G, z_G = posC[:, 0], posC[:, 1], posC[:, 2]
        posCO_G = np.vstack((x_G, y_G, z_G)).T

        eulGCorrection = np.array((float(axisGCorrection['x'])*np.pi/180, float(axisGCorrection['y'])*np.pi/180, float(axisGCorrection['z'])*np.pi/180))
        quatGCorrection = angleFuncs.Eul2Quat(eulGCorrection)
        quatG = np.vstack((qw, qx, qy, qz)).T
        quatC = angleFuncs.quatMul(np.vstack((quatGCorrection, )*len(quatG)), quatG)
        qw, qx, qy, qz = quatC[:, 0], quatC[:, 1], quatC[:, 2], quatC[:, 3]

        # Define rotation matrices for each axis 
        R_1 = np.array([(1 - 2*(qy*qy + qz*qz)), (2*(qx*qy - qw*qz)), (2*(qw*qy + qx*qz))])
        R_2 = np.array([(2*(qx*qy + qw*qz)), (1 - 2*(qx*qx + qz*qz)), (2*(qy*qz - qw*qx))])
        R_3 = np.array([2*(qx*qz - qw*qy), (2*(qw*qx + qy*qz)), (1 - 2*(qx*qx + qy*qy))])

        # Manipulate the indices of the rotation matrices above to get a vector of form
        # N x [3 x 3] such that each element corresponds to the rotation matrix for that
        # specific sample
        R_1 = R_1.T
        R_2 = R_2.T
        R_3 = R_3.T
        R_stack = np.zeros((3*len(R_1), 3))
        R_stack[0:(3*len(R_1)):3] = R_1
        R_stack[1:(3*len(R_1)):3] = R_2
        R_stack[2:(3*len(R_1)):3] = R_3
        R = R_stack.reshape((len(R_1), 3, 3))

        # G = OptiTrack ground, O = OptiTrack body
        # Default signs for angles in Sihao's frame
        # axisOCorrection = logFile.loc[rowIdx, 'Initial Yaw Offset']*np.pi/180 - 90*np.pi/180
        axisOCorrection = 0
        # Axis map for roll pitch yaw
        defaultMap = np.array([0, 1, 2])
        # defaultSigns = np.array([1, -1, -1])
        defaultSigns = np.array([1, -1, 1])
        axisMap = np.array(np.matmul(angleFuncs.EulRotY(axisOCorrection), defaultMap))
        signs = np.array(np.matmul(angleFuncs.EulRotY(axisOCorrection), defaultSigns))
        # signs = defaultSigns.reshape(1, -1)
        # Roll around OptiTrack Z 
        _roll_G2O = np.arctan2(R[:, 1, 0], R[:, 1, 1])
        # Pitch around OptiTrack X
        _pitch_G2O = np.arctan2(-R[:, 1, 2], np.real(np.sqrt(1 - R[:, 1, 2]**2)))
        # Yaw around OptiTrack Y
        _yaw_G2O = np.arctan2(R[:, 0, 2], R[:, 2, 2])

        anglesG2O = [_roll_G2O, _pitch_G2O, _yaw_G2O]

        roll_G2O = anglesG2O[int(abs(axisMap[0][0]))]*np.around(signs[0][0], 10)
        pitch_G2O = anglesG2O[int(abs(axisMap[0][1]))]*np.around(signs[0][1], 10)
        yaw_G2O = anglesG2O[int(abs(axisMap[0][2]))]*np.around(signs[0][2], 10)

        # unwrap yaw_G2O from 0, 2pi
        yaw_G2O = angleFuncs.unwrapPi(yaw_G2O)

        yaw_E2B_bias = yaw_G2O 
        roll_E2B_bias = roll_G2O
        pitch_E2B_bias = pitch_G2O

        # Get rotation matrix from body to earth frame to extract euler angles
        R_B2E = np.matmul(angleFuncs.EulRotZ_arr(yaw_E2B_bias), np.matmul(angleFuncs.EulRotY_arr(pitch_E2B_bias), angleFuncs.EulRotX_arr(roll_E2B_bias)))

        roll_E2B = np.arctan2(R_B2E[:, 2, 1], R_B2E[:, 2, 2])
        pitch_E2B = np.arctan2(-R_B2E[:, 2, 0], np.real(np.sqrt(1 - np.square(R_B2E[:, 2, 0]))))
        yaw_E2B = np.arctan2(R_B2E[:, 1, 0], R_B2E[:, 0, 0])

        # Unwrap yaw_E2B from 0, 2pi
        yaw_E2B = angleFuncs.unwrapPi(yaw_E2B)

        # Apply filter to attitude to remove large peaks in its derivatives
        if applyFreqFilter:
            roll_E2B = filtering._ButterFilter(roll_E2B, fs, 4, 10, 'low')
            pitch_E2B = filtering._ButterFilter(pitch_E2B, fs, 4, 10, 'low')
            yaw_E2B = filtering._ButterFilter(yaw_E2B, fs, 4, 10, 'low')


        axisCorrection = droneConfigData['axis direction correction']

        rots = {'x':angleFuncs.EulRotX, 'y':angleFuncs.EulRotY, 'z':angleFuncs.EulRotZ}
        axisRotOrder = axisCorrection['order'].lower()
        Rot1 = rots[axisRotOrder[0]](float(axisCorrection[axisRotOrder[0]])*np.pi/180)
        Rot2 = rots[axisRotOrder[1]](float(axisCorrection[axisRotOrder[1]])*np.pi/180)
        Rot3 = rots[axisRotOrder[2]](float(axisCorrection[axisRotOrder[2]])*np.pi/180)
        R_G2E = np.array(np.matmul(Rot1, np.matmul(Rot2, Rot3)))

        # Derive the position, velocity and acceleration information of the c.g. and IMU

        # Transform position from OptiTrack ground frame to E-Frame
        # Convert matrix to array so that we can apply element-wise multiplication
        R_G2E = np.array(R_G2E)
        # posCO_E = np.matmul(posCO_G.reshape(-1, 1, 3), np.vstack((R_G2E.T,)*posCO_G.shape[0]).reshape(-1, 3, 3)).reshape(-1, 3)
        posCO_E = np.matmul(np.vstack((R_G2E,)*posCO_G.shape[0]).reshape(-1, 3, 3), posCO_G.reshape(-1, 3, 1)).reshape(-1, 3)
        # posCO_E = np.matmul(posCO_G.reshape(-1, 1, 3), np.vstack((R_G2E.T,)*posCO_G.shape[0]).reshape(-1, 3, 3)).reshape(-1, 3)*np.sin(logFile.loc[rowIdx, 'Initial Yaw Offset']*np.pi/180)
        # Position c.g. w.r.t drone body frame, assume it is more-or-less at origin
        posCG_B = np.array([0, 0, 0])
        # Assume IMU is at c.g. position
        posIMU_B = np.array([0, 0, 0])
        # Position of cg of OptiTrack Markers on drone 
        posCO_B_dict = droneConfigData['optitrack marker cg offset']
        for k, v in posCO_B_dict.items():
            # Convert string input to floats
            posCO_B_dict.update({k:float(v)})


        # # Need to rotate by np.pi to get coordinates in body frame
        # posCO_B = np.array(np.array([posCO_B_dict['x'], posCO_B_dict['y'], posCO_B_dict['z']])*angleFuncs.EulRotX(-1*np.pi))
        posCO_B = np.array([posCO_B_dict['x'], posCO_B_dict['y'], posCO_B_dict['z']])

        # Derive position in E-frame from B-frame
        Rot_B2E = np.matmul(angleFuncs.EulRotZ_arr(yaw_E2B), np.matmul(angleFuncs.EulRotY_arr(pitch_E2B), angleFuncs.EulRotX_arr(roll_E2B)))
        posCG_E = posCO_E + np.matmul(Rot_B2E, np.vstack(((posCG_B - posCO_B),)*len(Rot_B2E)).reshape(-1, 3, 1)).reshape(-1, 3)
        posIMU_E = posCO_E + np.matmul(Rot_B2E, np.vstack(((posIMU_B - posCO_B),)*len(Rot_B2E)).reshape(-1, 3, 1)).reshape(-1, 3)


        # Derive velocity from position data, in E-frame
        velCO_E = solvers.derivative(posCO_E, time)
        velCG_E = solvers.derivative(posCG_E, time)
        velIMU_E = solvers.derivative(posIMU_E, time)

        # Apply filter to velocities to remove large spikes of OptiTrack (e.g. due to loss of tracking)
        # Filtering not applied direct to position since outliers are harder to detect (i.e. box-like signal forms
        # instead of large, instantenous, peaks)
        if applyFreqFilter:
            # plt.psd(velCO_E[:, 0]) # Cut-off at 4 Hz
            for i in range(velCO_E.shape[1]):
                velCO_E[:, i] = filtering._ButterFilter(velCO_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velCG_E[:, i] = filtering._ButterFilter(velCG_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velIMU_E[:, i] = filtering._ButterFilter(velIMU_E[:, i], fs, 4, velocityCutoffHz, 'low')


        # Derive accelerations from velocity data, in E-frame
        accCO_E = solvers.derivative(velCO_E, time)
        accCG_E = solvers.derivative(velCG_E, time)
        accIMU_E = solvers.derivative(velIMU_E, time)

        # Convert positions and velocities in E-frame to B-frame
        Rot_E2B = np.matmul(angleFuncs.EulRotX_arr(-1*roll_E2B), np.matmul(angleFuncs.EulRotY_arr(-1*pitch_E2B), angleFuncs.EulRotZ_arr(-1*yaw_E2B)))

        velCO_B = np.matmul(Rot_E2B, velCO_E.reshape(-1, 3, 1)).reshape(-1, 3)
        velCG_B = np.matmul(Rot_E2B, velCG_E.reshape(-1, 3, 1)).reshape(-1, 3)
        velIMU_B = np.matmul(Rot_E2B, velIMU_E.reshape(-1, 3, 1)).reshape(-1, 3)

        accCO_B = np.matmul(Rot_E2B, accCO_E.reshape(-1, 3, 1)).reshape(-1, 3)
        accCG_B = np.matmul(Rot_E2B, accCG_E.reshape(-1, 3, 1)).reshape(-1, 3)
        accIMU_B = np.matmul(Rot_E2B, accIMU_E.reshape(-1, 3, 1)).reshape(-1, 3)


        # Derive angular rates and accelerations
        dRoll_E2B = solvers.derivative(roll_E2B.reshape(-1, 1), time)
        dPitch_E2B = solvers.derivative(pitch_E2B.reshape(-1, 1), time)
        dYaw_E2B = solvers.derivative(yaw_E2B.reshape(-1, 1), time)

        _AAZeros = np.zeros(roll_E2B.shape)
        _AAOnes = np.ones(roll_E2B.shape)
        AA = np.array([[_AAOnes, _AAZeros, -1*np.sin(pitch_E2B)],
                        [_AAZeros, np.cos(roll_E2B), np.sin(roll_E2B)*np.cos(pitch_E2B)],
                        [_AAZeros, -1*np.sin(roll_E2B), np.cos(roll_E2B)*np.cos(pitch_E2B)]]).reshape(-1, 3, 3)

        omega_B_OT = np.matmul(AA, np.hstack((dRoll_E2B, dPitch_E2B, dYaw_E2B)).reshape(-1, 3, 1)).reshape(-1, 3)

        alpha_B_OT = solvers.derivative(omega_B_OT, time)

        # Write data to new DataFrame
        flips = {False:1, True:-1}

        OT_Data = pd.DataFrame()
        OT_Data['t'] = time

        OT_Data['x'] = posCG_E[:, 0]
        OT_Data['y'] = posCG_E[:, 1]
        OT_Data['z'] = posCG_E[:, 2]
        OT_Data['x_g'] = posCO_G[:, 0]
        OT_Data['y_g'] = posCO_G[:, 1]
        OT_Data['z_g'] = posCO_G[:, 2]

        # OT_Data['posCO_E'] = posCO_E

        OT_Data['roll'] = roll_E2B * flips[droneConfigData['flip attitude sign']['roll']] 
        OT_Data['pitch'] = pitch_E2B * flips[droneConfigData['flip attitude sign']['pitch']] 
        OT_Data['yaw'] = yaw_E2B * flips[droneConfigData['flip attitude sign']['yaw']] 

        # OT_Data['velCO_E'] = velCO_E

        OT_Data['u_E'] = velCG_E[:, 0] 
        OT_Data['v_E'] = velCG_E[:, 1]
        OT_Data['w_E'] = velCG_E[:, 2]
        OT_Data['u'] = velCG_B[:, 0] * flips[droneConfigData['flip accelerometer sign']['ax']] 
        OT_Data['v'] = velCG_B[:, 1] * flips[droneConfigData['flip accelerometer sign']['ay']] 
        OT_Data['w'] = velCG_B[:, 2] * flips[droneConfigData['flip accelerometer sign']['az']] 

        # OT_Data['accCO_B'] = accCO_B

        OT_Data['ax_E'] = accCG_E[:, 0]
        OT_Data['ay_E'] = accCG_E[:, 1]
        OT_Data['az_E'] = accCG_E[:, 2]
        OT_Data['ax'] = accCG_B[:, 0] * flips[droneConfigData['flip accelerometer sign']['ax']] 
        OT_Data['ay'] = accCG_B[:, 1] * flips[droneConfigData['flip accelerometer sign']['ay']] 
        OT_Data['az'] = accCG_B[:, 2] * flips[droneConfigData['flip accelerometer sign']['az']] 

        OT_Data['p'] = omega_B_OT[:, 0] * flips[droneConfigData['flip attitude sign']['roll']] 
        OT_Data['q'] = omega_B_OT[:, 1] * flips[droneConfigData['flip attitude sign']['pitch']] 
        OT_Data['r'] = omega_B_OT[:, 2] * flips[droneConfigData['flip attitude sign']['yaw']] 

        OT_Data['dp'] = alpha_B_OT[:, 0] * flips[droneConfigData['flip attitude sign']['roll']] 
        OT_Data['dq'] = alpha_B_OT[:, 1] * flips[droneConfigData['flip attitude sign']['pitch']] 
        OT_Data['dr'] = alpha_B_OT[:, 2] * flips[droneConfigData['flip attitude sign']['yaw']] 
    return OT_Data



def V2_import_OT(rowIdx, logFile, rigidBodyName, applyFreqFilter = True, filterOTSpikes = True, velocityCutoffHz = 40):
    filename = logFile.loc[rowIdx, 'Raw OptiTrack Name']
    filedir = logFile.loc[rowIdx, 'Raw OT Path']
    print('[ INFO ] Importing OptiTrack data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))

    configData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', nrows=1, header=None)
    # Extract save parameters
    saveParams = {}
    k, v = None, None
    for i in configData:
        # Even indices correspond to keys, odd to values
        if not i % 2:
            k = configData[i][0]
        else:
            v = configData[i][0]
            saveParams.update({k:v})

    fullRawData = pd.read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', 
                                        header=None, skiprows=3, low_memory=False)
    idxQuadrotor = np.where(fullRawData.loc[0, :] == rigidBodyName)[0]
    if len(idxQuadrotor) < 1:
        raise ValueError('Provided rigidBodyName, "{}", not found in {}'.format(rigidBodyName, os.path.join(filedir, '{}.csv'.format(filename))))
    else:
        print('[ INFO ] Found {} as rigidBodyName'.format(rigidBodyName))
        # Create new DataFrame with specified quadrotor data
        colNames = ['Frame', 'Time']
        for i in idxQuadrotor:
            colNames.append('{}_{}'.format(fullRawData.loc[2, i], fullRawData.loc[3, i]))
        rawBodyData = fullRawData.loc[4:, idxQuadrotor].to_numpy()
        frame = fullRawData.loc[4:, 0].to_numpy()
        time = fullRawData.loc[4:, 1].to_numpy()
        rawBodyData = np.hstack((frame.reshape(-1, 1), time.reshape(-1, 1), rawBodyData)).astype(float)

        bodyData = pd.DataFrame(data=rawBodyData, columns=colNames)

        idxStart_old = 0
        idxEnd_old = len(bodyData)
        # If there are NaNs in bodyData, attempt to interpolate
        if bodyData.isnull().values.any():
            # For each OptiTrack signal
            for i, col in enumerate(bodyData.columns):
                # Find NaN locations, if any
                locNaNs = bodyData[col].isnull()
                idxNaNs = np.where(locNaNs)[0]
                # If there are NaNs in signal
                if len(idxNaNs) > 0:
                    # If NaNs are at the start of end of signal, then we cannot interpolate on these regions
                    # Find where data starts and ends
                    idxsWithData = np.where(~locNaNs)[0]
                    if not len(idxsWithData):
                        raise ValueError('Inputted OptiTrack data is full of NaNs for column {}, cannot interpolate. Please verify that the OpitTrack data is correct.'.format(col))
                    idxStart = idxsWithData[0]
                    # idxEnd = idxsWithData[-11]
                    idxEnd = idxsWithData[-1]
                    # Replace current idxStart and idxEnd if necessary
                    if idxStart > idxStart_old:
                        idxStart_old = idxStart
                    if idxEnd < idxEnd_old:
                        idxEnd_old = idxEnd
                    usefulNaNIdxs = np.where((idxNaNs > idxStart) & (idxNaNs < idxEnd))[0]
                    if len(usefulNaNIdxs):
                        idxNaNs = idxNaNs[usefulNaNIdxs]
                        # Create a function which linearly interpolates the signal, based on known data, which takes
                        # index as input -> i.e. y = f(index) -< can be thought of as a proxy for time
                        func = interp1d(np.where(~locNaNs)[0], bodyData.to_numpy()[~locNaNs, i], kind = 'slinear')
                        # Interpolate NaN indexes 
                        bodyData.loc[idxNaNs, col] = func(idxNaNs)

        # Clip BodyData to regions where data is available
        bodyData = bodyData.iloc[idxStart_old:idxEnd_old, :]
        
        time = bodyData['Time'].to_numpy()
        dt = time[1] - time[0]
        fs = float(configData.iloc[0, 7])

        # Remove 'spikes' (i.e. artifacts from loss of tracking) in position from OptiTrack data before processing. 
        # Spikes in rotation are removed later after conversion to euler angles
        sigma = 6
        interpolation_window = 10
        if filterOTSpikes:
            spikeCols = ['Position_X', 'Position_Y', 'Position_Z']
            for col in spikeCols:
                bodyData[col] = filterSpikes(bodyData[col].to_numpy().copy(), time, sigma_dx = sigma, window = interpolation_window).reshape(-1)


        # Quaternions in OptiTrack Frame. OptiTrack z, x, y -> x, y, z
        qx = bodyData['Rotation_X'].to_numpy()
        qy = bodyData['Rotation_Y'].to_numpy()
        qz = bodyData['Rotation_Z'].to_numpy()
        qw = bodyData['Rotation_W'].to_numpy()
        quatOT = np.vstack((qw, qx, qy, qz)).T

        # G = OptiTrack Ground frame
        x_G = bodyData['Position_X'].to_numpy()
        y_G = bodyData['Position_Y'].to_numpy()
        z_G = bodyData['Position_Z'].to_numpy()
        posCO_G = np.vstack((x_G, y_G, z_G)).T

        # Open drone-config file 
        configFile = '{}.json'.format(logFile.loc[rowIdx, 'Drone Config File'])
        configPath = logFile.loc[rowIdx, 'Drone Config Path']
        path2File = os.path.join(configPath, configFile)

        with open(path2File, 'r') as f:
            droneConfigData = json.load(f)
            f.close()
        
        # ===============================================================================================
        # Coordinate system transformation
        # ===============================================================================================
        # Converion of optitrack ground system (G) to earth system (E)
        # OptiTrack axes system (G) - from perspective of computer - x forward, y up, z right
        # y   x
        # |  /
        # | /
        # |/_____z
        #
        # Earth axes system (E) - from perspective of computer - x right, y backwards (towards computer), z down
        #      |----x
        #     /|
        #    / |
        #   /  |
        #  y   z
        #
        # Define individual rotations for G to E in quaternions
        G2E_quat_x = angleFuncs.Eul2Quat(np.array([90*np.pi/180, 0, 0]))
        G2E_quat_y = angleFuncs.Eul2Quat(np.array([0, 0, 0]))
        G2E_quat_z = angleFuncs.Eul2Quat(np.array([0, 0, 90*np.pi/180]))
        # Apply rotations, in quaternions
        G2E_quat = angleFuncs.QuatConj(angleFuncs.quatMul(G2E_quat_x, angleFuncs.quatMul(G2E_quat_y, G2E_quat_z)))

        # Get position in E frame (from G frame)
        posCO_E = angleFuncs.QuatRot(G2E_quat.reshape(1, -1), posCO_G)

        # quatOT describes angles transforming G frame into Optitrack Body (O) frame. 
        # Here we first go from the E-frame to the G-frame, then we apply quatOT to transform into the O-frame
        #   Thus, we have a rotation from E-frame to O-frame
        quatE2O = angleFuncs.quatMul(quatOT, angleFuncs.QuatConj(G2E_quat.reshape(-1, 4)))
        # Finally, we can apply a correction from O-frame to B-frame (quadrotor body, which is coincident with E-frame @ origin)
        #   This gives us our rotation from E-Frame to B-frame (i.e. quadrotor attitude w.r.t E-frame)
        quatE2B = angleFuncs.QuatConj(angleFuncs.quatMul(G2E_quat.reshape(-1, 4), quatE2O))
        # Take the conjugate for angles describing transformation form B-frame to E-frame
        quatB2E = angleFuncs.QuatConj(quatE2B)
        # Extract euler equivalents        
        eulB2E = angleFuncs.Quat2Eul(quatB2E) 

        # ===============================================================================================
        # Velocity derivation
        # ===============================================================================================
        posCG_B = np.array([0, 0, 0])
        # Assume IMU is at c.g. position
        posIMU_B = np.array([0, 0, 0])
        # Position of cg of OptiTrack Markers on drone 
        posCO_B_dict = droneConfigData['optitrack marker cg offset']
        for k, v in posCO_B_dict.items():
            # Convert string input to floats
            posCO_B_dict.update({k:float(v)})

        posCO_B = np.array([posCO_B_dict['x'], posCO_B_dict['y'], posCO_B_dict['z']])
        # import code
        # code.interact(local=locals())
        posCG_E = posCO_E + angleFuncs.QuatRot(quatB2E, np.vstack((posCG_B - posCO_B,)*len(quatE2B)))
        posIMU_E = posCO_E + angleFuncs.QuatRot(quatB2E, np.vstack((posIMU_B - posCO_B,)*len(quatE2B)))

        # Derive velocity from position data, in E-frame
        velCO_E = solvers.derivative(posCO_E, time)
        velCG_E = solvers.derivative(posCG_E, time)
        velIMU_E = solvers.derivative(posIMU_E, time)

        # Apply filter to velocities to remove large spikes of OptiTrack (e.g. due to loss of tracking)
        # Filtering not applied direct to position since outliers are harder to detect (i.e. box-like signal forms
        # instead of large, instantenous, peaks)
        if applyFreqFilter:
            # plt.psd(velCO_E[:, 0]) # Cut-off at 4 Hz
            for i in range(velCO_E.shape[1]):
                velCO_E[:, i] = filtering._ButterFilter(velCO_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velCG_E[:, i] = filtering._ButterFilter(velCG_E[:, i], fs, 4, velocityCutoffHz, 'low')
                velIMU_E[:, i] = filtering._ButterFilter(velIMU_E[:, i], fs, 4, velocityCutoffHz, 'low')

        # Derive accelerations from velocity data, in E-frame
        accCO_E = solvers.derivative(velCO_E, time)
        accCG_E = solvers.derivative(velCG_E, time)
        accIMU_E = solvers.derivative(velIMU_E, time)

        # Convert positions and velocities in E-frame to B-frame
        velCO_B = angleFuncs.QuatRot(quatE2B, velCO_E)
        velCG_B = angleFuncs.QuatRot(quatE2B, velCG_E)
        velIMU_B = angleFuncs.QuatRot(quatE2B, velIMU_E)

        accCO_B = angleFuncs.QuatRot(quatE2B, accCO_E)
        accCG_B = angleFuncs.QuatRot(quatE2B, accCG_E)
        accIMU_B = angleFuncs.QuatRot(quatE2B, accIMU_E)

        
        # Derive p, q, r from euler angles
        p = solvers.derivative(eulB2E[:, 0].reshape(-1, 1), time)
        q = solvers.derivative(eulB2E[:, 1].reshape(-1, 1), time)
        r = solvers.derivative(eulB2E[:, 2].reshape(-1, 1), time)


        # Write data to new DataFrame
        OT_Data = pd.DataFrame()
        OT_Data['t'] = time

        OT_Data['x'] = posCG_E[:, 0]
        OT_Data['y'] = posCG_E[:, 1]
        OT_Data['z'] = posCG_E[:, 2]
        OT_Data['x_g'] = posCO_G[:, 0]
        OT_Data['y_g'] = posCO_G[:, 1]
        OT_Data['z_g'] = posCO_G[:, 2]

        OT_Data['roll'] = angleFuncs.unwrapPi(eulB2E[:, 0])
        OT_Data['pitch'] = angleFuncs.unwrapPi(eulB2E[:, 1])
        OT_Data['yaw'] = angleFuncs.unwrapPi(eulB2E[:, 2])

        OT_Data['u_E'] = velCG_E[:, 0] 
        OT_Data['v_E'] = velCG_E[:, 1]
        OT_Data['w_E'] = velCG_E[:, 2]
        OT_Data['u'] = velCG_B[:, 0] 
        OT_Data['v'] = velCG_B[:, 1] 
        OT_Data['w'] = velCG_B[:, 2] 

        OT_Data['ax_E'] = accCG_E[:, 0]
        OT_Data['ay_E'] = accCG_E[:, 1]
        OT_Data['az_E'] = accCG_E[:, 2]
        OT_Data['ax'] = accCG_B[:, 0] 
        OT_Data['ay'] = accCG_B[:, 1] 
        OT_Data['az'] = accCG_B[:, 2]

        OT_Data['p'] = p
        OT_Data['q'] = q
        OT_Data['r'] = r

        # import code
        # code.interact(local=locals())

    return OT_Data
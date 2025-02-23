import numpy as np
import pandas as pd
from scipy.signal import correlate as sp_correlate

from common import angleFuncs
from processing import optitrack, utility, cleanflight, indiflight

def runImport(rowIdx, log, OB_samplingRate, maxLag_seconds, alignUsing, filterOutliersOT, velocityCutoffHz = 40, importBackend_OB = None, importBackend_OT = None, useLegacyOBOT = [False, False], attitudeFromOB = True, doCGCorrection = True, showAlignedPlots = True):
    # If importBackend_XX is None, then infer based on log file
    # Define known controllers and map combinations of indoor/outdoor-flightcontroller to import script 
    import_OB = _FC_HANDLER(log.loc[rowIdx, 'Flight controller'], OB_backend=importBackend_OB, useLegacyOBOT=useLegacyOBOT)
    
    # Import on-board data from quadrotor
    rigidBody = log.loc[rowIdx, 'Rigid Body Name']
    OB_data = import_OB(rowIdx, log, rowBreak = log.loc[rowIdx, 'OB row skip'], numCols = log.loc[rowIdx, 'OB num columns'], doCGCorrection = doCGCorrection)

    # Sampling frequency of the OB data can be inconsistent, so we shall resample it
    # Need to convert angles to continuous values to avoid artifacts from forming due to interpolation
    OB_data['roll'] = angleFuncs.unwrapPi(OB_data['roll'].to_numpy())
    OB_data['pitch'] = angleFuncs.unwrapPi(OB_data['pitch'].to_numpy())
    OB_data['yaw'] = angleFuncs.unwrapPi(OB_data['yaw'].to_numpy())

    # OB_data = resample(OB_data, OB_samplingRate, interpolation_kind = 'cubic')
    OB_data = utility.resample(OB_data, OB_samplingRate, interpolation_kind = 'linear')

    dt = OB_data.loc[1, 't'] - OB_data.loc[0, 't']

    # Only attempt OT import if OT data is available
    knownOTImporters = {'V1_import_OT':optitrack.V1_import_OT,
                        'V2_import_OT':optitrack.V2_import_OT,
                        'V1_import_OT_withWind':optitrack.V1_import_OT_withWind,
                        'V2_import_OT_withWind':optitrack.V2_import_OT_withWind}
    if log.loc[rowIdx, 'Has OT data'].lower() == 'y':
        # Import OptiTrack Data
        # Get appropriate OT import function
        if importBackend_OT is None:
            if useLegacyOBOT[1]:
                # Check if has wind
                if 'Has OJF data' in log.columns:
                    if log.loc[rowIdx, 'Has OJF data'].lower() == 'y':
                        import_OT = knownOTImporters['V1_import_OT_withWind']
                    else:
                        import_OT = knownOTImporters['V1_import_OT']
                else:
                    import_OT = knownOTImporters['V1_import_OT']
            else:
                # Check if has wind
                if 'Has OJF data' in log.columns:
                    if log.loc[rowIdx, 'Has OJF data'].lower() == 'y':
                        import_OT = knownOTImporters['V2_import_OT_withWind']
                    else:
                        import_OT = knownOTImporters['V2_import_OT']
                else:
                    import_OT = knownOTImporters['V2_import_OT']
        else:
            # Check if passed function name exists
            if importBackend_OT in knownOTImporters:
                import_OT = knownOTImporters[importBackend_OT]
            else:
                raise ValueError('Passed importBackend_OT ({}) not known.'.format(importBackend_OT))

        OT_data = import_OT(rowIdx, log, rigidBody, applyFreqFilter = filterOutliersOT, filterOTSpikes = filterOutliersOT, velocityCutoffHz = velocityCutoffHz)

        # Resample OT data
        # OT_data_resampled = resample(OT_data, round(1/dt), interpolation_kind = 'cubic')
        OT_data_resampled = utility.resample(OT_data, round(1/dt), interpolation_kind = 'linear')
        # Reset time of OT
        OT_data_resampled['t'] = OT_data_resampled['t'] - OT_data_resampled['t'].to_numpy()[0]

        maxLag = int(maxLag_seconds/dt)
        lag, _ = findLagOBOT(OB_data, OT_data_resampled, alignUsing = alignUsing, maxLag = maxLag)
        lag_time = lag*dt

        OT_data_aligned = OT_data_resampled.copy(deep = True)
        OT_data_aligned['t'] = OT_data_aligned['t'] - lag_time

        # Since OptiTrack acts as local "GPS", and positions and velocities are w.r.t OptiTrack, we need to
        # align yaw angle of Onboard to that of OptiTrack
        idxAligned = np.where(OT_data_aligned >= 0)[0][0]
        OB_data['yaw'] = OB_data['yaw'] - OB_data['yaw'].to_numpy()[0] + OT_data_aligned['yaw'].to_numpy()[idxAligned]

        ###################################################################
        # Check alignment output
        if showAlignedPlots:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(OT_data_aligned['t'], OT_data_resampled['roll'], c='steelblue', label='OptiTrack')
            ax.plot(OB_data['t'], OB_data['roll'], c='firebrick', label='On-board')
            ax.legend(fontsize=14)
            ax.tick_params(which='both', direction='in', labelsize=14)
            ax.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 12)
            ax.set_ylabel(r'$\mathbf{Roll}$, rad', fontsize = 12)
            plt.tight_layout()
            # plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(OT_data_aligned['t'], OT_data_resampled['pitch'], c='steelblue', label='OptiTrack')
            ax.plot(OB_data['t'], OB_data['pitch'], c='firebrick', label='On-board')
            ax.legend(fontsize=14)
            ax.tick_params(which='both', direction='in', labelsize=14)
            ax.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 12)
            ax.set_ylabel(r'$\mathbf{Pitch}$, rad', fontsize = 12)
            plt.tight_layout()
            # plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(OT_data_aligned['t'], OT_data_resampled['yaw'], c='steelblue', label='OptiTrack')
            ax.plot(OB_data['t'], OB_data['yaw'], c='firebrick', label='On-board')
            ax.legend(fontsize=14)
            ax.tick_params(which='both', direction='in', labelsize=14)
            ax.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 12)
            ax.set_ylabel(r'$\mathbf{Yaw}$, rad', fontsize = 12)
            plt.tight_layout()
            # plt.show()


            g = np.vstack((np.array([0, 0, 9.81]),)*len(OB_data))
            # OB attitude gives rotations from body to earth
            gB = angleFuncs.QuatRot(angleFuncs.Eul2Quat(OB_data[['roll', 'pitch', 'yaw']].to_numpy()), g, rot = 'E2B')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(OB_data['t'], OB_data['ax'], c='firebrick', label='On-board (raw)')
            ax.plot(OB_data['t'], OB_data['ax'] + gB[:, 0], c='lightseagreen', label='On-board (g-corrected)')
            ax.plot(OT_data_aligned['t'], OT_data_resampled['ax'], c='steelblue', label='OptiTrack')
            ax.plot(OT_data_aligned['t'], OT_data_resampled['u'], c = 'orange', label = 'OptiTrack Velocity (u)')
            ax.legend(fontsize=14)
            ax.tick_params(which='both', direction='in', labelsize=14)
            ax.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 12)
            ax.set_ylabel(r'$\mathbf{Acceleration}$ $\mathbf{x}$, $ms^{-2}$ ($\mathbf{Velocity}$, $ms^{-1}$)', fontsize = 12)
            plt.tight_layout()
            # plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(OB_data['t'], OB_data['ay'], c='firebrick', label='On-board (raw)')
            ax.plot(OB_data['t'], OB_data['ay'] + gB[:, 1], c='lightseagreen', label='On-board (g-corrected)')    
            ax.plot(OT_data_aligned['t'], OT_data_resampled['ay'], c='steelblue', label='OptiTrack')
            ax.plot(OT_data_aligned['t'], OT_data_resampled['v'], c = 'orange', label = 'OptiTrack Velocity (v)')
            ax.legend(fontsize=14)
            ax.tick_params(which='both', direction='in', labelsize=14)
            ax.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 12)
            ax.set_ylabel(r'$\mathbf{Acceleration}$ $\mathbf{y}$, $ms^{-2}$ ($\mathbf{Velocity}$, $ms^{-1}$)', fontsize = 12)
            plt.tight_layout()
            # plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(OB_data['t'], OB_data['az'], c='firebrick', label='On-board (raw)')
            ax.plot(OB_data['t'], OB_data['az'] + gB[:, 2], c='lightseagreen', label='On-board (g-corrected)')    
            ax.plot(OT_data_aligned['t'], OT_data_resampled['az'], c='steelblue', label='OptiTrack')
            ax.plot(OT_data_aligned['t'], OT_data_resampled['w'], c = 'orange', label='OptiTrack Velocity (w)')
            ax.legend(fontsize=14)
            ax.tick_params(which='both', direction='in', labelsize=14)
            ax.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 12)
            ax.set_ylabel(r'$\mathbf{Acceleration}$ $\mathbf{z}$, $ms^{-2}$ ($\mathbf{Velocity}$, $ms^{-1}$)', fontsize = 12)
            plt.tight_layout()
            plt.show()

        # Note, accelerations may seem flipped, but that is due to effect of gravity and how the accelerometers measure work. Once the g component is removed, the accelerations should coincide. 
        rawData = combineOBOT(OB_data, OT_data_aligned, attitudeFromOB = attitudeFromOB)
    else:
        rawData = OB_data.copy()

    return rawData



def findLagOBOT(OB_Data, OT_Data, alignUsing = ['p', 'q', 'r'], corrCombine = 'product', maxLag = 500):
    if corrCombine.lower() not in ('sum', 'product'):
        raise ValueError('Unknown corrCombine, "{}", please use "sum" or "product"'.format(corrCombine))
    # Need to pad arrays to same size 
    N1 = OB_Data.shape[0]
    N2 = OT_Data.shape[0]
    corrs = []
    for i, param in enumerate(alignUsing):
        OB_param = OB_Data[param].to_numpy()
        OT_param = OT_Data[param].to_numpy()
        param_corr = sp_correlate(OB_param, OT_param, mode='full')
        # param_corr = sp_correlate(OB_param, OT_param, mode='valid')
        # param_corr = sp_correlate(OB_param, OT_param, mode='same')
        corrs.append(param_corr/np.max(param_corr))
    corrs = np.array(corrs)
    if corrCombine.lower() == 'product':
        fusedCorrs = np.prod(corrs, axis = 0)
    elif corrCombine.lower() == 'sum':
        fusedCorrs = np.sum(corrs, axis = 0)
    else:
        raise ValueError('Unknown corrCombine, "{}", please use "sum" or "product"'.format(corrCombine))
    # Snip correleation array into range of +- maxLag.
    zeroth_idx = (N1 - 1)
    if maxLag > zeroth_idx:
        maxLag = zeroth_idx
    lag = np.argmax(np.flip(fusedCorrs)[(zeroth_idx - maxLag):(zeroth_idx + maxLag + 1)]) - maxLag
    return lag, corrs.T



def combineOBOT(OB_data, OT_data, attitudeFromOB = True, timePrecision = 6):
    # Find starting and ending times (i.e. intersection of two data streams)
    startT = np.max((OB_data['t'].values[0], OT_data['t'].values[0]))
    endT = np.min((OB_data['t'].values[-1], OT_data['t'].values[-1]))

    OB_startIdx = np.where(np.around(OB_data['t'].to_numpy(), decimals=timePrecision) >= startT)[0][0]
    OB_endIdx = np.where(np.around(OB_data['t'].to_numpy(), decimals=timePrecision) <= endT)[0][-1]

    # Sometimes, the resampling causes some floating point errors (e.g. x.xxxE-16 instead of 0)
    # This therefore offsets the startIdx/endIdx by one. To correct this we use the decimal
    # places of the OB_data to round the OT_time array
    OT_startIdx = np.where(np.around(OT_data['t'].to_numpy(), decimals=timePrecision) >= startT)[0][0]
    OT_endIdx = np.where(np.around(OT_data['t'].to_numpy(), decimals=timePrecision) <= endT)[0][-1]

    # Extract necessary columns and cip data to starting and ending ranges
    OB_cols = ('t', 'w1', 'w2', 'w3', 'w4', 'ax', 'ay', 'az', 'p', 'q', 'r', 'w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD')
    OT_cols = ('x', 'y', 'z', 'u', 'v', 'w')
    if attitudeFromOB:
        OB_cols = OB_cols + ('roll', 'pitch', 'yaw')
    else:
        OT_cols = OT_cols + ('roll', 'pitch', 'yaw')

    combinedData = pd.DataFrame()
    for col in OB_cols:
        combinedData[col] = OB_data[col].to_numpy()[OB_startIdx:(OB_endIdx+1)]

    for col in OT_cols:
        if len(OT_data[col].to_numpy()[OT_startIdx:(OT_endIdx+1)]) < len(combinedData):
            combinedData[col] = OT_data[col].to_numpy()[OT_startIdx:(OT_endIdx+2)]
        elif len(OT_data[col].to_numpy()[OT_startIdx:(OT_endIdx+1)]) > len(combinedData):
            combinedData[col] = OT_data[col].to_numpy()[OT_startIdx:(OT_endIdx)]
        else:
            combinedData[col] = OT_data[col].to_numpy()[OT_startIdx:(OT_endIdx+1)]

    return combinedData


def _FC_HANDLER(flightController, OB_backend, **kwargs):
    knownOBImporters = {'V1_import_OB_btfl':cleanflight.V1_import_OB,
                        'V2_import_OB_btfl':cleanflight.V2_import_OB,
                        'import_OB_indiflight':indiflight.import_OB}
    fc_handlers = {
        'betaflight':BTFL_HANDLER,
        'indiflight':INDI_HANDLER
    }
    if OB_backend is None:
        if flightController.lower() in fc_handlers.keys():
            return fc_handlers[flightController.lower()](knownOBImporters, **kwargs)
        else:
            raise ValueError(f'Unknown flight controller: {flightController.lower()}')
    else:
        if OB_backend not in knownOBImporters.keys():
            raise ValueError(f'Passed importBacked_OB ({OB_backend}) not known.')
        return knownOBImporters[OB_backend]
    
def BTFL_HANDLER(knownOBImporters, useLegacyOBOT = [False, False], **kwargs):
    if useLegacyOBOT[0]:
        return knownOBImporters['V1_import_OB_btfl']
    else:
        return knownOBImporters['V2_import_OB_btfl']

def INDI_HANDLER(knownOBImporters, **kwargs):
    return knownOBImporters['import_OB_indiflight']
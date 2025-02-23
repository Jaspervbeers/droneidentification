'''
Script to identify polynomial models of a quadrotor through stepwise regression. 
'''
# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import os
import numpy as np
import pandas as pd
import pickle as pkl
import json
from datetime import datetime # Used to tag metadata with information about date of creation.

# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #
from processing import importing, filtering, utility, quadrotorFM

# [41, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 58, 59, 61, 65, 66, 67]
# 3, 4, "6-41", "43-55", "59-73", "75-98"
# ================================================================================================================================ #
# Definitions
# ================================================================================================================================ #
def ifNullDefault(extraction, default):
    if extraction is None:
        return default
    else:
        return extraction


def checkParamConsistency(currentParams, referenceParams):
    # If reference params is None, assume currentParams are master copy
    if referenceParams is None:
        return currentParams
    else:
        for k in referenceParams.keys():
            isConsistent = False
            try:
                isConsistent = currentParams[k] == referenceParams[k]
            except KeyError:
                raise ValueError('Parameter consistency check failed. Passed parameter keys do not match any of the references.')
            if not isConsistent:
                raise RuntimeError('Parameter consistency check failed: Current parameter ({} = {}) does not match reference parameter ({} = {})'.format(k, currentParams[k], k, referenceParams[k]))
        return currentParams


# ================================================================================================================================ #
# Processing
# ================================================================================================================================ #
'''
Extract & Define processing parameters
'''
# TODO: Save this config file as metadata
with open('processingConfig.json', 'r') as f:
    processingConfig = json.load(f)

# Extract logging file information
fileLogDir = ifNullDefault(processingConfig['logging file']['directory'], os.path.join(os.getcwd(), 'Data'))
fileLogName = processingConfig['logging file']['filename']
rowIdxs = processingConfig['logging file']['rows of flights to use (all)']

# Extract data importing information
importParams = processingConfig['data importing']
importRawData = importParams['import raw data']
resampleRate = float(importParams['resampling rate'])
useOnBoardAttitude = importParams['use onboard attitude']
doCGCorrection = importParams['do onboard c.g. correction']
filterOutliersOT = importParams['filter optitrack outliers']
velocityCutoffHz = importParams['optitrack velocity Hz cutoff']
saveRawData = importParams['save raw data']
importedDataSavePath = ifNullDefault(importParams['imported data save directory'], os.path.join(os.getcwd(), 'Data', 'processed', 'imported'))
alignUsing = importParams['align with optitrack using']
maxLag = float(importParams['max permitted lag for optitrack'])
showAlignedPlots = importParams['show onboard and optitrack alignment plots']

# Extract filtering information
filterParams = processingConfig['data filtering']
useConfigNoiseStats = filterParams['use noise statistics in drone config']
filterData = filterParams['run extended kalman filter']
saveEKFResults = filterParams['save kalman filter convergence results']
removeGComponent = filterParams['remove influence of gravity']
showPlots = filterParams['show filtering results']
saveFilteredData = filterParams['save filtered data']
filterSavePath = ifNullDefault(filterParams['filtered data save directory'], os.path.join(os.getcwd(), 'Data', 'processed', 'filtered'))
if removeGComponent:
    filterSavePath = os.path.join(filterSavePath, 'noGravity')
else:
    filterSavePath = os.path.join(filterSavePath, 'withGravity')
if not os.path.exists(filterSavePath):
    os.makedirs(filterSavePath)


# Open log_file
log = pd.read_csv(os.path.join(fileLogDir, '{}.csv'.format(fileLogName)), delimiter=',', header=0)

# Offset rowIdx to match python indexing, note that -2 is used since header is also skipped!
rowIdxs = utility._parseRows(rowIdxs)
rowIdxs = np.array(rowIdxs) - 2

# Check that the quadrotor names are correct for the selected rows, add additional checks if necessary
quadrotorName = log.loc[rowIdxs[0], 'Quadrotor']
quadBatteries = log.loc[rowIdxs[0], 'Batteries']
for i in range(len(rowIdxs)-1):
    rowQuadName = log.loc[rowIdxs[i+1], 'Quadrotor']
    rowQuadBatt = log.loc[rowIdxs[i+1], 'Batteries']
    if rowQuadName != quadrotorName:
        raise ValueError('Flight data mismatch: Expected quadrotor "{}" (row 0) but got "{}" (row {}) instead.'.format(quadrotorName, rowQuadName, i+1))
    if rowQuadBatt != quadBatteries:
        raise ValueError('Flight data mismatch: Expected battery configuration "{}" (row 0) but got "{}" (row {}) instead.'.format(quadBatteries, rowQuadBatt, i+1))


# Extract quadrotor configuration
r_sign = {'CCW':-1, 'CW':1}
droneParams, rotorConfig, rotorDir, idleRPM = utility.extractConfig(rowIdxs[0], log)
droneParams.update({'rho':1.225})
g = 9.81
droneParams.update({'g':g})
minRPM = {quadrotorName:float(idleRPM)}


'''
Import raw data
'''
if importRawData:
    # rawDataList = []
    rawDataDict = {}
    for row in rowIdxs:
        print('[ INFO ] Importing rowIdx: {}'.format(row +2))
        rawData = importing.runImport(rowIdx = row, log = log, OB_samplingRate = resampleRate, maxLag_seconds = maxLag, 
                                        alignUsing = alignUsing, filterOutliersOT=filterOutliersOT, velocityCutoffHz=velocityCutoffHz, 
                                        attitudeFromOB=useOnBoardAttitude, doCGCorrection=doCGCorrection, showAlignedPlots=showAlignedPlots)
        if saveRawData:
            print('[ INFO ] Saving imported data to {}'.format(importedDataSavePath))
            _filename = "{}-{}".format(log.loc[row, 'Onboard Name'], 'IM.csv') #IM suffix to indicate imported data
            savePath = os.path.join(importedDataSavePath, _filename)
            if not os.path.isdir(importedDataSavePath):
                os.makedirs(importedDataSavePath)
            rawData.to_csv(savePath)
            # Save importing parameters to metadata file
            now = datetime.today().isoformat()
            importParams.update({'Datetime (creation)':now})
            importParams.update({'log file':fileLogName})
            importParams.update({'rows of flights to use (all)':(rowIdxs + 2).tolist()})
            importParams.update({'current row':str(row + 2)})
            with open(os.path.join(importedDataSavePath, "{}-{}".format(log.loc[row, 'Onboard Name'], 'metadata.json')), 'w') as f:
                json.dump(importParams, f, indent = 4)
        rawDataDict.update({row:rawData})
else:
    print('[ INFO ] Loading rawData from {}'.format(importedDataSavePath))
    rawDataDict = {}
    referenceImportParams = None
    for row in rowIdxs:
        rawData = pd.read_csv(os.path.join(importedDataSavePath, "{}-{}".format(log.loc[row, 'Onboard Name'], 'IM.csv')))
        rawDataDict.update({row:rawData})
        # Load metadata and check consistency
        with open(os.path.join(importedDataSavePath, log.loc[row, 'Onboard Name'] + '-metadata.json'), 'r') as f:
            importParams = checkParamConsistency(json.load(f), referenceImportParams)
        referenceImportParams = {'resampling rate':importParams['resampling rate']}


'''
Filter data
'''
if filterData:
    filteredDataList = []
    for rowIdx, rawData in rawDataDict.items():
        # Check if rowIdx in rawDataDict
        if rowIdx in rowIdxs:
            droneMass = log.loc[rowIdx, 'Mass']*10**(-3)
            print(f'[ INFO ] Running EKF ({log.loc[rowIdx, "Onboard Name"]})...')
            filteredData, filtResults = filtering.runFilter(rowIdx, rawData, droneMass, droneParams, log, removeGravityComponent=removeGComponent, defaultNoise=(not useConfigNoiseStats), showPlots=showPlots)
            # Calculate induced velocity
            if filterParams['add induced velocity']:
                print('[ INFO ] Approximating induced velocity')
                filteredData['v_in'] = quadrotorFM.getInducedVelocity(filteredData, log, rowIdx, droneParams)
            # Add moments to filtered data
            print('[ INFO ] Calculating aerodynamic moments')
            filteredData = quadrotorFM.addMoments(droneParams, filteredData)            
            filteredDataList.append(filteredData)
            # Check if directory exists 
            if saveFilteredData:
                if not os.path.isdir(filterSavePath):
                    os.makedirs(filterSavePath)
                print('[ INFO ] Saving filtered data to {}'.format(filterSavePath))
                if saveEKFResults:
                    with open(os.path.join(filterSavePath, '{}-{}'.format(log.loc[rowIdx, 'Onboard Name'], 'EKFR.pkl')), 'wb') as f: # EKFR for KalmanFilter results
                        pkl.dump(filtResults, f)
                        f.close()
                filteredData.to_csv(os.path.join(filterSavePath, '{}-{}'.format(log.loc[rowIdx, 'Onboard Name'], 'FL.csv'))) # FL for filtered data
                # Save filtering parameters as metadata with some additional info
                now = datetime.today().isoformat()
                filterParams.update({'Datetime (creation)':now})
                filterParams.update({'log file':fileLogName})
                filterParams.update({'rows of flights to use (all)':(rowIdxs + 2).tolist()})
                filterParams.update({'current row':str(row + 2)})
                filterParams.update({'batteries':log.loc[rowIdx, "Batteries"]})
                with open(os.path.join(filterSavePath, '{}-{}'.format(log.loc[rowIdx, 'Onboard Name'], 'metadata.json')), 'w') as f:
                    json.dump(filterParams, f, indent = 4)


print('[ INFO ] [processQuadData.py] Done.')
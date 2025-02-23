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
from matplotlib import pyplot as plt
import json
import sys
from scipy.interpolate import interp1d

# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #
from processing import partitioning, utility

# This package relies on the system identification pipeline (sysidpipeline). 
with open('relativeImportLocations.json', 'r') as f:
    relativeLocs = json.load(f)

sys.path.append(relativeLocs['sysidpipeline'])
import SysID


# ================================================================================================================================ #
# Definitions
# ================================================================================================================================ #


def ifNullDefault(extraction, default):
    if extraction is None:
        return default
    else:
        return extraction


def genModelID(quadName, num, zeros = 3):
    num = '{}'.format(num).zfill(zeros)
    return 'MDL-{}-{}'.format(quadName,num)


def checkPath(savePath, mdlID):
    modelSavePath = os.path.join(savePath, mdlID)
    if os.path.isdir(modelSavePath):
        print('[ WARNING ] Model: {} already seems to exist. Continuing may override existing models.'.format(mdlID))
        print('[ WARNING ] Current sub-directories are: ')
        modelSubDirs = os.listdir(modelSavePath)
        for mSDir in modelSubDirs:
            print('\t{}'.format(mSDir))
        yn = str(input('\nDo you want to continue anyway (y/n) '))
        if yn.lower() == 'n':
            saveModelID = str(input('Please specify the new model name: '))
            modelSavePath = checkPath(os.path.join(savePath, saveModelID))
    return modelSavePath


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
with open('rotorIdentificationConfig.json', 'r') as f:
    identificationConfig = json.load(f)

# Extract loggin file information
fileLogDir = ifNullDefault(identificationConfig['logging file']['directory'], os.path.join(os.getcwd(), 'Data'))
fileLogName = identificationConfig['logging file']['filename']
rowIdxs = identificationConfig['logging file']['rows of flights to use (all)']
validationRows = identificationConfig['logging file']['rows of flights for validation']

# Extract filtering file location
filterSavePath = ifNullDefault(identificationConfig['filtered data save directory'], os.path.join(os.getcwd(), 'Data', 'processed', 'filtered'))

# Extract plotting information
plotTrajectories = identificationConfig['plotting']['show trajectories']
plotAnimation = identificationConfig['plotting']['show animation']

# Extract normalization information
normalizeParams = identificationConfig['data normalization']
normalizeData = normalizeParams['normalize data']
usableDataRatio = float(normalizeParams['usable data ratio'])
jointTimeHorizon = None

# Extract partitioning information
doRandomDataPartition = identificationConfig['data partitioning']['random partition']

# Extract model saving parameters 
saveModels = identificationConfig['saving models']['save identified models']
saveModelID = identificationConfig['saving models']['model ID']
modelDir = ifNullDefault(identificationConfig['saving models']["save directory"], os.path.join(os.getcwd(), 'models'))



# Open log_file
log = pd.read_csv(os.path.join(fileLogDir, '{}.csv'.format(fileLogName)), delimiter=',', header=0)

# Offset rowIdx to match python indexing, note that -2 is used since header is also skipped!
rowIdxs = np.array(rowIdxs) - 2
validationRows = np.array(validationRows) - 2

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


# Create save paths and being processing & identification
print('[ INFO ] IDENTIFYING MODELS FOR:')
print('[ INFO ] \t {}'.format(saveModelID))

savePath = checkPath(modelDir, saveModelID)

if not os.path.isdir(savePath):
    os.makedirs(savePath)

# Create/Import prediction interval performance metrics
if os.path.isfile(os.path.join(savePath, 'predictionIntervals.pkl')):
    with open(os.path.join(savePath, 'predictionIntervals.pkl'), 'rb') as f:
        PIresults = pkl.load(f)
        f.close()    
else:
    PIresults = {}


print('[ INFO ] Loading filtered data from: {}'.format(filterSavePath))
filteredDataList = []
referenceFilterParams = None
w_CMD_min = 4000
w_CMD_max = 0
for row in rowIdxs:
    rowIdx = row
    filteredData = pd.read_csv(os.path.join(filterSavePath, '{}-{}'.format(log.loc[rowIdx, 'Onboard Name'], 'FL.csv')))
    w_CMD_min = np.nanmin([np.nanmin(filteredData[['w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD']].to_numpy()), w_CMD_min])
    w_CMD_max = np.nanmax([np.nanmax(filteredData[['w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD']].to_numpy()), w_CMD_max])
    filteredDataList.append(filteredData)
    # Load metadata and check consistency
    with open(os.path.join(filterSavePath, "{}-{}".format(log.loc[row, 'Onboard Name'], 'metadata.json')), 'r') as f:
        filterParams = checkParamConsistency(json.load(f), referenceFilterParams)
    referenceFilterParams = {
        'remove influence of gravity':filterParams['remove influence of gravity'],
        'use noise statistics in drone config':filterParams['use noise statistics in drone config']
        }


DataList = filteredDataList


'''
Split into training and testing subsets
'''
if os.path.isfile(os.path.join(savePath, 'trainAndTestIndices.pkl')):
    with open(os.path.join(savePath, 'trainAndTestIndices.pkl'), 'rb') as f:
        trainingDataIdxs = pkl.load(f)
        f.close()
    idx_train = trainingDataIdxs['Training indices']
    idx_test = trainingDataIdxs['Test indices']
else:
    idx_train = []
    idx_test = []
    if doRandomDataPartition:
        startIdx = 0
        remainder = 0
        for i, _D in enumerate(DataList):
            if jointTimeHorizon is not None:
                remainder = len(_D) % jointTimeHorizon
            [_, _idx_train], [_, _idx_test] = partitioning.PartitionData(_D.to_numpy(), 0.75, Method='Random', batch_size = jointTimeHorizon)
            if rowIdxs[i] not in validationRows:
                idx_train += list(_idx_train + startIdx)
            else:
                _idx_test = np.arange(0, len(_D) - remainder, 1)
            idx_test += list(_idx_test + startIdx)
            startIdx += len(_D)-remainder
            # Trim _D to be compatible with jointTimeHorizon
            DataList[i] = _D.iloc[:len(_D)-remainder, :]

        idx_train = np.array(idx_train)
        idx_test = np.array(idx_test)    
    else:
        startIdx = 0
        remainder = 0
        for i, _D in enumerate(DataList):
            if jointTimeHorizon is not None:
                remainder = len(_D) % jointTimeHorizon
            if remainder != 0:
                print('[ WARNING ] Data (N = {}) could not be evenly split into batches of size {}. Trimming last remaining points (= {} samples)'.format(_D.shape[0], jointTimeHorizon, remainder))
                Data = Data[:-remainder]
                Data = Data.reshape(-1, jointTimeHorizon, *_D.shape[1:])
                Mask = np.arange(0, _D.shape[0]-remainder).reshape(-1, jointTimeHorizon)
            if rowIdxs[i] not in validationRows:
                idx_train += list(np.arange(0, len(_D) - remainder, 1) + startIdx)
                _idx_test = []
            else:
                _idx_test = np.arange(0, len(_D) - remainder, 1)
            idx_test += list(np.array(_idx_test) + startIdx)
            startIdx += len(_D)-remainder
            # Trim _D to be compatible with jointTimeHorizon
            DataList[i] = _D.iloc[:len(_D)-remainder, :]

        idx_train = np.array(idx_train)
        idx_test = np.array(idx_test) 
    # Save train and test indices
    with open(os.path.join(savePath, 'trainAndTestIndices.pkl'), 'wb') as f:
        pkl.dump({'Training indices':idx_train,'Test indices':idx_test}, f)


# Assume actuator dynamics can be represented by:
#               K
# H(s) = --------------
#          tau*s + 1
#
# thus:
# u_t[i+1] = u_t[i] + c * (u_cmd[i] - u_t[i]) 
#   where c = 1 - e^(-tau * dt)

# Build dataset 
# X1 = w_true[i]
# X2 = w_CMD[i+1] - w_true[i]
# Y1 = w_true[i+1]
# -> Use OLS to find: Y1 = X1 + C*X2
# To avoid jumps in w between flights, we need to make approximations per flight

ProcessedData = utility.aggregateData(DataList)

ModelingOBJ = SysID.Model('Stepwise_Regression')
OLS = ModelingOBJ.UtilityFuncs._OLS
dt = ProcessedData['t'][1] - ProcessedData['t'][0]


def wCMD2wTrue(wCMD, wTrue, tau):
    return wTrue + (1 - np.exp(-1*tau*dt))*(wCMD - wTrue)


# Create function to map W_CMD to units of W_true
mappingFactor = interp1d([w_CMD_min, w_CMD_max], [droneParams['idle RPM'], droneParams['max RPM']])
taus = np.zeros((len(DataList), 4))

for n, d in enumerate(DataList):
    w_true = d[['w1', 'w2', 'w3', 'w4']].to_numpy()
    _w_CMD = d[['w1_CMD', 'w2_CMD', 'w3_CMD', 'w4_CMD']].to_numpy()
    w_CMD = _w_CMD.copy()
    for i in range(_w_CMD.shape[1]):
        w_CMD[:, i] = mappingFactor(_w_CMD[:, i])
    
        X1 = w_true[:-1, i]
        X2 = w_CMD[1:, i] - X1
        Y1 = w_true[1:, i]
        Y1_prime = (Y1 - X1).reshape(-1, 1)

        A = np.matrix(X2).reshape(-1, 1)
        c, Y1_prime_est = OLS(A, Y1_prime)

        tau_i = np.log(1/(1 - c.__array__()[0][0]))/dt
        taus[n, i] = tau_i

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(w_true[:, 3], label = 'True')
    # ax.plot(w_CMD[:, 3], label = 'CMD')
    # ax.plot(wCMD2wTrue(w_CMD[1:, 3], w_true[:-1, 3], taus[n, 3]), label = 'Pred')

    # # # fig = plt.figure()
    # # # ax = fig.add_subplot(111)
    # # # ax.plot(Y1_prime)
    # # # ax.plot(Y1_prime_est)
    # plt.show()
    # import code
    # code.interact(local=locals())
        


tau_avg = np.nanmean(taus)
with open(os.path.join(savePath, 'tau.json'), 'w') as f:
    json.dump({'tau_avg':tau_avg, 'taus':{str(FID):float(np.nanmean(tau)) for FID, tau in zip(rowIdxs, taus)}}, f, indent = 4)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(taus)

plt.show()

# plt.plot(1 - np.exp(-1*taus*dt))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(w_true[:, 3], label = 'True')
ax.plot(w_CMD[:, 3], label = 'CMD')
ax.plot(wCMD2wTrue(w_CMD[1:, 3], w_true[:-1, 3], taus[n, 3]), label = 'Pred')
plt.show()

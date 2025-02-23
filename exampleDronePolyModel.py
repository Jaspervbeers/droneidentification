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
import matplotlib.lines as mlines
import json
import sys
from datetime import datetime # Used to tag metadata with information about date of creation.

# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #
from processing import importing, filtering, normalization, quadrotorFM, utility, partitioning
from common import plotter

# This package relies on the system identification pipeline (sysidpipeline). 
with open('relativeImportLocations.json', 'r') as f:
    relativeLocs = json.load(f)

sys.path.append(relativeLocs['sysidpipeline'])
import SysID


# ================================================================================================================================ #
# Classes
# ================================================================================================================================ #

class ModelPredictions:
    
    def __init__(self, Model, Data, TestIndices, TrainIndices, normalizer = None):
        self.testIdxs = TestIndices
        self.trainIdxs = TrainIndices
        self.PolyModel = Model
        prediction, predictionVariance = Model.predict(Data)
        if normalizer is None:
            normalizer = np.ones(len(Data)).reshape(-1)
        self.prediction = np.array(prediction).reshape(-1)*normalizer
        self.predictionVariance = np.array(predictionVariance).reshape(-1)*np.square(normalizer)

    def getSubPrediction(self, fullPrediction, idxs):
        return fullPrediction[idxs]

    @property
    def prediction_Test(self):
        return self.getSubPrediction(self.prediction, self.testIdxs)

    @property
    def prediction_Train(self):
        return self.getSubPrediction(self.prediction, self.trainIdxs)

    @property
    def predictionVariance_Test(self):
        return self.getSubPrediction(self.predictionVariance, self.testIdxs)

    @property
    def predictionVariance_Train(self):
        return self.getSubPrediction(self.predictionVariance, self.trainIdxs)



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


def printPIs(PIDict):
    for force in PIDict.keys():
        print('[ INFO ] Polynomial {} - Prediction Interval results (Confidence level = {})'.format(force, PIConfidenceLevel*100))
        print('[ INFO ] {:<30} {:>6}'.format('Probability Coverage:', PIDict[force]['PICP']))
        print('[ INFO ] {:<30} {:>6}'.format('Normalized mean PI width:', PIDict[force]['MPIW']))
    return None


def makePolyModel(Data, TargetID, polys, fixed, partitionIdxs, normalizer, cap = 15):
    train_idx = partitionIdxs[0]
    test_idx = partitionIdxs[1]

    Model_Poly = SysID.Model('Stepwise_Regression')
    Model_Poly.compile(Data.loc[train_idx, :].copy(), polys, fixed, includeBias=True)
    Model_Poly.train(Data.loc[train_idx, :].copy(), Data.loc[train_idx, TargetID].copy(), stop_criteria='PSE', k_lim =cap, force_k_lim = True)
    PredictionData = ModelPredictions(Model_Poly, Data, test_idx, train_idx, normalizer = normalizer)

    print('[ INFO ] Model performance:')
    print('[ INFO ] \t {:<25}   {:>20}'.format('RMSE w.r.t training data', Model_Poly._RMSE( Data.loc[train_idx, TargetID].to_numpy(), PredictionData.prediction_Train)))
    print('[ INFO ] \t {:<25}   {:>20}'.format('RMSE w.r.t test data', Model_Poly._RMSE(Data.loc[test_idx, TargetID].to_numpy(), PredictionData.prediction_Test)))
    print('[ INFO ] \t {:<25}   {:>20}'.format('RMSE w.r.t full data', Model_Poly._RMSE(Data[TargetID].to_numpy(), PredictionData.prediction)))

    return Model_Poly, PredictionData


def identifyPolyModel(modelID, modelFolder, metadata, data, targetColumn, polynominalCandidates, fixedRegressors, trainingIndices, testingIndices, excitationIndices, PIConfidenceLevel, predictionIntervalDict, normalizer = None, regressorCap = 15, DEBUG_FLAG = False, SAVE_INDICES = False):
    # Isolate excitations
    if len(excitationIndices):
        trainIdx = np.intersect1d(trainingIndices, excitationIndices)
    else:
        trainIdx = trainingIndices
    output = {}
    # Identify model
    model, predictions = makePolyModel(data, targetColumn, polynominalCandidates, fixedRegressors, [trainIdx, testingIndices], normalizer = normalizer, cap = regressorCap)
    targets = data[targetColumn].to_numpy().reshape(-1)
    if normalizer is not None:
        targets = targets * normalizer
    # Get prediction interval metrics 
    PICP, MPIW = utility.qualityPI(targets, predictions.prediction, predictions.predictionVariance, conf=PIConfidenceLevel)
    predictionIntervalDict.update({targetColumn:{'PICP':PICP, 'MPIW':MPIW}})
    output.update({'prediction interval metrics':predictionIntervalDict})
    # Plot predictions
    fig = plotter.plotModelWithPI(targets, (predictions.prediction,), (predictions.predictionVariance,), confidence = PIConfidenceLevel, returnFig=True)
    output.update({'fig':fig})
    # Save model
    saveModelDir = os.path.join(modelFolder, modelID, targetColumn)
    if not os.path.isdir(saveModelDir):
        os.makedirs(saveModelDir)
    with open(os.path.join(saveModelDir, 'processingMetadata.json'), 'w') as f:
        json.dump(metadata, f)
    model.save(saveModelDir)
    if SAVE_INDICES:
        with open(os.path.join(saveModelDir, 'IDX.pkl'), 'wb') as f:
            pkl.dump({'train':trainIdx, 'test':testingIndices}, f)
    if DEBUG_FLAG:
        output.update({'model':model})
        output.update({'predictions':predictions})
    return output

# ================================================================================================================================ #
# Processing
# ================================================================================================================================ #
'''
Extract & Define processing parameters
'''
# TODO: Save this config file as metadata
with open('identificationConfig-example.json', 'r') as f:
    identificationConfig = json.load(f)

# Extract loggin file information
fileLogDir = ifNullDefault(identificationConfig['logging file']['directory'], os.path.join(os.getcwd(), 'Data'))
fileLogName = identificationConfig['logging file']['filename']
rowIdxs = identificationConfig['logging file']['rows of flights to use (all)']
validationRows = identificationConfig['logging file']['rows of flights for validation']

# Extract data importing information
importParams = identificationConfig['data importing']
importRawData = importParams['import raw data']
resampleRate = float(importParams['resampling rate'])
filterOutliersOT = importParams['filter optitrack outliers']
saveRawData = importParams['save raw data']
importedDataSavePath = ifNullDefault(importParams['imported data save directory'], os.path.join(os.getcwd(), 'Data', 'processed', 'imported'))
alignUsing = importParams['align with optitrack using']
maxLag = float(importParams['max permitted lag for optitrack'])

# Extract filtering information
filterParams = identificationConfig['data filtering']
useConfigNoiseStats = filterParams['use noise statistics in drone config']
filterData = filterParams['run extended kalman filter']
saveEKFResults = filterParams['save kalman filter convergence results']
removeGComponent = filterParams['remove influence of gravity']
saveFilteredData = filterParams['save filtered data']
filterSavePath = ifNullDefault(filterParams['filtered data save directory'], os.path.join(os.getcwd(), 'Data', 'processed', 'filtered'))

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

# Extract excitation information
isolateExcitations = identificationConfig['manoeuvre excitations']['isolate to regions of excitation']
excitationThreshold = float(identificationConfig['manoeuvre excitations']['excitation threshold'])

# Extract model identification parameters
identificationParams = identificationConfig['identification parameters']
PIConfidenceLevel = float(identificationParams['prediction interval confidence level'])
polyModelCap = int(identificationParams['polynomial']['regressor cap'])
identifyFx = identificationParams['identify fx']
identifyFy = identificationParams['identify fy']
identifyFz = identificationParams['identify fz']
identifyMx = identificationParams['identify mx']
identifyMy = identificationParams['identify my']
identifyMz = identificationParams['identify mz']


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
# Backend for importing OptiTrack data. Choices are 'python' or 'matlab'. Decides which scripts should
# be used to import OptiTrack data. Python are located in Utility > processingCyberZoo, Matlab are Sihao's scripts. 
OT_importBackend = 'python' 



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


'''
Import raw data
'''
if importRawData:
    # rawDataList = []
    rawDataDict = {}
    for row in rowIdxs:
        print('[ INFO ] Importing rowIdx: {}'.format(row +2))
        rawData = importing.runImport(rowIdx = row, log = log, OB_samplingRate = resampleRate, maxLag_seconds = maxLag, alignUsing = alignUsing, filterOutliersOT=filterOutliersOT, velocityCutoffHz = 10, importBackend_OB = 'V1_import_OB_btfl')
        if saveRawData:
            print('[ INFO ] Saving imported data to {}'.format(importedDataSavePath))
            _filename = "{}-{}".format(log.loc[row, 'Onboard Name'], 'IM.csv') #IM suffix to indicate imported data
            rawData.to_csv(os.path.join(importedDataSavePath, _filename))
            # Save importing parameters to metadata file
            now = datetime.today().isoformat()
            importParams.update({'Datetime (creation)':now})
            importParams.update({'log file':fileLogName})
            importParams.update({'rows of flights to use (all)':(rowIdxs + 2).tolist()})
            if len(validationRows):
                importParams.update({'rows of flights for validation':(validationRows + 2).tolist()})
            else:
                importParams.update({'rows of flights for validation':[]})
            importParams.update({'current row':str(row + 2)})
            with open(os.path.join(importedDataSavePath, "{}-{}".format(log.loc[row, 'Onboard Name'], 'metadata.json')), 'w') as f:
                json.dump(importParams, f)
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
            print('[ INFO ] Running EKF...')
            filteredData, filtResults = filtering.runFilter(rowIdx, rawData, droneMass, droneParams, log, removeGravityComponent=removeGComponent, defaultNoise=(not useConfigNoiseStats))
            # Calculate induced velocity
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
                if len(validationRows):
                    filterParams.update({'rows of flights for validation':(validationRows + 2).tolist()})
                else:
                    filterParams.update({'rows of flights for validation':[]})
                filterParams.update({'current row':str(row + 2)})
                with open(os.path.join(filterSavePath, '{}-{}'.format(log.loc[rowIdx, 'Onboard Name'], 'metadata.json')), 'w') as f:
                    json.dump(filterParams, f)
else:
    print('[ INFO ] Loading filtered data from: {}'.format(filterSavePath))
    filteredDataList = []
    referenceFilterParams = None  
    for row in rowIdxs:
        rowIdx = row
        filteredData = pd.read_csv(os.path.join(filterSavePath, '{}-{}'.format(log.loc[rowIdx, 'Onboard Name'], 'FL.csv')))
        filteredDataList.append(filteredData)
        # Load metadata and check consistency
        with open(os.path.join(filterSavePath, "{}-{}".format(log.loc[row, 'Onboard Name'], 'metadata.json')), 'r') as f:
            filterParams = checkParamConsistency(json.load(f), referenceFilterParams)
        referenceFilterParams = {'remove influence of gravity':filterParams['remove influence of gravity']}



'''
Plot trajectories
'''
if plotTrajectories:
    c1 = 'orangered'
    c2 = 'gold'
    c3 = 'firebrick'
    trajFigs3D = []
    trajFigs = []
    animFigs = []
    for row, filteredData in enumerate(filteredDataList):
        # Position-time, velocity-time and acceleration-time plots across all three axes
        pva_fig = plotter.plotPosVelAccTime(filteredData['t'], filteredData['x'], filteredData['u'], filteredData['ax'], 
                                            colors=(c1, c1, c1))
        pva_fig = plotter.plotPosVelAccTime(filteredData['t'], filteredData['y'], filteredData['v'], filteredData['ay'], 
                                            colors=(c2, c2, c2), parentFig = pva_fig)
        pva_fig = plotter.plotPosVelAccTime(filteredData['t'], filteredData['z'], filteredData['w'], filteredData['az'], 
                                            colors=(c3, c3, c3), parentFig = pva_fig)
        _x_lines = mlines.Line2D([], [], color=c1)
        _y_lines = mlines.Line2D([], [], color=c2)
        _z_lines = mlines.Line2D([], [], color=c3)
        handles = [_x_lines, _y_lines, _z_lines]
        ax1 = pva_fig.axes[0]
        ax1.legend(handles = handles, labels = ['x', 'y', 'z'])
        ax2 = pva_fig.axes[1]
        ax2.legend(handles = handles, labels = ['u', 'v', 'w'])
        ax3 = pva_fig.axes[2]
        ax3.legend(handles = handles, labels = ['ax', 'ay', 'az'])
        pva_fig.suptitle('{} - Flight ID: {}'.format(log.loc[rowIdxs[row], 'Quadrotor'], log.loc[rowIdxs[row], 'Flight ID']))
        trajFigs.append(pva_fig)

        # 3D trajectory plots with time encoded in color
        fig3D = plotter.Trajectory3D(filteredData['t'], filteredData['x'], filteredData['y'], -1*filteredData['z'], 
                                    Gradient = True, returnFig=True, n_skip=10)
        trajFigs3D.append(fig3D)


        if plotAnimation:
            axisParams = {
                    'x_label':r'$\mathbf{x} \quad [m]$',
                    'y_label':r'$\mathbf{y} \quad [m]$',
                    'z_label':r'$\mathbf{z} \quad [m]$',
                    }
            plotParams = {
                    'color':[c1],
                    'alpha':[0.5]
                    }
            anim = plotter.livePlot3D(filteredData['x'], filteredData['y'], -1*filteredData['z'], 
                                    plotParams=plotParams, axisParams=axisParams, showProjection = True, 
                                    d_frame=1, interval = 1, trail = 100, trailDecay = 20000, useBlit=True)
            
            anim._fig.suptitle('{} - Flight ID: {}'.format(log.loc[rowIdxs[row], 'Quadrotor'], log.loc[rowIdxs[row], 'Flight ID']))
            animFigs.append(anim)

    plt.show()



'''
Normalize data, if chosen
'''
if normalizeData:
    DataList = []
    for filteredData in filteredDataList:
        NormalizedData = normalization.normalizeQuadData(filteredData, droneParams, rotorConfig, r_sign, rotorDir, N_rot = 4, minRPM=minRPM[quadrotorName])
        DataList.append(NormalizedData)
else:
    DataList = quadrotorFM.addControlMoments(filteredDataList, rotorConfig, r_sign, rotorDir)



'''
Add extra columns, useful for identification, to DataFrames and combine into a single DataFrame
'''
DataList = normalization.addExtraColsAndTrim(DataList, usableDataRatio)



'''
Isolate system identification manoeuvres
'''
isolationMethods = {'Fx':"Variance", 'Fy':"Variance", 'Fz':"Variance",
                    'Mx':"Peak", 'My':"Peak", 'Mz':"Peak"}
if os.path.isfile(os.path.join(savePath, 'excitationIdxs.pkl')):
    with open(os.path.join(savePath, 'excitationIdxs.pkl'), 'rb') as f:
        excitationIdxs = pkl.load(f)
        f.close()
else:
    excitationIdxs = {'Fx':[], 'Fy':[], 'Fz':[], 'Mx':[], 'My':[], 'Mz':[]}
if isolateExcitations:
    print('[ INFO ] Isolating system excitations')
    _excitationIdxs = partitioning.getSystemExcitations(excitationIdxs, DataList, locals(), isolationMethods,
                                            variance_threshold=excitationThreshold, height_threshold = 0.5*excitationThreshold, 
                                            prominence_threshold = 0.9*excitationThreshold)
    excitationIdxs.update({k:v for k, v in _excitationIdxs.items()})



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
            [_, _idx_train], [_, _idx_test] = partitioning.PartitionData(_D.to_numpy(), 0.8, Method='Random', batch_size = jointTimeHorizon)
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


ProcessedData = utility.aggregateData(DataList)


if len(validationRows):
    validationIdxs, segregatedVIdxs = utility.findAggregatedIdxs(validationRows, rowIdxs, DataList)


# Set to true to keep record of all models in global scope. Is more memory intensive. 
DEBUG_FLAG = False
# Set to true to save (model specific) training and testing indices
SAVE_INDICES = True


# Create metadata file
metadata = identificationConfig.copy()
metadata.pop('data importing')
metadata.update({'data filtering':filterParams})
metadata.update({'identification parameters':identificationParams})

if identifyFx:
    print('[ INFO ] Identifying polynomial model for Fx...')
    polys_Fx = [
                {
                    'vars':['u', '|v|', 'w', 'v_in'],
                    'degree':4,
                    'sets':[1, 'w_tot', 'q', '|r|', 'u_q', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[yaw]', 'cos[yaw]']
                },
                {
                    'vars':['mu_x', '|mu_y|', 'mu_z', 'mu_vin'],
                    'degree':4,
                    'sets':[1, 'w_tot', 'q', '|r|', 'u_q', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[yaw]', 'cos[yaw]']
                },
                {
                    'vars':['q', '|r|'],
                    'degree':4,
                    'sets':[1, 'w_tot', 'v_in', 'mu_vin']
                },
                {
                    'vars':['u_q', '|u_p|', '|u_r|'],
                    'degree':4,
                    'sets':[1, 'sin[pitch]', 'cos[pitch]', 'w_tot']
                },
                {
                    'vars':['v_in', 'mu_vin', 'w_tot'],
                    'degree':4,
                    'sets':[1, 'sin[pitch]', 'cos[pitch]', 'sin[yaw]', 'cos[yaw]']
                }
            ]
    fixed_Fx = ['u']
    FxOut = identifyPolyModel(saveModelID, modelDir, metadata, ProcessedData, 'Fx', polys_Fx, fixed_Fx, 
                              idx_train, idx_test, excitationIdxs['Fx'], PIConfidenceLevel, PIresults,
                              normalizer = ProcessedData['F_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, 
                              DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)
    PIresults = FxOut['prediction interval metrics']
    figFx = FxOut['fig']
    ax = figFx.axes[0]
    ax.set_ylabel(r'$\mathbf{Force, \quad F_{x}}\quad [N]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)



if identifyFy:
    print('[ INFO ] Identifying polynomial model for Fy...')
    polys_Fy = [
            {
                'vars':['|u|', 'v', 'w', 'v_in'],
                'degree':4,
                'sets':[1, 'w_tot', 'p', '|r|', 'u_p', '|u_r|', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['|mu_x|', 'mu_y', 'mu_z', 'mu_vin'],
                'degree':4,
                'sets':[1, 'w_tot', 'p', '|r|', 'u_p', '|u_r|', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['p', '|r|'],
                'degree':4,
                'sets':[1, 'w_tot', 'v_in', 'mu_vin']
            },
            {
                'vars':['|u_q|', 'u_p', '|u_r|'],
                'degree':4,
                'sets':[1, 'sin[roll]', 'cos[roll]', 'w_tot']
            },
            {
                'vars':['v_in', 'mu_vin', 'w_tot'],
                'degree':4,
                'sets':[1, 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            }
        ]
    fixed_Fy = ['v']
    FyOut = identifyPolyModel(saveModelID, modelDir, metadata, ProcessedData, 'Fy', polys_Fy, fixed_Fy, 
                              idx_train, idx_test, excitationIdxs['Fy'], PIConfidenceLevel, PIresults,
                              normalizer = ProcessedData['F_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, 
                              DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)
    PIresults = FyOut['prediction interval metrics']                              
    figFy = FyOut['fig']
    ax = figFy.axes[0]
    ax.set_ylabel(r'$\mathbf{Force, \quad F_{y}}\quad [N]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)    



if identifyFz:
    print('[ INFO ] Identifying polynomial model for Fz...')
    polys_Fz = [
            {
                'vars':['|u|', '|v|', 'w', 'v_in'],
                'degree':4,
                'sets':[1, 'w_tot', '|p|', '|q|', '|r|', '|u_p|', '|u_q|', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['|mu_x|', '|mu_y|', 'mu_z', 'mu_vin'],
                'degree':4,
                'sets':[1, 'w_tot', '|p|', '|q|', '|r|', '|u_p|', '|u_q|', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['|u_q|', '|u_p|', '|u_r|'],
                'degree':4,
                'sets':[1, 'sin[yaw]', 'cos[yaw]', 'w_tot']
            },
            {
                'vars':['v_in', 'mu_vin', 'w_tot'],
                'degree':4,
                'sets':[1, 'sin[pitch]', 'cos[pitch]', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['|p|', '|q|', '|r|'],
                'degree':4,
                'sets':[1, 'w_tot', 'v_in', 'mu_vin']
            },
        ]   
    fixed_Fz = ['(w_tot)^(2)', '(u^2 + v^2)', '(v_in - w)^2', 'w']
    FzOut = identifyPolyModel(saveModelID, modelDir, metadata, ProcessedData, 'Fz', polys_Fz, fixed_Fz, 
                              idx_train, idx_test, excitationIdxs['Fz'], PIConfidenceLevel, PIresults,
                              normalizer = ProcessedData['F_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, 
                              DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)
    PIresults = FzOut['prediction interval metrics']
    figFz = FzOut['fig']
    ax = figFz.axes[0]
    ax.set_ylabel(r'$\mathbf{Force, \quad F_{z}}\quad [N]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)    



if identifyMx:
    print('[ INFO ] Identifying polynomial model for Mx...')
    polys_Mx = [
        {
            'vars':['|u|', 'v', 'w', 'v_in'],
            'degree':4,
            'sets':[1, 'w_tot', 'p', '|r|', 'u_p', '|u_r|', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
        },
        {
            'vars':['|mu_x|', 'mu_y', 'mu_z', 'mu_vin'],
            'degree':4,
            'sets':[1, 'w_tot', 'p', '|r|', 'u_p', '|u_r|', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
        },
        {
            'vars':['p', '|q|', '|r|'],
            'degree':4,
            'sets':[1, 'w_tot', 'v_in', 'mu_vin']
        },
        {
            'vars':['u_p', '|u_q|', '|u_r|'],
            'degree':4,
            'sets':[1, 'sin[roll]', 'cos[roll]', 'w_tot']
        },
        {
            'vars':['v_in', 'mu_vin', 'w_tot'],
            'degree':4,
            'sets':[1, 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
        }
    ]
    fixed_Mx = ['p', 'u_p']
    MxOut = identifyPolyModel(saveModelID, modelDir, metadata, ProcessedData, 'Mx', polys_Mx, fixed_Mx, 
                              idx_train, idx_test, excitationIdxs['Mx'], PIConfidenceLevel, PIresults,
                              normalizer = ProcessedData['M_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, 
                              DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)
    PIresults = MxOut['prediction interval metrics']
    figMx = MxOut['fig']
    ax = figMx.axes[0]
    ax.set_ylabel(r'$\mathbf{Moment, \quad M_{x}}\quad [Nm]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)



if identifyMy:
    print('[ INFO ] Identifying polynomial model for My...')
    polys_My = [
            {
                'vars':['u', '|v|', 'w', 'v_in'],
                'degree':4,
                'sets':[1, 'w_tot', 'q', '|r|', 'u_q', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['mu_x', '|mu_y|', 'mu_z', 'mu_vin'],
                'degree':4,
                'sets':[1, 'w_tot', 'q', '|r|', 'u_q', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['|p|', 'q', '|r|'],
                'degree':4,
                'sets':[1, 'w_tot', 'v_in', 'mu_vin']
            },
            {
                'vars':['u_q', '|u_p|', '|u_r|'],
                'degree':4,
                'sets':[1, 'sin[pitch]', 'cos[pitch]', 'w_tot']
            },
            {
                'vars':['v_in', 'mu_vin', 'w_tot'],
                'degree':4,
                'sets':[1, 'sin[pitch]', 'cos[pitch]', 'sin[yaw]', 'cos[yaw]']
            }
        ]
    fixed_My = ['q', 'u_q']
    MyOut = identifyPolyModel(saveModelID, modelDir, metadata, ProcessedData, 'My', polys_My, fixed_My, 
                              idx_train, idx_test, excitationIdxs['My'], PIConfidenceLevel, PIresults,
                              normalizer = ProcessedData['M_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, 
                              DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)
    PIresults = MyOut['prediction interval metrics']
    figMy = MyOut['fig']
    ax = figMy.axes[0]
    ax.set_ylabel(r'$\mathbf{Moment, \quad M_{y}}\quad [Nm]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)



if identifyMz:
    print('[ INFO ] Identifying polynomial model for Mz...')
    polys_Mz = [
            {
                'vars':['|u|', '|v|', 'w', 'v_in'],
                'degree':4,
                'sets':[1, 'w_tot', '|p|', '|q|', '|r|', 'u_p', '|u_q|', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['|mu_x|', '|mu_y|', 'mu_z', 'mu_vin'],
                'degree':4,
                'sets':[1, 'w_tot', '|p|', '|q|', '|r|', 'u_p', '|u_q|', '|u_r|', 'sin[pitch]', 'cos[pitch]', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            },
            {
                'vars':['|p|', '|q|', 'r'],
                'degree':4,
                'sets':[1, 'w_tot', 'v_in', 'mu_vin']
            },
            {
                'vars':['|u_q|', '|u_p|', 'u_r'],
                'degree':4,
                'sets':[1, 'sin[yaw]', 'cos[yaw]', 'w_tot']
            },
            {
                'vars':['v_in', 'mu_vin', 'w_tot'],
                'degree':4,
                'sets':[1, 'sin[pitch]', 'cos[pitch]', 'sin[roll]', 'cos[roll]', 'sin[yaw]', 'cos[yaw]']
            }
        ]   
    fixed_Mz = ['r', 'u_r']    
    MzOut = identifyPolyModel(saveModelID, modelDir, metadata, ProcessedData, 'Mz', polys_Mz, fixed_Mz, 
                              idx_train, idx_test, excitationIdxs['Mz'], PIConfidenceLevel, PIresults,
                              normalizer = ProcessedData['M_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, 
                              DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)  
    PIresults = MzOut['prediction interval metrics']                                
    figMz = MzOut['fig']
    ax = figMz.axes[0]
    ax.set_ylabel(r'$\mathbf{Moment, \quad M_{z}}\quad [Nm]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)    



printPIs(PIresults)
plt.show()

with open(os.path.join(savePath, 'PIresults.pkl'), 'wb') as f:
    pkl.dump(PIresults, f)
    f.close()  

# End
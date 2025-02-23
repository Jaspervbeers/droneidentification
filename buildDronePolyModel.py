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

# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #
from processing import normalization, partitioning, utility, quadrotorFM
from common import plotter, angleFuncs, solvers

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


def addVarToPolys(polys, vars, toAdd):
    # Find vars
    for i, p in enumerate(polys):
        if p['vars'] == vars:
            p.update({'vars':vars + toAdd})
            break
        if i == len(polys)-1:
            raise ValueError(f'Could not find {vars} in polys.')
    return None


def printPIs(PIDict):
    for force in PIDict.keys():
        print('[ INFO ] Polynomial {} - Prediction Interval results (Confidence level = {})'.format(force, PIConfidenceLevel*100))
        print('[ INFO ] {:<30} {:>6}'.format('Probability Coverage:', PIDict[force]['PICP']))
        print('[ INFO ] {:<30} {:>6}'.format('Normalized mean PI width:', PIDict[force]['MPIW']))
        thresh = 0.05
        if PIDict[force]['PICP'] <= (PIConfidenceLevel - thresh)*100:
            print(f'[ WARNING ] \t Probability Coverage for {force} <= {np.around(PIConfidenceLevel-thresh, 1)*100}, model may be overfitting or validation data contains unobserved dynamics.')
            print('[ INFO ] \t {:<30} {:>6}'.format('Probability Coverage (Train):', globals()[f'{force}Out']['_PI_breakdown']['PICP_train']))
            print('[ INFO ] \t {:<30} {:>6}'.format('Probability Coverage (Test):', globals()[f'{force}Out']['_PI_breakdown']['PICP_test']))
            if globals()[f'{force}Out']['_PI_breakdown']['PICP_test'] <= globals()[f'{force}Out']['_PI_breakdown']['PICP_train'] - thresh*100:
                print('[ WARNING ] \t Given difference in probability coverage between training and test data, model overfits or test data ill-constructed')
            else:
                print('[ WARNING ] \t Training and test probability coverage do not suggest an issue, issues likely lie with validation data')
    return None


def makePolyModel(Data, TargetID, polys, fixed, partitionIdxs, normalizer, cap = 15, bias = True):
    train_idx = partitionIdxs[0]
    test_idx = partitionIdxs[1]

    Model_Poly = SysID.Model('Stepwise_Regression')
    Model_Poly.compile(Data.loc[train_idx, :].copy(), polys, fixed, includeBias=bias)
    Model_Poly.train(Data.loc[train_idx, :].copy(), Data.loc[train_idx, TargetID].copy(), stop_criteria='PSE', k_lim =cap, force_k_lim = True)
    PredictionData = ModelPredictions(Model_Poly, Data, test_idx, train_idx, normalizer = normalizer)

    print('[ INFO ] Model performance:')
    print('[ INFO ] \t {:<25}   {:>20}'.format('RMSE w.r.t training data', Model_Poly._RMSE( Data.loc[train_idx, TargetID].to_numpy(), PredictionData.prediction_Train)))
    print('[ INFO ] \t {:<25}   {:>20}'.format('RMSE w.r.t test data', Model_Poly._RMSE(Data.loc[test_idx, TargetID].to_numpy(), PredictionData.prediction_Test)))
    print('[ INFO ] \t {:<25}   {:>20}'.format('RMSE w.r.t full data', Model_Poly._RMSE(Data[TargetID].to_numpy(), PredictionData.prediction)))

    return Model_Poly, PredictionData


def identifyPolyModel(data, targetColumn, polynominalCandidates, fixedRegressors, trainingIndices, testingIndices, excitationIndices, normalizer = None, regressorCap = 15, bias = True):
    # Isolate excitations
    if len(excitationIndices):
        trainIdx = np.intersect1d(trainingIndices, excitationIndices)
    else:
        trainIdx = trainingIndices

    # Identify model
    model, predictions = makePolyModel(data, targetColumn, polynominalCandidates, fixedRegressors, [trainIdx, testingIndices], normalizer = normalizer, cap = regressorCap, bias = bias)
    return model, predictions


def savePolyModel(modelID, modelFolder, metadata, model, predictions, data, targetColumn, trainingIndices, testingIndices, segregatedIdxs, segregatedVIdxs, excitationIndices, PIConfidenceLevel, predictionIntervalDict, normalizer = None, DEBUG_FLAG = False, SAVE_INDICES = False):
    # Isolate excitations
    if len(excitationIndices):
        trainIdx = np.intersect1d(trainingIndices, excitationIndices)
    else:
        trainIdx = trainingIndices
    
    targets = data[targetColumn].to_numpy().reshape(-1)
    if normalizer is not None:
        targets = targets * normalizer

    output = {}

    # Check for over-fitting between training and test rmse
    RMSE_train = model._RMSE( data.loc[trainIdx, targetColumn].to_numpy(), predictions.prediction_Train)
    RMSE_test = model._RMSE( data.loc[testingIndices, targetColumn].to_numpy(), predictions.prediction_Test)
    RMSEdiff_train = 100 - RMSE_train/RMSE_test*100
    print(f'[ INFO ] Relative difference in RMSE between TEST and TRAIN ({targetColumn}): {RMSEdiff_train} %')
    # NOTE: Not abs(RMSE) here since we care more about situations where TEST > TRAIN. While -RMSEdiff can be a problem, it is highly subject to differences in magnitude between the data sets
    if RMSEdiff_train > 10:
        print(f'[ WARNING ] The TEST RMSE is over 10% different than the TRAIN RMSE. The {targetColumn} model may be OVERFITTING. Consider removing regressors.')
        validationIdxs = None
        if len(segregatedVIdxs):
            validationIdxs = []
            # Highlight validation plots in figures
            for sVIdx in segregatedVIdxs:
                validationIdxs = validationIdxs + list(np.arange(sVIdx[0], sVIdx[1]))
        figValid = plotter.ValidationRMSEExplorer(data, model, targetColumn, predictions, validationIdxs, trainIdx, testingIndices)
        output.update({'figValidCheck':figValid})

    # Get prediction interval metrics 
    PICP, MPIW = utility.qualityPI(targets, predictions.prediction, predictions.predictionVariance, conf=PIConfidenceLevel)
    PICP_tr, MPIW_tr = utility.qualityPI(targets[trainIdx], predictions.prediction_Train, predictions.predictionVariance_Train, conf=PIConfidenceLevel)
    PICP_te, MPIW_tr = utility.qualityPI(targets[testingIndices], predictions.prediction_Test, predictions.predictionVariance_Test, conf=PIConfidenceLevel)
    predictionIntervalDict.update({targetColumn:{'PICP':PICP, 'MPIW':MPIW}})
    output.update({'prediction interval metrics':predictionIntervalDict})
    output.update({'_PI_breakdown':{'PICP_train':PICP_tr, 'PICP_test':PICP_te}})
    # Plot predictions
    fig = plotter.plotModelWithPI(targets, (predictions.prediction,), (predictions.predictionVariance,), labels = (targetColumn,), 
                                  ylabel = r'$\mathbf{Force}$, N' if targetColumn.startswith('F') else r'$\mathbf{Moment}$, Nm', 
                                  xlabel = r'$\mathbf{Sample}$, -', confidence = PIConfidenceLevel, returnFig=True)

    # Segment plots based on end of flights 
    ax = fig.axes[0]
    handles, _ = ax.get_legend_handles_labels()
    for sIdx in segregatedIdxs:
        plotter.addVLINE(ax, sIdx, -1000, 1000, color = '#e67d0a')
    plotter.addLegendLine(handles, label = 'End of flight', color = '#e67d0a')

    if len(segregatedVIdxs):
        validationIdxs = []
        # Highlight validation plots in figures
        for sVIdx in segregatedVIdxs:
            plotter.addXVSPAN(ax, sVIdx[0], sVIdx[1], color = 'orange', alpha = 0.2)
            validationIdxs = validationIdxs + list(np.arange(sVIdx[0], sVIdx[1]))
        plotter.addLegendPatch(handles, label = 'Validation flight', color = 'orange', alpha = 0.2)

        # Check for over-fitting through test vs validation rmse
        validationRMSE = model._RMSE(data[targetColumn].to_numpy()[validationIdxs], predictions.prediction[validationIdxs])
        print('[ INFO ] \t {:<25}   {:>20}'.format('RMSE w.r.t VALIDATION data', validationRMSE))
        RMSEdiff = 100 - validationRMSE/RMSE_test*100
        print(f'[ INFO ] Relative difference in RMSE between TEST and VALIDATION ({targetColumn}): {RMSEdiff} %')
        # print(f'[ INFO ] The {targetColumn} model TEST RMSE {"underestimates" if RMSEdiff < 0 else "overestimates"} the VALIDATION RMSE by {np.abs(RMSEdiff)} %')
        # NOTE: Not abs(RMSE) here since we care more about situations where VALID > TEST. While +RMSEdiff can be a problem, it is highly subject to differences in magnitude between the data sets
        if RMSEdiff < -10:
            print(f'[ WARNING ] The TEST RMSE is over 10% different than the VALIDATION RMSE. The {targetColumn} model may be OVERFITTING. Consider removing regressors.')
            # Do not plot if plot already created in train vs test check! 
            if 'figValidCheck' not in output.keys():
                figValid = plotter.ValidationRMSEExplorer(data, model, targetColumn, predictions, validationIdxs, trainIdx, testingIndices)
                output.update({'figValidCheck':figValid})

    ax.legend(handles=handles)
    output.update({'fig':fig})

    # Save model
    saveModelDir = os.path.join(modelFolder, modelID, targetColumn)
    if not os.path.isdir(saveModelDir):
        os.makedirs(saveModelDir)
    with open(os.path.join(saveModelDir, 'processingMetadata.json'), 'w') as f:
        json.dump(metadata, f, indent = 4)
    model.save(saveModelDir, saveTrainingData = True)

    if SAVE_INDICES:
        with open(os.path.join(saveModelDir, 'IDX.pkl'), 'wb') as f:
            pkl.dump({'train':trainIdx, 'test':testingIndices}, f)
    if DEBUG_FLAG:
        output.update({'model':model})
        output.update({'predictions':predictions})

    # Save figures 
    print('[ INFO ] Saving figures...')
    fig.savefig(os.path.join(saveModelDir, f'figure_{modelID}_{targetColumn}.pdf'))
    fig.savefig(os.path.join(saveModelDir, f'figure_{modelID}_{targetColumn}.png'), dpi = 600)
    if 'figValidCheck' in output.keys():
        figValid.savefig(os.path.join(saveModelDir, f'validationCheck_{modelID}_{targetColumn}.pdf'))
        figValid.savefig(os.path.join(saveModelDir, f'validationCheck_{modelID}_{targetColumn}.png'), dpi = 600)
    return output

# ================================================================================================================================ #
# Processing
# ================================================================================================================================ #
'''
Extract & Define processing parameters
'''
# TODO: Save this config file as metadata
with open('identificationConfig.json', 'r') as f:
    identificationConfig = json.load(f)

# Extract loggin file information
fileLogDir = ifNullDefault(identificationConfig['logging file']['directory'], os.path.join(os.getcwd(), 'Data'))
fileLogName = identificationConfig['logging file']['filename']
rowIdxs = utility._parseRows(identificationConfig['logging file']['rows of flights to use (all)'])
validationRows = identificationConfig['logging file']['rows of flights for validation']

# Extract filtering file location
filterSavePath = ifNullDefault(identificationConfig['filtered data save directory'], os.path.join(os.getcwd(), 'Data', 'processed', 'filtered'))

# Extract plotting information
plotTrajectories = identificationConfig['plotting']['show trajectories']
NShots = identificationConfig['plotting']['number of drone stills']
plotAnimation = identificationConfig['plotting']['show animation']
animationSpeedUpFactor = identificationConfig['plotting']['animation speed up factor']

# Extract normalization information
normalizeParams = identificationConfig['data normalization']
normalizeData = normalizeParams['normalize data']
usableDataRatio = float(identificationConfig['data partitioning']['usable data ratio'])
jointTimeHorizon = None

# Extract partitioning information
doRandomDataPartition = identificationConfig['data partitioning']['random partition']
partitionRatio = identificationConfig['data partitioning']['partition ratio']

# Extract excitation information
isolateExcitations = identificationConfig['manoeuvre excitations']['isolate to regions of excitation']
showExcitations = identificationConfig['manoeuvre excitations']['show isolation results']
excitationThreshold = float(identificationConfig['manoeuvre excitations']['excitation threshold'])
excitationSpread = float(identificationConfig['manoeuvre excitations']['spread'])

# Extract model identification parameters
identificationParams = identificationConfig['identification parameters']
PIConfidenceLevel = float(identificationParams['prediction interval confidence level'])
polyModelCap = int(identificationParams['polynomial']['regressor cap'])
polyFile = identificationParams['polynomial specification file']
if polyFile is None:
    raise ValueError('buildDronePolyModel.py requires you to specify a polynomial candidate (json) file.')
else:
    print(f'[ INFO ] Loading polynomial candidate file: {polyFile}')
with open(polyFile, 'r') as f:
    polyData = json.load(f)
identifyFx = identificationParams['identify fx']
addBiasFx = identificationParams['add bias term fx']
identifyFy = identificationParams['identify fy']
addBiasFy = identificationParams['add bias term fy']
identifyFz = identificationParams['identify fz']
addBiasFz = identificationParams['add bias term fz']
identifyMx = identificationParams['identify mx']
addBiasMx = identificationParams['add bias term mx']
identifyMy = identificationParams['identify my']
addBiasMy = identificationParams['add bias term my']
identifyMz = identificationParams['identify mz']
addBiasMz = identificationParams['add bias term mz']


# Extract model saving parameters 
saveModels = identificationConfig['saving models']['save identified models']
saveModelID = identificationConfig['saving models']['model ID']
modelDir = ifNullDefault(identificationConfig['saving models']["save directory"], os.path.join(os.getcwd(), 'models'))



# Open log_file
log = pd.read_csv(os.path.join(fileLogDir, '{}.csv'.format(fileLogName)), delimiter=',', header=0)

# Offset rowIdx to match python indexing, note that -2 is used since header is also skipped!
rowIdxs = np.array(utility._parseRows(rowIdxs)) - 2
validationRows = np.array(utility._parseRows(validationRows)) - 2

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
for row in rowIdxs:
    rowIdx = row
    filteredData = pd.read_csv(os.path.join(filterSavePath, '{}-{}'.format(log.loc[rowIdx, 'Onboard Name'], 'FL.csv')))
    filteredDataList.append(filteredData)
    # Load metadata and check consistency
    with open(os.path.join(filterSavePath, "{}-{}".format(log.loc[row, 'Onboard Name'], 'metadata.json')), 'r') as f:
        filterParams = checkParamConsistency(json.load(f), referenceFilterParams)
    referenceFilterParams = {
        'remove influence of gravity':filterParams['remove influence of gravity'],
        'use noise statistics in drone config':filterParams['use noise statistics in drone config'],
        'batteries':filterParams['batteries']
        }


'''
Plot trajectories
'''
if plotTrajectories or plotAnimation:
    # Attempt to load droneviz
    hasDroneViz = False
    try:
        sys.path.append(relativeLocs['droneviz'])
        import viz
        from common import drone
        hasDroneViz = True
    except:
        print('[ WARNING ] Could not load droneviz, skipping some plotting and animation steps. Continuing with identfication!')

    c1 = 'orangered'
    c2 = 'gold'
    c3 = 'firebrick'
    trajFigs = []

    if hasDroneViz:
        print('[ INFO ] Found droneviz! Generating plots...')
        def animationWrapper(frames, objectPoses, wrapperKwargs):
            idx = int(frames * wrapperKwargs['AnimationUpdateFactor'])
            xyz = objectPoses['myDrone']['position'][idx, :]
            axLims.update({'x':[xyz[:, 0] - wrapperKwargs['delta'], xyz[:, 0] + wrapperKwargs['delta']]})
            axLims.update({'y':[xyz[:, 1] - wrapperKwargs['delta'], xyz[:, 1] + wrapperKwargs['delta']]})
            axLims.update({'z':[xyz[:, 2] - wrapperKwargs['delta'], xyz[:, 2] + wrapperKwargs['delta']]})

        for row, data in enumerate(filteredDataList):
            t = data['t'].to_numpy()

            # Downsample to optimize visualization
            sampleDes = 30
            downSampleRatio = int(1/(sampleDes*(t[1] - t[0])))

            # Import flight data
            t = t[::downSampleRatio]
            xyz = data[['x', 'y', 'z']].to_numpy()[::downSampleRatio]
            eul = data[['roll', 'pitch', 'yaw']].to_numpy()[::downSampleRatio]
            omega = data[['w1', 'w2', 'w3', 'w4']].to_numpy()[::downSampleRatio]
            quat = angleFuncs.Eul2Quat(eul) # Orientation in quaternions

            # Initialize animation object
            anim = viz.animation(posHorizon=int(len(t)/(NShots-1)))

            # Define drone object
            droneBody = drone.body(origin = xyz[0, :], rpy = angleFuncs.Quat2Eul(quat[0, :]))
            droneBody.R = 0.1
            droneBody.b = 0.2
            droneBody.rotorArms = []
            droneBody.bodyPlotKwargs.update({'linewidth':2})
            droneBody.hubHeight = -1*droneBody.b/3
            droneBody._initRotorArms()
            droneBody._setHistoryColor('#00A6D6')
            # Add droneBody to simulation
            anim.addActor(droneBody, name = 'myDrone')

            # Compile data into pose object
            myDronePose = {
                'time':t,
                'position':xyz.reshape(-1, 1, 3),
                'rotation_q':quat.reshape(-1, 1, 4),
                'inputs':omega.reshape(-1, 1, 4)
            }
            objectPoses = {'myDrone':myDronePose}

            if plotTrajectories:
                # 3-D plots
                t_interval = np.linspace(t[0], t[-1], NShots)
                anim.asImage(objectPoses, t_interval, uniformAxes = True)
                
                # 2-D plots
                pva_fig = plotter.plotPosVelAccTime(data['t'], data['x'], data['u'], data['ax'], 
                                                colors=(c1, c1, c1))
                pva_fig = plotter.plotPosVelAccTime(data['t'], data['y'], data['v'], data['ay'], 
                                                    colors=(c2, c2, c2), parentFig = pva_fig)
                pva_fig = plotter.plotPosVelAccTime(data['t'], data['z'], data['w'], data['az'], 
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
                
                plt.show()

            if plotAnimation:
                # Create animation
                axLims = {'x':[-2, 2], 'y':[-2, 2], 'z':[-2, 2]}
                # Adjust frame rate based on speed up factor and modify pos horizon
                anim.setFPS(t, int(sampleDes/animationSpeedUpFactor))
                anim.posHorizon = int(2/(t[1] - t[0])) # 2 seconds
                # anim.animate(t, objectPoses, axisLims=axLims, wrapper=animationWrapper, wrapperKwargs={'delta':1})
                anim.liveAnimation(t, objectPoses, keepAliveForSaving = False)
                plt.show()
    
    else:
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
    # NOTE: Control moments also converts rotor speeds to rad/s and adds w_tot and w_avg. It is an important step even if control moments are not used!
    DataList = quadrotorFM.addControlMoments(filteredDataList, rotorConfig, r_sign, rotorDir)
    for d in DataList:
        d['F_den'] = np.ones(d['t'].to_numpy().shape)
        d['M_den'] = np.ones(d['t'].to_numpy().shape)
        w_avg = d['w_avg'].to_numpy()
        # Advance ratios
        d['mu_x'] = np.divide(d['u'].to_numpy(), w_avg*droneParams['R'])
        d['mu_y'] = np.divide(d['v'].to_numpy(), w_avg*droneParams['R'])
        d['mu_z'] = np.divide(d['w'].to_numpy(), w_avg*droneParams['R'])

        # Normalize the induced velocity
        if 'v_in' in list(d.columns):
            d['mu_vin'] = np.divide(d['v_in'].to_numpy(), w_avg*droneParams['R'])




'''
Add extra columns, useful for identification, to DataFrames and combine into a single DataFrame
'''
DataList = normalization.addExtraColsAndTrim(DataList, usableDataRatio)


# Remove gravity component from data and add
# Check if g is removed from metadata files
hasGravity = not referenceFilterParams['remove influence of gravity']

droneMasses = []
wHovers = []


for rowIdx, d in zip(rowIdxs, DataList):
    droneMass = log.loc[rowIdx, 'Mass']*10**(-3)
    droneMasses.append(droneMass)

    F = d[['Fx', 'Fy', 'Fz']].to_numpy()
    acc = d[['ax', 'ay', 'az']].to_numpy()
    att = d[['roll', 'pitch', 'yaw']].to_numpy()
    quat = angleFuncs.Eul2Quat(att)
    gVector = np.zeros((len(d), 3))
    gVector[:, 2] = g

    gB = angleFuncs.QuatRot(quat, gVector, rot='E2B')

    # Find hovering flight, assume where all angles approx 0 is hovering
    hoverIdxs = np.where(np.isclose(gB[:, 2], g, rtol = 0.01))
    if len(hoverIdxs[0]):
        wi = d[['w1', 'w2', 'w3', 'w4']].to_numpy()[hoverIdxs, :]
        _F = F.copy()
        if not hasGravity:
            # If gravity was removed, recover gravity information for hovering flight such that
            # kappa can be estimated
            _F = F.copy() - droneMass*gB
        FzHover = _F[hoverIdxs, 2]
        wi2 = np.sum(np.square(wi).reshape(-1, 4), axis = 1).reshape(-1, 1)
        kappa0 = np.matmul(np.matmul(np.linalg.inv(np.matmul(wi2.T, wi2)), wi2.T), FzHover.reshape(-1, 1)).__array__()[0][0]

        # Record hovering thrust
        wHover = np.sqrt((-droneMass*g)/(4*kappa0))
        wHovers.append(wHover)

        # Compute the relative thrust difference from hover
        deltaW = d[['w1', 'w2', 'w3', 'w4']].to_numpy() - wHover
        d[['d_w1', 'd_w2', 'd_w3', 'd_w4']] = deltaW
        d['d_w_tot'] = np.sum(deltaW, axis = 1)

        # Sote results
        if hasGravity:
            d[['Fx_g', 'Fy_g', 'Fz_g']] = droneMass*gB
            d[['ax_g', 'ay_g', 'az_g']] = gB
            d[['Fx_B', 'Fy_B', 'Fz_B']] = F + droneMass*gB
            d[['ax_B', 'ay_B', 'az_B']] = acc + gB
            d[['Fx_tot', 'Fy_tot', 'Fz_tot']] = F
            d[['Fx', 'Fy', 'Fz']] = F
            d[['ax', 'ay', 'az']] = acc
        else:
            d[['Fx_g', 'Fy_g', 'Fz_g']] = droneMass*gB
            d[['ax_g', 'ay_g', 'az_g']] = gB
            d[['Fx_B', 'Fy_B', 'Fz_B']] = F
            d[['ax_B', 'ay_B', 'az_B']] = acc
            d[['Fx_tot', 'Fy_tot', 'Fz_tot']] = _F
            d[['Fx', 'Fy', 'Fz']] = F
            d[['ax', 'ay', 'az']] = acc

    else:
        wHovers.append(np.nan)


droneParams['m'] = np.nanmean(droneMasses)
droneParams['wHover'] = np.nanmean(wHovers)
droneParams['wHover_ERPM'] = np.nanmean(wHovers)/(2*np.pi/60) # to eRPM

droneMass = droneParams['m']


'''
Estimate actuator dynamics per motor
'''
showTauDist = identificationParams['show actuator constant estimation']
taus = np.zeros((len(DataList), 4))
for i, d in enumerate(DataList):
    _taus = quadrotorFM.estimateActuatorDynamics(d, threshold = (2*np.pi/60) * 50)
    taus[i] = np.array([v for v in _taus.values()])

if showTauDist:
    nonNaNTaus = taus[~np.isnan(np.sum(taus, axis = 1))]
    labels = [r'$\tau_{\omega_{1}}$', r'$\tau_{\omega_{2}}$', r'$\tau_{\omega_{3}}$', r'$\tau_{\omega_{4}}$']
    colors = ('#008bb4', '#e67d0a', 'mediumaquamarine', 'indianred')
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111)
    bplots = ax.boxplot(nonNaNTaus, labels = labels, patch_artist=True)
    for lines, face, c in zip(bplots['medians'], bplots['boxes'], colors):
        # lines.set_color('#ffbe3c')
        lines.set_color('firebrick')
        lines.set_linewidth(3)
        face.set_facecolor(c)

    ax.set_xlabel(r'$\mathbf{Motor}$, -', fontsize = 14)
    ax.set_ylabel(r'$\mathbf{Time}$ $\mathbf{constant}$, s', fontsize = 14)
    plotter.prettifyAxis(ax)
    plt.tight_layout()
    plt.show()

tau = np.nanmedian(taus, axis = 0)
if any(tau < 0):
    print('[ WARNING ] One of the estimated rotor constants appears to be negative. This is not physical. Defaulting the value to the average of the other rotors')
    violatingTaus = np.where(tau < 0)[0]
    if len(violatingTaus) == 4:
        raise ValueError('[ ERROR ] All rotor rate constants appear to be negative. This cannot happen.')
    avgTau = np.nanmean(tau[~(tau<0)])
    tau[violatingTaus] = avgTau

tausDict = {'taus':{k:v for k, v in zip(['w1', 'w2', 'w3', 'w4'], list(tau))}}
droneParams.update(tausDict)

with open(os.path.join(savePath, 'taus.json'), 'w') as f:
    json.dump(tausDict, f, indent = 4)


'''
Aggregate data
'''
ProcessedData = utility.aggregateData(DataList)
segregatedIdxs = []
counter = 0
for i in DataList:
    segregatedIdxs.append(counter + len(i) - 1)
    counter += len(i)

segregatedVIdxs = []
if len(validationRows):
    validationIdxs, segregatedVIdxs = utility.findAggregatedIdxs(validationRows, rowIdxs, DataList)



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
    _excitationIdxs = partitioning.getSystemExcitations(excitationIdxs, DataList, locals(), isolationMethods, spread = excitationSpread,
                                            variance_threshold=excitationThreshold, height_threshold = 0.5*excitationThreshold, 
                                            prominence_threshold = 0.9*excitationThreshold)
    excitationIdxs.update({k:v for k, v in _excitationIdxs.items()})

    if showExcitations:
        for key in excitationIdxs.keys():
            if identificationParams[f'identify {key.lower()}']:
                _fig = plotter.plotExcitations(key, excitationIdxs, ProcessedData, segregatedIdxs=segregatedIdxs)
        plt.show()

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
            [_, _idx_train], [_, _idx_test] = partitioning.PartitionData(_D.to_numpy(), partitionRatio, Method='Random', batch_size = jointTimeHorizon)
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


'''
IDENTIFY MODELS
'''
# Set to true to keep record of all models in global scope. Is more memory intensive. 
DEBUG_FLAG = True
# Set to true to save (model specific) training and testing indices
SAVE_INDICES = False

# Load simpleModel data, or set up if it does not exist
if os.path.exists(os.path.join(savePath, 'simpleModel.json')):
    with open(os.path.join(savePath, 'simpleModel.json'), 'r') as f:
        simpleModel = json.load(f)
else:
    simpleModel = {}

# Create metadata file
metadata = identificationConfig.copy()
metadata.update({'data filtering':filterParams})
metadata.update({'identification parameters':identificationParams})

metadata.update({'additional info':{'droneMass':droneMass, 'hover omega (rad/s)':np.nanmean(wHovers), 'hover omega (eRPM)':np.nanmean(wHovers)/(2*np.pi/60)}})


if identifyFx:
    print('\n\n[ INFO ] Identifying polynomial model for Fx...')
    polys_Fx = polyData['Fx']['candidates']
    fixed_Fx = polyData['Fx']['fixed']
    
    FxModel, FxPreds = identifyPolyModel(ProcessedData, 'Fx', polys_Fx, fixed_Fx, idx_train, idx_test, excitationIdxs['Fx'],
                                normalizer = ProcessedData['F_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, bias = addBiasFx)

    FxOut = savePolyModel(saveModelID, modelDir, metadata, FxModel, FxPreds, ProcessedData, 'Fx', idx_train, idx_test, segregatedIdxs, segregatedVIdxs,
                                excitationIdxs['Fx'], PIConfidenceLevel, PIresults, normalizer = ProcessedData['F_den'].to_numpy().reshape(-1),
                                DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)


    PIresults = FxOut['prediction interval metrics']
    figFx = FxOut['fig']
    ax = figFx.axes[0]
    ax.set_ylabel(r'$\mathbf{Force, \quad F_{x}}\quad [N]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)



if identifyFy:
    print('\n\n[ INFO ] Identifying polynomial model for Fy...')
    polys_Fy = polyData['Fy']['candidates']
    fixed_Fy = polyData['Fy']['fixed']
    
    FyModel, FyPreds = identifyPolyModel(ProcessedData, 'Fy', polys_Fy, fixed_Fy, idx_train, idx_test, excitationIdxs['Fy'], 
                                normalizer = ProcessedData['F_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, bias = addBiasFy)
    
    FyOut = savePolyModel(saveModelID, modelDir, metadata, FyModel, FyPreds, ProcessedData, 'Fy', idx_train, idx_test, segregatedIdxs, segregatedVIdxs,
                                excitationIdxs['Fy'], PIConfidenceLevel, PIresults, normalizer = ProcessedData['F_den'].to_numpy().reshape(-1),
                                DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)

    PIresults = FyOut['prediction interval metrics']                              
    figFy = FyOut['fig']
    ax = figFy.axes[0]
    ax.set_ylabel(r'$\mathbf{Force, \quad F_{y}}\quad [N]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)    



if identifyFz:
    print('\n\n[ INFO ] Identifying polynomial model for Fz...') 
    polys_Fz = polyData['Fz']['candidates']
    fixed_Fz = polyData['Fz']['fixed']
        
    FzModel, FzPreds = identifyPolyModel(ProcessedData, 'Fz', polys_Fz, fixed_Fz, idx_train, idx_test, excitationIdxs['Fz'],
                                normalizer = ProcessedData['F_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, bias = addBiasFz)

    FzOut = savePolyModel(saveModelID, modelDir, metadata, FzModel, FzPreds, ProcessedData, 'Fz', idx_train, idx_test, segregatedIdxs, segregatedVIdxs,
                                excitationIdxs['Fz'], PIConfidenceLevel, PIresults, normalizer = ProcessedData['F_den'].to_numpy().reshape(-1),
                                DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)


    PIresults = FzOut['prediction interval metrics']
    figFz = FzOut['fig']
    ax = figFz.axes[0]
    ax.set_ylabel(r'$\mathbf{Force, \quad F_{z}}\quad [N]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)

    # Get simple model for T = -kappa * sum(w_i^2); useful for some controllers which assume affine systems
    _w_2_total = ProcessedData['w1']**2 + ProcessedData['w2']**2 + ProcessedData['w3']**2 + ProcessedData['w4']**2
    _w_total = ProcessedData['w1'] + ProcessedData['w2'] + ProcessedData['w3'] + ProcessedData['w4']
    
    [kappaFz_w_2, _] = FzModel._techniqueModule._OLS(np.matrix(_w_2_total[idx_train]).T, -1*np.matrix(ProcessedData['Fz'][idx_train]).T, hasBias = False)
    [kappaFz_w, _] = FzModel._techniqueModule._OLS(np.matrix(_w_total[idx_train]).T, -1*np.matrix(ProcessedData['Fz'][idx_train]).T, hasBias = False)

    simpleModel.update({'kappaFz_w_2':float(kappaFz_w_2.__array__()[0][0])})
    simpleModel.update({'kappaFz_w':float(kappaFz_w.__array__()[0][0])})



if identifyMx:
    print('\n\n[ INFO ] Identifying polynomial model for Mx...')
    polys_Mx = polyData['Mx']['candidates']
    fixed_Mx = polyData['Mx']['fixed']

    MxModel, MxPreds = identifyPolyModel(ProcessedData, 'Mx', polys_Mx, fixed_Mx, idx_train, idx_test, excitationIdxs['Mx'],
                                normalizer = ProcessedData['M_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, bias = addBiasMx)
    
    MxOut = savePolyModel(saveModelID, modelDir, metadata, MxModel, MxPreds, ProcessedData, 'Mx', idx_train, idx_test, segregatedIdxs, segregatedVIdxs,
                                excitationIdxs['Mx'], PIConfidenceLevel, PIresults, normalizer = ProcessedData['M_den'].to_numpy().reshape(-1),
                                DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)
 
    PIresults = MxOut['prediction interval metrics']
    figMx = MxOut['fig']
    ax = figMx.axes[0]
    ax.set_ylabel(r'$\mathbf{Moment, \quad M_{x}}\quad [Nm]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)

    # Get simple model for Mx = kappaMx * Up
    [kappaMx_w_2, _] = MxModel._techniqueModule._OLS(np.matrix(ProcessedData['U_p'][idx_train]).T, np.matrix(ProcessedData['Mx'][idx_train]).T, hasBias = False)
    [kappaMx_w, _] = MxModel._techniqueModule._OLS(np.matrix(ProcessedData['u_p'][idx_train]).T, np.matrix(ProcessedData['Mx'][idx_train]).T, hasBias = False)

    simpleModel.update({'kappaMx_w_2':float(kappaMx_w_2.__array__()[0][0])})
    simpleModel.update({'kappaMx_w':float(kappaMx_w.__array__()[0][0])})


if identifyMy:
    print('\n\n[ INFO ] Identifying polynomial model for My...')
    polys_My = polyData['My']['candidates']
    fixed_My = polyData['My']['fixed']

    MyModel, MyPreds = identifyPolyModel(ProcessedData, 'My', polys_My, fixed_My, idx_train, idx_test, excitationIdxs['My'],
                                normalizer = ProcessedData['M_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, bias = addBiasMy)

    MyOut = savePolyModel(saveModelID, modelDir, metadata, MyModel, MyPreds, ProcessedData, 'My', idx_train, idx_test, segregatedIdxs, segregatedVIdxs,
                                excitationIdxs['My'], PIConfidenceLevel, PIresults, normalizer = ProcessedData['M_den'].to_numpy().reshape(-1),
                                DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)
    
    PIresults = MyOut['prediction interval metrics']
    figMy = MyOut['fig']
    ax = figMy.axes[0]
    ax.set_ylabel(r'$\mathbf{Moment, \quad M_{y}}\quad [Nm]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)

    # Get simple model for My = kappaMy * Uq
    [kappaMy_w_2, _] = MyModel._techniqueModule._OLS(np.matrix(ProcessedData['U_q'][idx_train]).T, np.matrix(ProcessedData['My'][idx_train]).T, hasBias = False)
    [kappaMy_w, _] = MyModel._techniqueModule._OLS(np.matrix(ProcessedData['u_q'][idx_train]).T, np.matrix(ProcessedData['My'][idx_train]).T, hasBias = False)

    simpleModel.update({'kappaMy_w_2':float(kappaMy_w_2.__array__()[0][0])})
    simpleModel.update({'kappaMy_w':float(kappaMy_w.__array__()[0][0])})



if identifyMz:
    print('\n\n[ INFO ] Identifying polynomial model for Mz...')
    polys_Mz = polyData['Mz']['candidates']
    fixed_Mz = polyData['Mz']['fixed']

    MzModel, MzPreds = identifyPolyModel(ProcessedData, 'Mz', polys_Mz, fixed_Mz, idx_train, idx_test, excitationIdxs['Mz'],
                                normalizer = ProcessedData['M_den'].to_numpy().reshape(-1), regressorCap=polyModelCap, bias = addBiasMz)

    MzOut = savePolyModel(saveModelID, modelDir, metadata, MzModel, MzPreds, ProcessedData, 'Mz', idx_train, idx_test, segregatedIdxs, segregatedVIdxs,
                                excitationIdxs['Mz'], PIConfidenceLevel, PIresults, normalizer = ProcessedData['M_den'].to_numpy().reshape(-1),
                                DEBUG_FLAG=DEBUG_FLAG, SAVE_INDICES=SAVE_INDICES)    
   
    PIresults = MzOut['prediction interval metrics']                                
    figMz = MzOut['fig']
    ax = figMz.axes[0]
    ax.set_ylabel(r'$\mathbf{Moment, \quad M_{z}}\quad [Nm]$', fontsize = 16)
    ax.set_xlabel(r'$\mathbf{Sample}\quad [-]$', fontsize = 16)    

    # Get simple model for Mz = kappaMz * Ur
    [kappaMz_w_2, _] = MzModel._techniqueModule._OLS(np.matrix(ProcessedData['U_r'][idx_train]).T, np.matrix(ProcessedData['Mz'][idx_train]).T, hasBias = False)
    [kappaMz_w, _] = MzModel._techniqueModule._OLS(np.matrix(ProcessedData['u_r'][idx_train]).T, np.matrix(ProcessedData['Mz'][idx_train]).T, hasBias = False)

    simpleModel.update({'kappaMz_w_2':float(kappaMz_w_2.__array__()[0][0])})
    simpleModel.update({'kappaMz_w':float(kappaMz_w.__array__()[0][0])})


# Save simpleModel json
with open(os.path.join(savePath, 'simpleModel.json'), 'w') as f:
    json.dump(simpleModel, f, indent = 4)

printPIs(PIresults)

# Save PI results
with open(os.path.join(savePath, 'PIresults.pkl'), 'wb') as f:
    pkl.dump(PIresults, f)
    f.close()  

plt.show()

# End
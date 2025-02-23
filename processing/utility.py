from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import os
import json
import platform
import fnmatch
from scipy.stats import norm

# Note 'cubic' interpolation_kind gives third-order spline interpolation 
def resample(Data, newFreq, interpolation_kind = 'linear'):
    time = Data['t'].to_numpy()
    vals = Data.to_numpy()
    func = interp1d(time, vals, kind = interpolation_kind, axis=0)
    newTime = np.arange(time[0], time[-1], 1/newFreq)
    newVals = func(newTime)
    resampled_Data = pd.DataFrame(newVals, columns=Data.columns)
    return resampled_Data



def aggregateData(dataList):
    i = 0
    # Create new indices for aggregated dataframe while keeping track of original indices
    for df in dataList:
        # # Make copy so original is not modified. 
        # df = df.copy()
        # Set new index values
        new_indices = np.arange(i, (i+len(df)))
        df.set_index(pd.Index(new_indices), inplace=True)
        # Remove automatically added column
        try:
            df.drop(labels=['Unnamed: 0'], axis = 1, inplace=True)
        except KeyError:
            pass
        i += len(df)
    aggrData = pd.concat(dataList)
    return aggrData



def segregateData(dataFrame, col='index', delimiter=0):
    indices = dataFrame[col].to_numpy()
    splitIdx = np.where(indices == delimiter)[0]
    # If there are splits in the DataFrame using col and delimiter as splitting criteria
    if len(splitIdx):
        splitIdx = np.hstack((splitIdx, len(indices)))
        i = 0
        df_list = []
        for j in range(len(splitIdx)-1):
            idx = splitIdx[j + 1]
            df = dataFrame.loc[i:(idx - 1), :].copy()
            df.reset_index(inplace=True, drop=True)
            df.drop(labels=['index'], axis = 1, inplace=True)
            df_list.append(df)
            i = idx
    else:
        print('[ INFO ] Could not segregate input DataFrame based on column "{}" with delimiter "{}"'.format(col, delimiter))
        print('[ INFO ] Packaging input DataFrame into list')
        df_list = [dataFrame]
    return df_list



def timeOfDay2Seconds(x):
    '''
    x as string in format HH:MM:SS
    '''
    HH = float(x[:2])
    MM = float(x[3:5])
    SS = float(x[6:])
    return SS + 60*(MM + 60*(HH))


def milliseconds2seconds(x):
    return x*10**(-6)



def BBL2CSV(filedir, filename, index = -1, options = '--merge-gps --simulate-imu --debug'):
    # Open relativeImportLocations to find where blackbox tools is located
    with open('relativeImportLocations.json', 'r') as f:
        relativeImports = json.load(f)

    if platform.system() == 'Windows':
        blackboxToolsLoc = os.path.abspath(relativeImports['blackbox-tools-0.4.3-windows'])
    else:
        blackboxToolsLoc = os.path.abspath(relativeImports['blackbox-tools-0.4.3'])        

    file = os.path.join(filedir, filename)
    # blackboxToolsLoc = os.path.abspath(relativeImports['blackbox-tools-0.4.3'])
    cwd = os.getcwd()
    absfiledir = os.path.abspath(filedir)
    absPath = os.path.abspath(file + '.BFL')
    if not os.path.exists(absPath):
        absPath = os.path.abspath(file + '.BBL')
        if not os.path.exists(absPath):
            raise RuntimeError('Cannot find file {}'.format(filename))

    # Change working directory to location of blackbox-tools
    os.chdir(blackboxToolsLoc)
    
    # Run blackbox tools on file
    if platform.system() == 'Linux':
        os.system('./blackbox_decode {} "{}"'.format(options, absPath))
    else:
        os.system('blackbox_decode {} {}'.format(options, absPath))

    # Find file of interest through index
    # filesInDir = os.listdir(filedir)
    filesInDir = os.listdir(absfiledir)
    filesOfInterest = [f for f in filesInDir if checkConditions_string(['{}*.csv'.format(filename)], f) and f.endswith('.csv')]
    createdFile = filesOfInterest[index]

    # # Rename file
    # filename = filename.split('.')[0]
    # app = str(index).zfill(2)
    # createdFile = filename + '.{}.csv'.format(app)
    os.chdir(absfiledir)
    os.rename(createdFile, filename + '.csv')

    # Return to working directory
    os.chdir(cwd)
    
    return None

def checkConditions_string(conditions, string):
    satisfied = False
    for c in conditions:
        if fnmatch.fnmatch(string, c):
            satisfied = True
            break
    return satisfied



def extractConfig(rowIdx, logFile):
    configFile = '{}.json'.format(logFile.loc[rowIdx, 'Drone Config File'])
    configPath = logFile.loc[rowIdx, 'Drone Config Path']
    path2File = os.path.join(configPath, configFile)

    with open(path2File, 'r') as f:
        configData = json.load(f)
        f.close()

    droneParams = {'R':float(configData['rotor radius']),
                   'b':float(configData['b']),
                   'Iv':np.array(np.matrix(configData['moment of inertia'])),
                   'idle RPM':float(configData['idle RPM']),
                   'max RPM':float(configData['max RPM'])}
    
    if '(estimated) imu sensor noise statistics' in configData.keys():
        droneParams.update({'imu noise statistics':configData['(estimated) imu sensor noise statistics']})

    rotorConfig = configData['rotor config']

    rotor1Dir = configData['rotor1 rotation direction']

    quadIdleRPM = configData['idle RPM']

    return droneParams, rotorConfig, rotor1Dir, quadIdleRPM




def findAggregatedIdxs(rows, rowIdxs, DataList):
    segregatedIdxs = []
    segregatedIdxsLims = []
    for r in rows:
        idx_in_list = np.where(np.array(rowIdxs) == r)[0][0]
        idxStart = 0
        idxEnd = len(DataList[0])
        for i in range(idx_in_list):
            idxStart = idxEnd
            idxEnd += len(DataList[i + 1])
        segregatedIdxs = segregatedIdxs + list(np.arange(idxStart, idxEnd))
        segregatedIdxsLims.append([idxStart, idxEnd])
    return segregatedIdxs, segregatedIdxsLims




# Find index of loss-of-control (LOC)
def findLOC(onBoardData):
    idxs = np.where((onBoardData['w1'] == 0) & (onBoardData['w2'] == 0) & (onBoardData['w3'] == 0) & (onBoardData['w4'] == 0))[0]
    if len(idxs):
        # Find where idxs 'jumps'
        d_idxs = idxs[1:] - idxs[:-1]
        d_idxs = np.hstack((0, d_idxs))
        jumps = np.where(d_idxs != 1)[0]
        # Take last time, assume it is a crash
        idx = idxs[jumps[::-1][0]]
    else:
        idx = len(onBoardData['time'])
    return idx



def qualityPI(y_true, y_pred, y_var, conf = 0.95, normalizeMPIW = False):
    '''Function to determine the quality of the prediction intervals, based on their coverage probability (PICP) and the (normalized) mean width of the PIs (MPIW). 

    :param y_true: True target values, as 1-D array
    :param y_pred: Predicted (or estimated) target values, as 1-D array 
    :param y_var: Associated variance with predicted targets, as 1-D array
    :param conf: Confidence level used to construct the prediction intervals, Default = 0.95 (95% prediction intervals)

    :return: (PICP, MPIW)
    '''
    # Define lower and upper bounds of the prediction interval based on the confidence level
    PI_lower, PI_upper = buildIntervalBounds(conf, y_pred, y_var, N = 1)

    N = len(y_true)

    ''' Probability Coverage '''
    # Count number of samples where y_true is within the prediction interval
    n_coverage = np.where((y_true >= PI_lower) & (y_true <= PI_upper))[0]
    # Compute the probability coverage 
    PI_coverage = (len(n_coverage)/N)*100

    ''' Mean PI interval '''
    MPIW = np.nanmean((PI_upper - PI_lower), axis = 0)
    if normalizeMPIW:
        N_MPIW = (MPIW/(np.nanmax(y_true) - np.nanmin(y_true)))*100
    else:
        N_MPIW = MPIW

    return PI_coverage, N_MPIW



def buildIntervalBounds(confidenceLevel, y_pred, y_var, N = 1):
    '''Function to determine lower and upper limits of a prediction interval based on a given confidence level

    :param confidenceLevel: Confidence level used to construct the prediction intervals
    :param y_pred: Predicted (or estimated) target values, as 1-D array 
    :param y_var: Associated variance with predicted targets, as 1-D array
    :param N: Number of samples for target predictions. Typically N = 1 when making point predictions, so Default = 1
    
    :return: (PI_lower, PI_upper); tuple of 1-D arrays corresponding to the lower and upper interval bounds respectively 
    '''
    if confidenceLevel >= 1:
        print('[ WARNING ] User specified a confidence interval >= 1. Defaulting to 0.99.')
        confidenceLevel = 0.99
    z_conf = norm.ppf((1+confidenceLevel)/2)
    PI_lower = y_pred.reshape(-1) - z_conf*np.sqrt(y_var.reshape(-1))/np.sqrt(N)
    PI_upper = y_pred.reshape(-1) + z_conf*np.sqrt(y_var.reshape(-1))/np.sqrt(N)
    return PI_lower, PI_upper


def _parseRows(rows):
    rowIdxs = []
    for r in rows:
        # Is number 
        if isinstance(r, int):
            rowIdxs.append(r)
        else:
            # Check if string integer
            try:
                _r = int(r)
                rowIdxs.append(_r)
            except ValueError:
                # Is a range
                rng = r.split('-')
                rRange = np.arange(int(rng[0]), int(rng[1])+1, dtype=int)
                for _r in rRange:
                    rowIdxs.append(_r)
    return rowIdxs

'''
Dependencies
'''
import os
import json
from numpy import nanmean, nanstd, pi, where
from pandas import read_csv, to_numeric
import matplotlib.pyplot as plt

from processing.utility import BBL2CSV
from processing.utility import aggregateData



'''
Definitions
'''
def checkExtensions(x, acceptedExtensions):
    satisfied = False
    for ext in acceptedExtensions:
        if x.endswith(ext):
            satisfied = True
            break
    return satisfied


def importData(filedir, filename, BBLOptions = '--merge-gps --simulate-imu --debug --unit-acceleration m/s2 --unit-frame-time s --unit-rotation rad/s'):
    # Check if .csv exits
    filepath = os.path.join(filedir, filename.split('.')[0] + '.csv')
    if not os.path.exists(filepath):
        print('[ INFO ] Could not find .csv file for {}. Attempting to convert .bfl instead'.format(filename))
        try:
            BBL2CSV(filedir, filename, index = -1, options = BBLOptions)
            print('[ INFO ] Successfully converted {}.bfl to .csv'.format(filename))
        except FileNotFoundError:
            raise RuntimeError('File: {} not found.')
    print('[ INFO ] Importing On-Board data from: {}'.format(os.path.join(filedir, '{}.csv'.format(filename))))
    rawData = read_csv(os.path.join(filedir, '{}.csv'.format(filename)), delimiter=',', header=0)

    colMapping = {
                ' time (s)':'t',                         # s
                ' roll':'roll',                          # deg
                ' pitch':'pitch',                        # deg
                ' heading':'yaw',                        # deg (with offset)
                ' gyroADC[0] (rad/s)':'p',               # rad/s
                ' gyroADC[1] (rad/s)':'q',               # rad/s
                ' gyroADC[2] (rad/s)':'r',               # rad/s
                ' accSmooth[0] (m/s/s)':'ax',            # m/s/s
                ' accSmooth[1] (m/s/s)':'ay',            # m/s/s
                ' accSmooth[2] (m/s/s)':'az',            # m/s/s
                ' debug[0]':'w1',                        # erpm
                ' debug[1]':'w2',                        # erpm
                ' debug[2]':'w3',                        # erpm
                ' debug[3]':'w4'                         # erpm
                }

    rawData.rename(columns=colMapping, inplace=True)
    nanIdx = to_numeric(rawData['loopIteration'], errors = 'coerce').isnull() # Coerce changes strings that are not numeric to nan
    # Drop nan columns
    rawData.drop(index = where(nanIdx)[0], inplace=True)
    Data = rawData.loc[:, list(colMapping.values())].copy().reset_index().astype(float)
    Data.rename(columns=colMapping, inplace=True)
    # import code
    # code.interact(local=locals())

    # Shift time back to 'zero'
    Data['t'] = Data['t'] - Data.loc[0, 't']

    # Convert angles to rad
    Data[['roll', 'pitch', 'yaw']] = Data[['roll', 'pitch', 'yaw']]*pi/180

    # Transform from betaflight axis system (x-forward, y-left, z-up) to conventional aerospace system (x-forward, y-right, z-down) 
    # Rotate 180 degrees about x axis
    flips = ['pitch', 'yaw', 'q', 'r', 'ay', 'az']
    for f in flips:
        Data[f] = -1*Data[f]

    return Data


def estimateNoiseStats(data):
    x_bar = nanmean(data)
    x_std = nanstd(data)
    return x_bar, x_std


'''
Main script
'''
# Open configuration file
with open('estimateDroneNoiseConfig.json', 'r') as f:
    configFile = json.load(f)

# Load config contents
droneConfigLoc = configFile['drone configuration file path']
droneConfigFile = configFile['drone configuration file name']
addResults = configFile['add noise results to config file']
cutoffRatio = float(configFile['cutoff ratio'])
noiseFilesLoc = configFile['noise files (onboard data) path'] # Note, these data should not involve any manoeuvres (or flying), only arming such that variations are soley due to noise
toExclude = configFile['exclude files']

# Import onboard data
files = os.listdir(noiseFilesLoc)
# Only include expected file types
expectedExtensions = ['.bfl', '.bbl', '.BFL', '.BBL']
_datafiles = [f for f in files if checkExtensions(f, expectedExtensions) and f not in toExclude]
datafiles = [f.split('.')[0] for f in _datafiles]

rawDataList = []
for i, f in enumerate(datafiles):
    print('[ INFO ] Importing file {}/{}'.format(i+1, len(datafiles)))
    d = importData(noiseFilesLoc, f)
    idx = int(cutoffRatio*len(d))
    data = d.iloc[idx:-idx, :]
    # Remove means
    for col in data.columns:
        data[col] = data[col] - nanmean(data[col])
    rawDataList.append(data)

rawData = aggregateData(rawDataList)

# Get IMU noise stats
noiseVars = ['ax', 'ay', 'az', 'p', 'q', 'r', 'roll', 'pitch', 'yaw']
noiseStats = {}
for v in noiseVars:
    mu, sig = estimateNoiseStats(rawData[v])
    noiseStats.update({v:{'mean':str(mu), 'std':str(sig)}})

# Add results to droneConfigurationFile, if requested
if addResults:
    # Load drone configuration file
    with open(os.path.join(droneConfigLoc, droneConfigFile + '.json'), 'r') as f:
        droneParams = json.load(f)

    # Add results
    droneParams.update({'(estimated) imu sensor noise statistics':noiseStats})

    # Save updated configuration file
    with open(os.path.join(droneConfigLoc, droneConfigFile + '.json'), 'w') as f:
        json.dump(droneParams, f, indent = 4)
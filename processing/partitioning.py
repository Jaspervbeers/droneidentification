import numpy as np
from scipy.signal import find_peaks


def PartitionData(Data, Ratio, Method='Random', batch_size = None):
    '''Function to split data into training and testing subsets
    
    :param Data: Array-like data to partition
    :param Ratio: Float indicating ratio of Data that should be allocated to training (e.g. 0.8 means 80% of the data is used for training)
    :param Method: String indicating the way data should be split. Default = 'Random' where data subsets are obtained from a random sampling of the Data. 
    :param batch_size: Integer describing the sequence size of data. Default = None, so no batches will be assigned. If batch_size is not None, then Data (N, M) will be reshaped into (N/batch_size, batch_size, M). The data is then partitioned along the batch_size axis. This is useful for preserving temporally dependent information. 

    :return: Indices corresponding to the partition, as tuple (Training, Test) 
    '''
    Methods = {'Random':Random_PartitionData}
    DataOri = Data.copy()
    if batch_size is not None:
        shape = Data.shape
        remainder = shape[0] % batch_size
        if shape[0] % batch_size != 0:
            print('[ WARNING ] Data (N = {}) could not be evenly split into batches of size {}. Trimming last remaining points (= {} samples)'.format(shape[0], batch_size, remainder))
            # remainIdxs = -1*np.arange(-1*shape[0], -1*shape[0] + remainder + 1, 1)[::-1] - 1
            Data = Data[:-remainder]
        Data = Data.reshape(-1, batch_size, *shape[1:])
        Mask = np.arange(0, shape[0]-remainder).reshape(-1, batch_size)
    else:
        pass
        # remainIdxs = []

    if Method not in Methods.keys():
        print('[ ERROR ] Unknown Method: "{}" \nExpected one of the following: {}'.format(Method, list(Methods.keys())))
    else:
        [_trainingData, _trainingIdx], [_testData, _testIdx] = Methods[Method](Data, Ratio)
        if batch_size is not None:
            trainingIdx = Mask[_trainingIdx].flatten()
            # testIdx = np.hstack((Mask[_testIdx].flatten(), remainIdxs))
            testIdx = Mask[_testIdx].flatten()
            trainingData = DataOri[trainingIdx]
            testData = DataOri[testIdx]
        else:
            trainingIdx = _trainingIdx
            trainingData = _trainingData
            testIdx = _testIdx
            testData = _testData

        Test = [testData, testIdx]
        Training = [trainingData, trainingIdx]

    return Training, Test



def Random_PartitionData(Data, TrainingRatio, *argv):
    '''Function to partition data through random sampling without replacement

    :param Data: Data to be partitioned
    :param TrainingRatio: Ratio of data, as float ]0, 1[, to be allocated to the training data subset
    :param argv: Additional positional arguments, unused.
    
    :return: Tuple of lists as ([Partitioned training Data, Indices training], [Partitioned testing data, Indices testing])
    '''
    N = Data.shape[0]
    if TrainingRatio >= 1:
        print('\n[ WARNING ] Inputted training ratio is >= 1 when it should be < 1. Defaulting to 0.7\n')
        TrainingRatio = 0.7
    N_Training = int(TrainingRatio*N)

    indices = np.arange(N)
    indices_bool = np.ones((N, 1))
    indices_training = np.sort(np.random.choice(indices, N_Training, replace = False))
    indices_bool[indices_training] = 0
    indices_test = np.where(indices_bool)[0]

    try:
        TrainingData = Data[:, indices_training]
        TestData = Data[:, indices_test]
    except (IndexError, MemoryError) as e:
        TrainingData  = Data[indices_training, :]
        TestData = Data[indices_test, :]

    return [TrainingData, indices_training], [TestData, indices_test]


def getSystemExcitations(excitationIdxs, DataList, local, isolationMethods, variance_threshold=0.5, height_threshold = 0.3, prominence_threshold = 0.3, spread = 0.02):
    references = {'identifyFx':'Fx', 'identifyFy':'Fy', 'identifyFz':'Fz', 'identifyMx':'Mx', 'identifyMy':'My', 'identifyMz':'Mz'}
    # excitationIdxs = {}
    # for ref in references.keys():
    #     excitationIdxs.update({references[ref]:[]})
    startIdx = 0
    for D in DataList:
        for ref in references.keys():
            if ref in local.keys():
                if local[ref]:
                    ls = excitationIdxs[references[ref]]
                    idxs = findSystemExcitation(D[references[ref]].to_numpy(), Method = isolationMethods[references[ref]], variance_threshold = variance_threshold, height_threshold=height_threshold, prominence_threshold=prominence_threshold, spread = spread) + startIdx
                    ls = ls + list(idxs)
                    excitationIdxs.update({references[ref]:ls})
        startIdx += len(D)         
    return excitationIdxs



def _electIdentifiableData(X, threshold = 0.15, outlier_cutoff = 0.95, spread = 0.02):
    '''(Utility) Function to isolate system excitations in signal based on the local variance

    :param X: 1-D array containing samples of signal for which excitations should be extracted
    :param threshold: Threshold for what is considered an excitation, as a ratio of the maximum value of the data. Default = 0.15
    :param outlier_cutoff: Proportion, as float ]0, 1], of the ordered data magnitudes which is kept for excitation identification. That is data whose magnitudes are in the top (1 - outlier_cutoff) are removed before analysis. Default = 0.95
    :param spread: Ratio, as float < |1|, which describes the window width of additional points to be included before and after the identified excitation points. Ratio is expressed with respect to number of samples of the data.
    
    :return: Indices corresponding to isolated system excitations
    '''

    # Sort X by increasing magnitude
    X_sorted = sorted(np.abs(X))
    # Remove highest (1 - outlier_cutoff) of the data (considered rudimentary outliers)
    X_sorted_rm = X_sorted[:int(outlier_cutoff*len(X_sorted))]
    # Take last index as representative maximum
    X_max = X_sorted_rm[-1]
    X_lim = threshold*X_max
    # Find indices of X which are greater than this magnitude 
    relevant_idxs = np.where(np.abs(X) >= X_lim)[0]
    # Simply taking these indices will result in an incomplete signal since regions between peaks 
    # which are less than X_lim are also ignored. 
    # Our goal here is to remove regions where there is no system excitement, hence values < X_lim
    # around regions of excitement are necessary.
    # Therefore, we will also include indices surrounding relevant_idxs above. 
    spread_val = int(spread*len(X_sorted))
    spread_arr = np.arange(-spread_val, spread_val + 1, 1)

    idxs = np.zeros(len(spread_arr)*len(relevant_idxs))
    for i, v in enumerate(relevant_idxs):
        i_start = i*len(spread_arr)
        i_end = i_start + len(spread_arr)
        idxs[i_start:i_end] = v + spread_arr

    # Remove indices less than zero
    pos_idxs = idxs[np.where(idxs >= 0)]

    # Remove indices greater than len(X_sorted)
    pos_idxs = pos_idxs[np.where(pos_idxs < len(X_sorted))]

    # Remove duplicates
    idxs_out = np.unique(pos_idxs)

    return idxs_out.astype(int)



def findSystemExcitation(X, Method = 'variance', window = None, variance_threshold = 0.15, outlier_cutoff = 0.95, spread = 0.02, prominence_threshold = 0.3, height_threshold = 0.2, minHeight = 0.02):
    '''Function to isolate excitations in a signal
    
    :param X: 1-D array containing samples of signal for which excitations should be extracted
    :param Method: String describing which isolation method should be used; 'variance' or 'peak'. Default is 'variance', for which the excitations are determined based on the local variance of the signal. 'peak' isolated excitations based on the peaks in the data, and is more suitable for high-frequency and sharp responses. 
    :param window: (Only relevant if Method = 'variance') Running window size, as an integer of samples, for which the rolling local variances should be computed.
    :param variance_threshold: (Only relevant if Method = 'variance') Threshold for what is considered an excitation, as a ratio of the maximum value of the data. Default = 0.15
    :param outlier_cutoff: (Only relevant if Method = 'variance') Proportion, as float ]0, 1], of the ordered data magnitudes which is kept for excitation identification. That is data whose magnitudes are in the top (1 - outlier_cutoff) are removed before analysis. Default = 0.95
    :param spread: Ratio, as float < |1|, which describes the window width of additional points to be included before and after the identified excitation points. Ratio is expressed with respect to number of samples of the data.
    :param prominence_threshold: (Only relevant if Method = 'peak') Threshold, as float < |1|, describing the minimum prominence for peaks to be considered an excitation. Default = 0.3
    :param height_threshold: (Only relevant if Method = 'peak') Minimum peak height, relative to maximum magnitude in the signal, for a peak to be considered an excitation. Expressed as a float < |1|. Default = 0.2

    :return: Indices corresponding to system excitations. 
    '''
    
    # Define running window size
    if window is None:
        window = int(0.01 * max(X.shape))

    if Method.lower() == 'variance':
        # Compute rolling std
        rollingStd = np.nanstd(_rollingWindow(X, window), axis=-1)
        # Find regions of large excitations, based on local variance
        excitedIdxs = _electIdentifiableData(rollingStd, threshold=variance_threshold, outlier_cutoff=outlier_cutoff, spread=spread)
    
    elif Method.lower() == 'peak':
        # _X = X[:int(outlier_cutoff*len(X))]
        peakIdxs = find_peaks(np.abs(X), height=np.nanmax([minHeight, height_threshold*np.max(np.abs(X))]), prominence=prominence_threshold*np.max(np.abs(X)))
        spread_val = int(spread*len(X))
        spread_arr = np.arange(-spread_val, spread_val + 1, 1)
        idxs = np.zeros(len(spread_arr)*len(peakIdxs[0]))
        for i, v in enumerate(peakIdxs[0]):
            i_start = i*len(spread_arr)
            i_end = i_start + len(spread_arr)
            idxs[i_start:i_end] = v + spread_arr

        # Remove indices less than zero
        pos_idxs = idxs[np.where(idxs >= 0)]
        # Remove indices greater than len(X_sorted)
        pos_idxs = pos_idxs[np.where(pos_idxs < len(X))]
        # Remove duplicates
        excitedIdxs = np.unique(pos_idxs).astype(int)

    else:
        raise ValueError('Unknown Method. Expected <variance> or <peak>, got {} instead'.format(Method))

    return excitedIdxs



def _rollingWindow(x, window):
    '''(Utility) function that generates a rollowing window
    
    :param x: 1-D array of data for which a rolling window should be created
    :param window: Window size

    :return: Numpy array corresponding to rolling window instances
    '''

    # Expect only 1-D arrays
    if len(x.shape) > 1:
        x = np.array(x).reshape(-1)
    pad = np.ones(len(x.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(x.shape), dtype=np.int32)))
    x = np.pad(x, pad, mode='reflect')
    shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


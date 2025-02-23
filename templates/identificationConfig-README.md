# This readme describes how the runConfig file is structured and what the different fields mean. 

## Default folder structure
Much of this code relies (and creates) a certain folder structure for convenience. This folder structure may be changed in accordance with the needs of the user. 

The default folder structure is as follows, where the current working directory is assumed to be the location of the main model identification script, and will be called 'cwd'. 

**-cwd**
    **-Data** *(folder hosting the processed data. It is recommended that the raw data is stored here as well, e.g. under a subfolder called 'raw')*
        **-file_log.csv** *(book keeping file which links a given flight to the directories in which the raw on-board, optitrack (if used), video, and configuration data is stored. Comments are also provided which summarize what is done in the flight)*
        **-processed** *(folder containing results of the data processing scripts)*
            **-imported** *(location in which the imported (optitrack aligned) flight data is saved. Note that the imported files overwrite each other, so previous imports are not saved.)*
            **-filtered** *(location in which the filtered (i.e. processed) flight data is saved. Unlike for the imported files, these are saved using the associated raw data file names and will thus not be overwritten unless the same file is filtered again. The idea is that, once filtering is completed for a data set, it does not need to be conducted again each time a model is to be identified.)*
    **-models** *(location in which identified models are stored. Each model will be stored in its own subfolder)*
        **-model_log.xlsx** *(book keeping file for identified models, associated performance, and flights used. See provided template)*



## Logging file variables
The following section relates to the *parameters associated with the .csv file which book keeps the flight data.* For example, where the raw data files are located (see example file_log.csv)
```json
{"logging file":{             
    "directory":null,
    "filename":"file_log",
    "rows of flights to use (all)":[2, 3, 4],
    "rows of flights for validation":[3]}
}
```
- "directory" - string - provides an (absolute) path to the location of the file_log.csv (or equivalent logging file). Passing `null` indicates that the default location will be used (see **Default folder structure** above). 
- "filename" - string - is the name of the flight book keeping file. Default is 'file_log'(.csv)
- "rows of flights to use (all)" - list of int - is a list of row numbers in file_log.csv which correspond to the flights to be used for data processing and/or model identification. This includes validation flights. 
- "rows of flights for validation" -list of int - is a subset of the list above that indicates which of the flights are to be used solely for validation (i.e. removed from training). If left empty (i.e. []), no individual flights will be used for validation. 


## Data importing file variables
```json
{
    "data importing":{
        "import raw data":true,
        "resampling rate":500,
        "filter optitrack outliers":true,
        "save imported data":true,
        "imported data save directory":null,
        "align with optitrack using":["roll", "pitch"],
        "max permitted lag for optitrack":25
    }
}
```
- "import raw data" - boolean - to specify if the raw data should be imported. This can be set to false if the data has previously been imported and saved. 
- "resampling rate" - float - in Hz, that describes what sampling rate the data should be (re)sampled to. 
- "filter optitrack outliers" - boolean - that dictates whether optitrack outliers (e.g. due to loss of tracking or flickering) should be removed from the optitrack data. Only relevant if the optitrack system was used to collect data. 
- "save imported data" - boolean - to indicate whether the imported (resampled, and aligned if using opitrack) should be saved. 
- "imported data save directory" - string - is the directory in which the imported data should be saved, if desired. Default = null which indicates that the default directory should be used (see **Default folder structure** above)
- "align with optitrack using" - list of strings - of attitude angles which should be used when aligning the optitrack data with the on-board data. Permitted strings are "roll", "pitch", and "yaw". This is only relevant when using optitrack data
- "max permitted lag for optitrack" - float - in seconds, describing the maximum allowable lag between the optitrack measurements and the onboard data. This should typically be in the order of a few seconds. 


## Data filtering variables
```json
{
    "data filtering":{
        "run extended kalman filter":true,
        "save kalman filter convergence results":false,
        "remove influence of gravity":true,
        "save filtered data":true,
        "filtered data save directory":null
    }
}
```
- "run extended kalman filter" - boolean - describing if imported data should be filtered
- "save kalman filter convergence results" - boolean - describing if kalman filter convergence results should be saved alongside filtered data
- "remove influence of gravity" - boolean - indicating if the influence of gravity should be removed from the filtered force measurements 
- "save filtered data" - boolean - indicating if filtered data should be saved 
- "fileted data save directory" - string - directory in which the filtered data should be saved. If null, then the default directory is used (see **Default folder structure above**)



## Plotting variables 
```json
{
    "plotting":{
        "show trajectories":false,
        "show animation":false
    }
}
```
- "show trajectories" - boolean - indicates if the flight trajectories should be plotted and shown
- "show animation" - boolean - indicates if these flight trajectories should be animated, only applies of 'show trajectories' is true



## Data normalization parameters
```json
{
    "data normalization":{
            "normalize data":true,
            "usable data ratio":0.95
        }
}
```
- "normalize data" - boolean - describes if filtered data should be normalized
- "usable data ratio" - float - in range [0, 1] which indicates what proportion of the data should be used for identification, starting from sample 0 and ending at sample "usable data ratio"*length(filteredData)


## Partitioning parameters
```json
{
    "data partitioning":{
        "random partition":true,
    }    
}
```
- "random partition" - boolean - indicates if the data should be partitioned randomly into a training and testing subset. 


## Manoeuvre excitations
```json
{
    "manoeuvre excitations":{
        "isolate to regions of excitation":false,
        "excitation threshold":0.3
    }    
}
```
- "isolate to regions of excitation" - boolean - indicates if the data used for model identification should be restricted to regions of excitation only. 
- "excitation threshold" - float - describes the sensitivity of the excitation isolation algorithm to sensing excitations. If 0, then the whole dataset is considered an excitation (i.e. no isolation occurs)


## Identification parameters
```json
{
    "identification parameters":{
        "identify fx":true,
        "identify fy":true,
        "identify fz":true,
        "identify mx":true,
        "identify mz":true,
        "identify my":true,
        "prediction interval confidence level":0.95,
        "polynomial":{
            "regressor cap":5
        }
    }
}
```
- "identify fx" - boolean - indicates if a model of Fx (force along x-axis) should be identified
- "identify fy" - boolean - indicates if a model of Fy (force along y-axis) should be identified
- "identify fz" - boolean - indicates if a model of Fz (force along z-axis) should be identified
- "identify mx" - boolean - indicates if a model of Mx (moment about x-axis) should be identified
- "identify my" - boolean - indicates if a model of My (moment about y-axis) should be identified
- "identify mz" - boolean - indicates if a model of Mz (moment about z-axis) should be identified
- "prediction interval confidence level" - float - in range [0, 1] which describes the confidence level used to construct the prediction intervals
- "polynomial" - dict - polynomial model specific parameters
    - "regressor cap" - int - Upper (hard) limit of the number of regressors allowed in a polynomial model 


## Model saving parameters 
```json
{
    "saving models":{
        "save identified models":true,
        "save directory":null,
        "model ID":"MDL-MyDrone-001"
    }
}
```
- "save identified models" - boolean - indicates if identified model should be saved 
- "save directory" - string - directory in which the identified models should be saved
- "model ID" - string - Name of the model subfolder, in which the model will be saved
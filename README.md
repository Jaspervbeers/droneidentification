# droneIdentification

Pipeline to identify quadrotor models from raw flight data using the SysID pipeline. *Currently, only polynomial model identification is implemented.*




## Dependencies

Relies on the `SysID` module of the **sysidpipeline** (available from: [here](https://github.com/Jaspervbeers/sysidpipeline.git)). If you're unable to access the sysidpipeline repo, please send me an e-mail (j.j.vanbeers@tudelft.nl) with your request.

Once cloned, you need to specify the directory of the cloned library in `relativeImportLocations.json`. 




## Usage

Users primarily interact with the identification scripts through the `identificationConfig.json` file. Here, they may specify the raw data processing parameters (i.e. which data sets to import, filtering variables, normalization, data partitioning) and configure the identification. See the associated readme file for a detailed explanation of the customizable parameters. 
This configuration file is also saved alongside any identified models to keep track of the processing parameters used when identified said models. 

Another important file is the `file_log.csv`. This keeps track of the locations and filenames of the different raw data files, in addition to other useful comments, in a tabular format. See the associated readme file for a detailed explanation of the necessary entries. 

### Identification of polynomial quadrotor models
**Example**
Using sample data provided in the 'exampleData' folder, an example polynomial model of a quadrotor can be identified by running `exampleDronePolyModel.py` with `identificationConfig-example.json` (in the root folder) and `file_log-example.csv` (in the exampleData folder). 

**Identifying your own models**
To identify your own quadrotor models, users need to configure `identificationConfig.json` and their own `file_log.csv` (see template and readme under the templates folder). 

After configuring `identificationConfig.json` and `file_log.csv`, polynomial models of the quadrotor can be identified by running the `buildDronePolyModel.py` script. 

The polynomial candidates themselves can be configured in the `buildDronePolyModel.py` script itself. 
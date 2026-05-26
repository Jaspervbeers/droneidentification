import json
import importlib.util
import sys

# Open build instructions
with open('build_portable_model_instructions.json', 'r') as f:
    ins = json.load(f)

# Load json helper to build portable model
jsonhelperpath = ins['json helper path']

spec = importlib.util.spec_from_file_location('jsonModelHelper', f'{jsonhelperpath}/jsonModelHelper.py')
jsonModelHelper = importlib.util.module_from_spec(spec)

sys.modules['jsonModelHelper'] = jsonModelHelper
spec.loader.exec_module(jsonModelHelper)

modelPath = ins['path']
modelID = ins['model prefix']
models = ins['models']
savepath = ins['save path']
jsonModel = jsonModelHelper.jsonDroneModel(modelPath, modelID, savepath=savepath)
jsonModel.toPortable(jsonModelHelper.DronePolynomialModel, model_set=models)
'''
Script to inspect identified polynomial model structures and modify them (e.g. fuse terms of Fx and Fy)
'''

# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import os
import json
import sys
import numpy as np
import shutil

# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #

# This package relies on the system identification pipeline (sysidpipeline). 
with open('relativeImportLocations.json', 'r') as f:
    relativeLocs = json.load(f)

sys.path.append(relativeLocs['sysidpipeline'])
import SysID

# ================================================================================================================================ #
# Definitions
# ================================================================================================================================ #

def tabulateModels(Model):
    regressors = list(Model.TrainedModel['Model']['Regressors'])
    coefficients = list(Model.TrainedModel['Model']['Parameters'].__array__().reshape(-1))
    infoLog = Model.TrainedModel['Additional']['Log']['Info Log']
    addLog = [r for r in Model.TrainedModel['Additional']['Log']['Add Log'].values()]
    print('{:-^70}'.format(''))
    print('{:<30} {:>19} {:>19}'.format('Regressor', 'Coefficient [N]', 'R2'))
    rCounter = 1
    for r, c in zip(regressors, coefficients):
        r2 = 0
        if r in addLog:
            r2 = infoLog[rCounter]['R2'].__array__()[0][0]
            rCounter += 1
        print(f'{r:<30} {c:>20.3e} {r2:>20.3e}')
    print('{:-^70}\n'.format(''))


def _replaceRegressor(old, new, regressors):
    replaced = []
    for r in [i for i in regressors]:
        _r = r.replace(old, new)
        replaced.append(_r)
    return replaced


def replaceRegressors(oldRegs, newRegs, regressors):
    for o, c in zip(oldRegs, newRegs):
        regressors = _replaceRegressor(o, c, regressors)
    return regressors


def makePolyModel(fixedRegs, chosenRegs, x, y, hasBias = True, saveA = False):
    N = len(y)
    # Define surrogate model
    model = SysID.Model('stepwise_regression')
    # Define model attributes in expected structure
    m = {'Regressors':None, 'Parameters':None}
    log = {'Info Log':{}, 'Add Log':{}}
    for i, r in enumerate(chosenRegs):
        regs = fixedRegs + chosenRegs[:(i + 1)]
        A = getA(regs, x)
        params, pred = OLS(A, y.to_numpy().reshape(-1, 1))
        # sigma and inv_XtX are only needed for Prediction Interval estimation
        sigma = 1/(N-2)*np.sum(np.square(np.array(pred).reshape(-1) - y.to_numpy().reshape(-1) ))
        inv_XtX = np.linalg.inv(np.dot(np.transpose(A), A))
        m.update({'A':A})
        m.update({'Regressors':regs})
        m.update({'Parameters':params})
        m.update({'_sigma2':sigma})
        m.update({'_inv(XtX)':inv_XtX})
        m.update({'Has Bias':hasBias})
        R2 = getR2(pred, np.matrix(y).T)
        iLog = log['Info Log']
        iLog.update({i+1:{'R2':R2}})
        aLog = log['Add Log']
        aLog.update({i+1:r})
    if not saveA:
        m.update({'A':None})
    model.TrainedModel = {'Model':m, 
                          'Additional':{'Log':log}}
    model.CurrentModel = model.TrainedModel
    model.x_train = x
    model.y_train = y
    model.ModelState = 'Trained (Modified)'
    model.ModelStateHistory.append('Trained')
    model.ModelStateHistory.append(model.ModelState)
    return model

# ================================================================================================================================ #
# Main script
# ================================================================================================================================ #

ModelPath = "../../../Models/HDBeetle"
ModelID = "MDL-HDBeetle-NN-II-NGP003"

'''
Lateral forces (Fx and Fy)
'''
# Look at Fx and Fy model terms
FxModel = SysID.Model.load(os.path.join(ModelPath, ModelID, 'Fx'))
FyModel = SysID.Model.load(os.path.join(ModelPath, ModelID, 'Fy'))

print('Original Fx')
tabulateModels(FxModel)

print('Original Fy')
tabulateModels(FyModel)

getA = FxModel.UtilityFuncs._BuildRegressorMatrix # (regressors, data, hasBias = True)
OLS = FxModel.UtilityFuncs._OLS # (A, z, hasBias = True)
getR2 = FxModel.UtilityFuncs._CoeffOfDetermination_R2 # (pred, tar)

x_terms = ['[pitch]', 'u^', 'q^', 'v|', 'p|', 'x^', 'y|', 'q)']
y_terms = ['[roll]', 'v^', 'p^', 'u|', 'q|', 'y^', 'x|', 'p)']

FxRegressors = FxModel.TrainedModel['Model']['Regressors']
FyRegressors = FyModel.TrainedModel['Model']['Regressors']

# Get fixed regressors; first three are fixed (bias, u, sin[pitch])
FxFixed = FxRegressors[:3]
FyFixed = FyRegressors[:3]

# Get only candidate regressors
FxCandReg = FxRegressors[3:]
FyCandReg = FyRegressors[3:]

FxWithFyReg = replaceRegressors(y_terms, x_terms, FyCandReg)
FyWithFxReg = replaceRegressors(x_terms, y_terms, FxCandReg)

# Make new models, and see performance
FxWithFyModel = makePolyModel(FxFixed, FxWithFyReg, FxModel.x_train, FxModel.y_train, hasBias = FxModel.TrainedModel['Model']['Has Bias'])
FyWithFxModel = makePolyModel(FyFixed, FyWithFxReg, FyModel.x_train, FyModel.y_train, hasBias = FyModel.TrainedModel['Model']['Has Bias'])

# Print results, incl R2. 
print('Fx with Fy')
tabulateModels(FxWithFyModel)

print('Fy with Fx')
tabulateModels(FyWithFxModel)


# Create mapping of which regressors to choose
regs = [FxCandReg, FyCandReg]
mapping = [1, 1, 0, 0] # for regressor i, 0 = Take Fx regressor, 1 = Take Fy regressor, None = Take none

cands = []
for i in range(len(mapping)):
    if mapping[i] is not None:
        cands.append(regs[mapping[i]][i])

FxFinalCands = replaceRegressors(y_terms, x_terms, cands)
FyFinalCands = replaceRegressors(x_terms, y_terms, cands)

# Create new models
newID = ModelID + '-Fused'
newFx = makePolyModel(FxFixed, FxFinalCands, FxModel.x_train, FxModel.y_train, hasBias = FxModel.TrainedModel['Model']['Has Bias'])
newFy = makePolyModel(FyFixed, FyFinalCands, FyModel.x_train, FyModel.y_train, hasBias = FyModel.TrainedModel['Model']['Has Bias'])


'''
Lateral moments (Mx and My)
'''
# Look at Mx and My model terms
MxModel = SysID.Model.load(os.path.join(ModelPath, ModelID, 'Mx'))
MyModel = SysID.Model.load(os.path.join(ModelPath, ModelID, 'My'))

print('Original Mx')
tabulateModels(MxModel)

print('Original My')
tabulateModels(MyModel)

MxRegressors = MxModel.TrainedModel['Model']['Regressors']
MyRegressors = MyModel.TrainedModel['Model']['Regressors']

# Get fixed regressors; first three are fixed (bias, u, sin[pitch])
MxFixed = MxRegressors[:3]
MyFixed = MyRegressors[:3]

# Get only candidate regressors
MxCandReg = MxRegressors[3:]
MyCandReg = MyRegressors[3:]

MxWithMyReg = replaceRegressors(x_terms, y_terms, MyCandReg)
MyWithMxReg = replaceRegressors(y_terms, x_terms, MxCandReg)

# Make new models, and see performance
MxWithMyModel = makePolyModel(MxFixed, MxWithMyReg, MxModel.x_train, MxModel.y_train, hasBias = MxModel.TrainedModel['Model']['Has Bias'])
MyWithMxModel = makePolyModel(MyFixed, MyWithMxReg, MyModel.x_train, MyModel.y_train, hasBias = MyModel.TrainedModel['Model']['Has Bias'])

# Print results, incl R2. 
print('Mx with My')
tabulateModels(MxWithMyModel)

print('My with Mx')
tabulateModels(MyWithMxModel)


# Create mapping of which regressors to choose
regs = [MxCandReg, MyCandReg]
mapping = [0, 1, 0, 0] # for regressor i, 0 = Take Mx regressor, 1 = Take My regressor, None = Take none

cands = []
for i in range(len(mapping)):
    if mapping[i] is not None:
        cands.append(regs[mapping[i]][i])

MxFinalCands = replaceRegressors(x_terms, y_terms, cands)
MyFinalCands = replaceRegressors(y_terms, x_terms, cands)

# Create new models
newID = ModelID + '-Fused'
newMx = makePolyModel(MxFixed, MxFinalCands, MxModel.x_train, MxModel.y_train, hasBias = MxModel.TrainedModel['Model']['Has Bias'])
newMy = makePolyModel(MyFixed, MyFinalCands, MyModel.x_train, MyModel.y_train, hasBias = MyModel.TrainedModel['Model']['Has Bias'])



'''
Save combined models
'''
shutil.copytree(os.path.join(ModelPath, ModelID), os.path.join(ModelPath, newID))
newFx.save(os.path.join(ModelPath, newID, 'Fx'))
newFy.save(os.path.join(ModelPath, newID, 'Fy'))
newMx.save(os.path.join(ModelPath, newID, 'Mx'))
newMy.save(os.path.join(ModelPath, newID, 'My'))
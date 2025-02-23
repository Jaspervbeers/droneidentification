import numpy as np
import json
import os
import subprocess
import dill as pkl
from scipy.io import savemat
import sys

variableMapper = {
    'bias':'1',
    'sin[pitch]':'sin(pitch)',
    'cos[pitch]':'cos(pitch)',
    'sin[roll]':'sin(roll)',
    'cos[roll]':'cos(roll)',
    'sin[yaw]':'sin(yaw)',
    'cos[yaw]':'cos(yaw)'
}

def getSignMask(droneParams):
    # Define dictionary which maps the rotor numbers to their position on the quadrotor (e.g. rotor 4 is located 'front left')
    invRotorConfig = {v:k for k, v in droneParams['rotor configuration'].items()}
    # Extract ditcionary which maps the yaw sign to CW (clockwise) and CCW (counterclockwise) rotation of the rotors
    r_sign = droneParams['r_sign']
    # Create a toggle to parse through rotor mapping using text cues (e.g. front, left, etc.)
    signMap = {True:1, False:-1}
    # Create a matrix mask describing the signs of each rotor on each of the control moments
    # -> A matrix which describes how the rotor configuration produces rotations of the quadrotor and the sign therein
    signMask = np.array([[signMap[invRotorConfig[1].endswith('left')], signMap[invRotorConfig[2].endswith('left')], signMap[invRotorConfig[3].endswith('left')], signMap[invRotorConfig[4].endswith('left')]],
                        [signMap[invRotorConfig[1].startswith('front')], signMap[invRotorConfig[2].startswith('front')], signMap[invRotorConfig[3].startswith('front')], signMap[invRotorConfig[4].startswith('front')]],
                        r_sign[droneParams['rotor 1 direction']]*np.array([1, -1, -1, 1])]).T
    return signMask.copy()


def __writeEq(polynomial, coefficients):
    stringEq = ""
    for c, p in zip(coefficients.__array__().reshape(-1), polynomial):
        stringEq += str(c) + '*' + p + '+'
    return stringEq[:-1]


def __parsePoly(m):
    _vars = []
    _hasAbs = []
    for reg in m.regressors:
        for r in np.array(reg.RPN)[reg.variableIndices]:
            if r.startswith('|'):
                _r = r.split('|')[1]
                if _r not in _hasAbs:
                    _hasAbs.append(_r)
            elif r.startswith('sin') or r.startswith('cos'):
                _r = r.split('[')[-1].split(']')[0]
            else:
                _r = r
            if _r not in _vars:
                _vars.append(_r)

    poly = [p for p in m.polynomial]
    for i, p in enumerate(m.polynomial):
        for _x in _hasAbs:
            if r'|' + f'{_x}' + r'|' in p:
                p = p.replace(r'|' + f'{_x}' r'|', f'abs({_x})')
            if f'abs({_x})/{_x}' in p:
                p = p.replace(f'abs({_x})/{_x}', f'sign({_x})')
        for _v in variableMapper.keys():
            if _v in p:
                p = p.replace(_v, variableMapper[_v])
        poly[i] = p
    return poly, _vars


def _toFMFile(models, key, path = None):
    if path is None:
        path = os.getcwd()
    filename = os.path.join(path, f'get_{key}.m')
    m = models[key]
    poly, _vars = __parsePoly(m)

    with open(filename, 'w') as f:
        f.write(f'function [{key}] = get_{key}({",".join(_vars)})')
        f.write('\n')
        f.write('\t' + f'{key}= ' + f'{__writeEq(poly, m.coefficients)};')
        f.write('\n')

    return _vars


def _toDiffSys(DiffSys, DiffSysInputVec, path = None):
    if path is None:
        path = os.getcwd()  
    
    Mx = DiffSys._DfDx_DfDu_Variable[0]
    My = DiffSys._DfDx_DfDu_Variable[1]
    Mz = DiffSys._DfDx_DfDu_Variable[2]
    
    _vars = []

    # A(x, u) MATRIX:
    # A11 -> dMx(x, u)/dp
    _poly_Mx_dfdx_p, _vars_Mx_dfdx_p = __parsePoly(Mx['dfdx']['p'])
    _poly_p_dot_dp, _vars_p_dot_dp = __parsePoly(DiffSys.p_dot_dp)
    a11 = DiffSys.invIv[0][0]
    for p in _vars_Mx_dfdx_p + _vars_p_dot_dp:
        if p not in _vars:
            _vars.append(p)
    # A12 -> dMx(x, u)/dq
    _poly_Mx_dfdx_q, _vars_Mx_dfdx_q = __parsePoly(Mx['dfdx']['q'])
    _poly_p_dot_dq, _vars_p_dot_dq = __parsePoly(DiffSys.p_dot_dq)
    a12 = DiffSys.invIv[0][1]
    for p in _vars_Mx_dfdx_q + _vars_p_dot_dq:
        if p not in _vars:
            _vars.append(p)
    # A13 -> dMx(x, u)/dr
    _poly_Mx_dfdx_r, _vars_Mx_dfdx_r = __parsePoly(Mx['dfdx']['r'])
    _poly_p_dot_dr, _vars_p_dot_dr = __parsePoly(DiffSys.p_dot_dr)
    a13 = DiffSys.invIv[0][2]
    for p in _vars_Mx_dfdx_r + _vars_p_dot_dr:
        if p not in _vars:
            _vars.append(p)
    # A21 -> dMy(x, u)/dp
    _poly_My_dfdx_p, _vars_My_dfdx_p = __parsePoly(My['dfdx']['p'])
    _poly_q_dot_dp, _vars_q_dot_dp = __parsePoly(DiffSys.q_dot_dp)
    a21 = DiffSys.invIv[1][0]
    for p in _vars_My_dfdx_p + _vars_q_dot_dp:
        if p not in _vars:
            _vars.append(p)
    # A22 -> dMy(x, u)/dq
    _poly_My_dfdx_q, _vars_My_dfdx_q = __parsePoly(My['dfdx']['q'])
    _poly_q_dot_dq, _vars_q_dot_dq = __parsePoly(DiffSys.q_dot_dq)
    a22 = DiffSys.invIv[1][1]
    for p in _vars_My_dfdx_q + _vars_q_dot_dq:
        if p not in _vars:
            _vars.append(p)
    # A23 -> dMy(x, u)/dr
    _poly_My_dfdx_r, _vars_My_dfdx_r = __parsePoly(My['dfdx']['r'])
    _poly_q_dot_dr, _vars_q_dot_dr = __parsePoly(DiffSys.q_dot_dr)
    a23 = DiffSys.invIv[1][2]
    for p in _vars_My_dfdx_r + _vars_q_dot_dr:
        if p not in _vars:
            _vars.append(p)
    # A31 -> dMz(x, u)/dp
    _poly_Mz_dfdx_p, _vars_Mz_dfdx_p = __parsePoly(Mz['dfdx']['p'])
    _poly_r_dot_dp, _vars_r_dot_dp = __parsePoly(DiffSys.r_dot_dp)
    a31 = DiffSys.invIv[2][0]
    for p in _vars_Mz_dfdx_p + _vars_r_dot_dp:
        if p not in _vars:
            _vars.append(p)
    # A32 -> dMz(x, u)/dq
    _poly_Mz_dfdx_q, _vars_Mz_dfdx_q = __parsePoly(Mz['dfdx']['q'])
    _poly_r_dot_dq, _vars_r_dot_dq = __parsePoly(DiffSys.r_dot_dq)
    a32 = DiffSys.invIv[2][1]
    for p in _vars_Mz_dfdx_q + _vars_r_dot_dq:
        if p not in _vars:
            _vars.append(p)
    # A33 -> dMz(x, u)/dr
    _poly_Mz_dfdx_r, _vars_Mz_dfdx_r = __parsePoly(Mz['dfdx']['r'])
    _poly_r_dot_dr, _vars_r_dot_dr = __parsePoly(DiffSys.r_dot_dr)
    a33 = DiffSys.invIv[2][2]
    for p in _vars_Mz_dfdx_r + _vars_r_dot_dr:
        if p not in _vars:
            _vars.append(p)           

    # B(x, u) MATRIX:
    [up, uq, ur] = DiffSysInputVec
    # B11 -> dMx(x, u)/du_p
    _poly_Mx_dfdu_up, _vars_Mx_dfdu_up = __parsePoly(Mx['dfdu'][up])
    for p in _vars_Mx_dfdu_up:
        if p not in _vars:
            _vars.append(p)          
    # B12 -> dMx(x, u)/du_q
    _poly_Mx_dfdu_uq, _vars_Mx_dfdu_uq = __parsePoly(Mx['dfdu'][uq])
    for p in _vars_Mx_dfdu_uq:
        if p not in _vars:
            _vars.append(p)    
    # B13 -> dMx(x, u)/du_r
    _poly_Mx_dfdu_ur, _vars_Mx_dfdu_ur = __parsePoly(Mx['dfdu'][ur])
    for p in _vars_Mx_dfdu_ur:
        if p not in _vars:
            _vars.append(p)   
    # B21 -> dMy(x, u)/du_p
    _poly_My_dfdu_up, _vars_My_dfdu_up = __parsePoly(My['dfdu'][up])
    for p in _vars_My_dfdu_up:
        if p not in _vars:
            _vars.append(p)          
    # B22 -> dMy(x, u)/du_q
    _poly_My_dfdu_uq, _vars_My_dfdu_uq = __parsePoly(My['dfdu'][uq])
    for p in _vars_My_dfdu_uq:
        if p not in _vars:
            _vars.append(p)    
    # B23 -> dMy(x, u)/du_r
    _poly_My_dfdu_ur, _vars_My_dfdu_ur = __parsePoly(My['dfdu'][ur])
    for p in _vars_My_dfdu_ur:
        if p not in _vars:
            _vars.append(p)
    # B31 -> dMz(x, u)/du_p
    _poly_Mz_dfdu_up, _vars_Mz_dfdu_up = __parsePoly(Mz['dfdu'][up])
    for p in _vars_Mz_dfdu_up:
        if p not in _vars:
            _vars.append(p)          
    # B32 -> dMz(x, u)/du_q
    _poly_Mz_dfdu_uq, _vars_Mz_dfdu_uq = __parsePoly(Mz['dfdu'][uq])
    for p in _vars_Mz_dfdu_uq:
        if p not in _vars:
            _vars.append(p)    
    # B33 -> dMz(x, u)/du_r
    _poly_Mz_dfdu_ur, _vars_Mz_dfdu_ur = __parsePoly(Mz['dfdu'][ur])
    for p in _vars_Mz_dfdu_ur:
        if p not in _vars:
            _vars.append(p)               

    with open(os.path.join(path, 'get_PQRDiffSys.m'), 'w') as f:
        f.write(f'function [Axu, Bxu] = get_PQRDiffSys(droneInputs)')
        f.write('\t%' + 'Function to calculate the quadrotor velocity form matrices at a point (x, u)')
        f.write('\t%' + 'For a quadrotor system defined by x_dot = f(x, u), the velocity form is:')
        f.write('\t%' + '\tx_ddot = A(x, u)x_dot + B(x, u)u_dot')
        f.write('\t%' + 'This system is linear w.r.t x_dot and u_dot at a point (x, u)')
        f.write('\n\n')
        f.write('\t% Pre-allocate the matrices\n')
        f.write('\t' + 'Axu = zeros(3);\n')
        f.write('\t' + 'Bxu = zeros(3);\n')
        f.write('\n')
        f.write('\t' + r'% Extract variables' + '\n')
        for _v in _vars:
            f.write('\t' + f'{_v} = droneInputs.{_v};\n')
        f.write('\n')

        f.write('\t% Compute entries of Axu and Bxu\n')
        f.write('\t% A(x, u)\n')
        f.write('\tAxu(1, 1) = ' 
                + f'{a11}*(' + __writeEq(_poly_Mx_dfdx_p, Mx['dfdx']['p'].coefficients) + ') + ' 
                + __writeEq(_poly_p_dot_dp, DiffSys.p_dot_dp.coefficients) + ';\n')
        f.write('\tAxu(1, 2) = ' 
                + f'{a12}*(' + __writeEq(_poly_Mx_dfdx_q, Mx['dfdx']['q'].coefficients) + ') + ' 
                + __writeEq(_poly_p_dot_dq, DiffSys.p_dot_dq.coefficients) + ';\n')
        f.write('\tAxu(1, 3) = ' 
                + f'{a13}*(' + __writeEq(_poly_Mx_dfdx_r, Mx['dfdx']['r'].coefficients) + ') + ' 
                + __writeEq(_poly_p_dot_dr, DiffSys.p_dot_dr.coefficients) + ';\n')
        f.write('\tAxu(2, 1) = ' 
                + f'{a21}*(' + __writeEq(_poly_My_dfdx_p, My['dfdx']['p'].coefficients) + ') + ' 
                + __writeEq(_poly_q_dot_dp, DiffSys.q_dot_dp.coefficients) + ';\n')
        f.write('\tAxu(2, 2) = ' 
                + f'{a22}*(' + __writeEq(_poly_My_dfdx_q, My['dfdx']['q'].coefficients) + ') + ' 
                + __writeEq(_poly_q_dot_dq, DiffSys.q_dot_dq.coefficients) + ';\n')
        f.write('\tAxu(2, 3) = ' 
                + f'{a23}*(' + __writeEq(_poly_My_dfdx_r, My['dfdx']['r'].coefficients) + ') + ' 
                + __writeEq(_poly_q_dot_dr, DiffSys.q_dot_dr.coefficients) + ';\n')
        f.write('\tAxu(3, 1) = ' 
                + f'{a31}*(' + __writeEq(_poly_Mz_dfdx_p, Mz['dfdx']['p'].coefficients) + ') + ' 
                + __writeEq(_poly_r_dot_dp, DiffSys.r_dot_dp.coefficients) + ';\n')
        f.write('\tAxu(3, 2) = ' 
                + f'{a32}*(' + __writeEq(_poly_Mz_dfdx_q, Mz['dfdx']['q'].coefficients) + ') + ' 
                + __writeEq(_poly_r_dot_dq, DiffSys.r_dot_dq.coefficients) + ';\n')
        f.write('\tAxu(3, 3) = ' 
                + f'{a33}*(' + __writeEq(_poly_Mz_dfdx_r, Mz['dfdx']['r'].coefficients) + ') + ' 
                + __writeEq(_poly_r_dot_dr, DiffSys.r_dot_dr.coefficients) + ';\n')
        f.write('\n')

        f.write('\t% B(x, u)\n')
        f.write('\tBxu(1, 1) = ' + f'{a11}*(' + __writeEq(_poly_Mx_dfdu_up, Mx['dfdu'][up].coefficients) + ');\n')
        f.write('\tBxu(1, 2) = ' + f'{a12}*(' + __writeEq(_poly_Mx_dfdu_uq, Mx['dfdu'][uq].coefficients) + ');\n')
        f.write('\tBxu(1, 3) = ' + f'{a13}*(' + __writeEq(_poly_Mx_dfdu_ur, Mx['dfdu'][ur].coefficients) + ');\n')
        f.write('\tBxu(2, 1) = ' + f'{a21}*(' + __writeEq(_poly_My_dfdu_up, My['dfdu'][up].coefficients) + ');\n')
        f.write('\tBxu(2, 2) = ' + f'{a22}*(' + __writeEq(_poly_My_dfdu_uq, My['dfdu'][uq].coefficients) + ');\n')
        f.write('\tBxu(2, 3) = ' + f'{a23}*(' + __writeEq(_poly_My_dfdu_ur, My['dfdu'][ur].coefficients) + ');\n')
        f.write('\tBxu(3, 1) = ' + f'{a31}*(' + __writeEq(_poly_Mz_dfdu_up, Mz['dfdu'][up].coefficients) + ');\n')
        f.write('\tBxu(3, 2) = ' + f'{a32}*(' + __writeEq(_poly_Mz_dfdu_uq, Mz['dfdu'][uq].coefficients) + ');\n')
        f.write('\tBxu(3, 3) = ' + f'{a33}*(' + __writeEq(_poly_Mz_dfdu_ur, Mz['dfdu'][ur].coefficients) + ');\n')

    return None

'''
Initialize file and generate standalone models
'''
# Load config file
with open('modelConfig.json', 'r') as f:
    modelConfig = json.load(f)

# Write to standaloneConfig in parent directory
standaloneConfig = {}
standaloneConfig.update({'model path':modelConfig['model path']})
standaloneConfig.update({'model ID':modelConfig['model ID']})
standaloneConfig.update({'droneConfig path':modelConfig['droneConfig path']})
standaloneConfig.update({'droneConfig name':modelConfig['droneConfig name']})
standaloneConfig.update({'make moment DiffSys':modelConfig['make moment DiffSys']})
standaloneConfig.update({'DiffSys x':['p', 'q', 'r']}) # TODO: Parameterize this
standaloneConfig.update({'DiffSys u':modelConfig['DiffSys u']})

with open('../standaloneConfig.json', 'r') as f:
    originalStandaloneConfig = json.load(f)

with open('../standaloneConfig.json', 'w') as f:
    json.dump(standaloneConfig, f, indent = 4)

# Load models and export to standlone versions
try:
    subprocess.call([sys.executable, 'makeStandalonePolyModelV3.py'], cwd = os.path.join(os.getcwd(), '../'))
except Exception as e:
    with open('../standaloneConfig.json', 'w') as f:
        json.dump(originalStandaloneConfig, f, indent = 4)
    raise e

with open('../standaloneConfig.json', 'w') as f:
    json.dump(originalStandaloneConfig, f, indent = 4)


'''
Load polynomial F, M models and port to MATLAB
'''
# Load standalone models
modelPath = os.path.join('../' + modelConfig['model path'], modelConfig['model ID'], 'standaloneDiff')
models = {}
toLoad = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
for m in toLoad:
    with open(os.path.join(modelPath, modelConfig['model ID'] + f'-{m}.pkl'), 'rb') as f:
        _m = pkl.load(f)
    models.update({m:_m})


print('[ INFO ] Converting model...')
# Create .m model files
matModelsPath = os.path.join('models', modelConfig['model ID'])
if not os.path.exists(matModelsPath):
    os.makedirs(matModelsPath)

_mVars = {}
for m in toLoad:
    _vars = _toFMFile(models, m, path = matModelsPath)
    _mVars.update({m:_vars})


# Create droneParams and modelParams files
modelParams = {}
modelParams.update({'isNormalized':models[m].isNormalized})
modelParams.update({'hasGravity':models[m].hasGravity})
modelParams.update({'VINFLAG':models[m].VINFLAG})
modelParams.update({'g':models[m].droneParams['g']})
modelParams.update({'rho':models[m].droneParams['rho']})
savemat(os.path.join(matModelsPath, 'modelParams.mat'), modelParams)

droneParams = {}
droneParams.update({'R':models[m].droneParams['R']})
droneParams.update({'b':models[m].droneParams['b']})
droneParams.update({'Iv':models[m].droneParams['Iv']})
rotorConfig = {}
rotorConfig.update({'frontLeft':models[m].droneParams['rotor configuration']['front left']})
rotorConfig.update({'frontRight':models[m].droneParams['rotor configuration']['front right']})
rotorConfig.update({'aftLeft':models[m].droneParams['rotor configuration']['aft left']})
rotorConfig.update({'aftRight':models[m].droneParams['rotor configuration']['aft right']})
droneParams.update({'rotorConfig':rotorConfig})
droneParams.update({'r1Sign':models[m].droneParams['r_sign'][models[m].droneParams['rotor 1 direction']]})
droneParams.update({'r1Dir':models[m].droneParams['rotor 1 direction']})
droneParams.update({'minRPM':models[m].droneParams['idle RPM']})
droneParams.update({'maxRPM':models[m].droneParams['max RPM']})
droneParams.update({'mass':models[m].droneParams['m']})
droneParams.update({'wHover':models[m].droneParams['wHover (rad/s)']})
droneParams.update({'wHoverERPM':models[m].droneParams['wHover (eRPM)']})
droneParams.update({'nRotors':models[m].droneParams['number of rotors']})
droneParams.update({'taus':models[m].droneParams['taus']})
droneParams.update({'signMask':getSignMask(models[m].droneParams).T})
droneParams.update({'simpleModel':models[m].droneParams['simpleModel']})
droneParams.update({'kappaFz':models[m].droneParams['simpleModel']['kappaFz_w_2']*(2*np.pi/60)**2})
M2OmegaMapping = np.ones((3, 4))
M2OmegaMapping[0, :] = droneParams['simpleModel']['kappaMx_w_2']*(2*np.pi/60)**2
M2OmegaMapping[1, :] = droneParams['simpleModel']['kappaMy_w_2']*(2*np.pi/60)**2
M2OmegaMapping[2, :] = droneParams['simpleModel']['kappaMz_w_2']*(2*np.pi/60)**2
M2OmegaMapping = M2OmegaMapping*droneParams['signMask']
droneParams.update({'M2OmegaMapping':M2OmegaMapping})
savemat(os.path.join(matModelsPath, 'droneParams.mat'), droneParams)


# Create getFM() file
with open(f'models/{modelConfig["model ID"]}/getFM.m', 'w') as f:
    f.write('function [F, M] = getFM(state, omega, modelParams, droneParams)\n')
    f.write('\t' + r'% Model ID: ' + f'{modelConfig["model ID"]}' + '\n')
    f.write('\t' + r'% Model specific parameters should be defined in the workspace' + '\n')
    f.write('\t' + r'% -> Make sure <modelParams> and <droneParams> are loaded for ' + f'{modelConfig["model ID"]}' + '\n')
    f.write('\t' + r'% Otherwise, run model_init(' + f'{modelConfig["model ID"]})' + '\n\n\n')
    f.write('\t' + r'% Get input vector for quadrotor models' + '\n')
    f.write('\t' + 'droneInputs = getDroneInputs(state, omega, modelParams, droneParams);\n\n')
    f.write('\t' + r'% Compute forces and moments' + '\n')
    for m in toLoad:
        f.write('\t' + f'{m} = get_{m}({", ".join(["droneInputs." + _p for _p in _mVars[f"{m}"]])});' + '\n')
    
    f.write('\n')
    f.write('\t' + r'% Collect forces, F, and moments, M' + '\n')
    f.write('\tF = [' + f'{",".join([_f for _f in toLoad if _f.lower().startswith("f")])}' + '] ./ droneInputs.Fden;\n')
    f.write('\tM = [' + f'{",".join([_f for _f in toLoad if _f.lower().startswith("m")])}' + '] ./ droneInputs.Mden;\n')

print('[ INFO ] Model conversion successful!')


'''
Load and port DiffSys models, if present
'''
if modelConfig['make moment DiffSys']:
    print('[ INFO ] Option make moment DiffSys selected.')
    try:
        # Load DiffSys
        with open(os.path.join(modelPath, 'PQR-DiffSys.pkl'), 'rb') as f:
            DiffSys = pkl.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not find DiffSys information associated with model: {modelConfig["model ID"]}')
    
    _toDiffSys(DiffSys, modelConfig['DiffSys u'], path = matModelsPath)
    print('[ INFO ] DiffSys conversion successful!')
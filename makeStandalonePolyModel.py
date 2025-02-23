'''
Script to create a standalone barebones polynomial model file from a trained SysID polynomial model, without the local dependencies of the SysID class.
Resultant models are much smaller in size, since they only contain information necessary for making predictions. However, including functionality for
calculating prediction intervals will increase filesize proportional to training data size, due to required knowledge on observed training data. 

Written by: Jasper van Beers
Contact: j.j.vanbeers@tudelft.nl
Date: 09-02-2022
'''
# ================================================================================================================================ #
# Imports
# ================================================================================================================================ #
import os
import subprocess
import dill as pickle
from numpy import matrix, ones, hstack, dot, add, subtract, divide, multiply, power, nan, array, where, arange, isnan, roots, isclose, pi, abs, zeros, vstack, cos, sin, sum
from re import sub as reSub
from io import StringIO
from tokenize import generate_tokens
from numpy.linalg import LinAlgError, inv
from json import load as jLoad
from json import dump as jDump

# This package relies on the system identification pipeline (sysidpipeline). 
with open('relativeImportLocations.json', 'r') as f:
    relativeLocs = jLoad(f)

from sys import path as sysPath
sysPath.append(relativeLocs['sysidpipeline'])
import SysID


# ================================================================================================================================ #
# Classes
# ================================================================================================================================ #
'''
Class bundles SysID polynomial model into an independent object that can be used for predictions without the need for local modules.
Predictions are made using the .predict(x) method, where x holds the same structure as the data used as model inputs for training. 
'''
# TODO: Add ReadMe which documents what is expected of x, or reference to other document/file/script which outlines this. 
class PolynomialModel:
    
    # def __init__(self, config, isNormalized = True, hasGravity = False):
    #     # Keep track of important model metadata
    #     self.config = config
    #     self.isNormalized = isNormalized
    #     self.hasGravity = hasGravity
    def __init__(self, config):
        # Keep track of important model metadata
        self.config = config
        # Import necessary functions
        self.npMatrix = matrix
        self.npOnes = ones
        self.npHstack = hstack
        self.npDot = dot
        self.npAdd = add
        self.npSubtract = subtract
        self.npDivide = divide
        self.npMultiply = multiply
        self.npPower = power
        self.npNan = nan
        self.npWhere = where
        self.npArray = array
        self.npArange = arange
        self.npIsnan = isnan
        self.StringIO = StringIO
        self.generate_tokens = generate_tokens
        self.isExtracted = False
        self._usePI = False
        self.coefficients = None
        self.polynomial = None
        return None


    def extractModel(self, sysIDModel, predictionIntervals = False, forceExtraction = False):
        if not self.isExtracted or forceExtraction: 
            self.coefficients = self._getCoefficients(sysIDModel)
            self.polynomial = self._getPolynomial(sysIDModel)
            if predictionIntervals:
                self._usePI = True
                # self._inv_XtX = sysIDModel.TrainedModel['Model']['_inv(XtX)']
                # self._s2 = sysIDModel.TrainedModel['Model']['_sigma2']
                self._inv_XtX = sysIDModel.CurrentModel['Model']['_inv(XtX)']
                self._s2 = sysIDModel.CurrentModel['Model']['_sigma2']
            self.makeRegressors()
        else:
            raise AttributeError('A polynomial model has already been extracted. Set forceExtraction = True extract anyway (will overwrite existing polynomial).')
        return None


    def predict(self, x):
        A = self._BuildRegressorMatrix(x, hasBias = ('bias' in self.polynomial))
        if self._usePI:
            pred = A*self.coefficients
            var = pred.copy()
            AT = A.T
            for i in range(len(var)):
                var[i] = self._s2 + self._s2*self.npDot(self.npDot(A[i, :], self._inv_XtX), AT[:, i])
            return A*self.coefficients, var
        else:
            return A*self.coefficients

    
    def _getCoefficients(self, sysIDModel):
        # return sysIDModel.TrainedModel['Model']['Parameters']
        return sysIDModel.CurrentModel['Model']['Parameters']


    def _getPolynomial(self, sysIDModel):
        # return sysIDModel.TrainedModel['Model']['Regressors']
        return sysIDModel.CurrentModel['Model']['Regressors']
    

    def _BuildRegressorMatrix(self, data, hasBias = True):
        # Pre-allocate matrix
        N = len(self.regressors)
        regMat = self.npMatrix(self.npOnes((data.shape[0], N)))
        # Fill in matrix using regressors and data
        for i, reg in enumerate(self.regressors):
            regMat[:, i] = self.npMatrix(reg.resolve(data)).T
        # Pre-pend the bias vector, if present
        if hasBias:
            biasVec = self.npMatrix(self.npOnes((data.shape[0], 1)))
            regMat = self.npHstack((biasVec, regMat))
        return regMat


    def makeRegressors(self):
        parsing = self.Parser()
        self.regressors = []
        for p in self.polynomial:
            if p != 'bias':
                p_RPN = parsing.parse(p)
                reg = self.Regressor(p_RPN)
                self.regressors.append(reg)


    # Class to convert string equations to postfix form (i.e. Reverse Polish Notation - RPN) which can be easily interpretted from left to right.
    class Parser:
        # Initialize the RPN (output) stack and operator stack (which handles order of operations prior to addition in the output stack)
        def __init__(self):
            self.sub = reSub
            self.operatorStack = []
            self.outputStack = []
            # Define allowable operators, along with their precedence and associativity
            self.operatorInfo = {
                '^':{'precedence':4,
                        'associativity':'R'},
                '*':{'precedence':3,
                        'associativity':'L'},
                '/':{'precedence':3,
                        'associativity':'L'},
                '+':{'precedence':2,
                        'associativity':'L'},
                '-':{'precedence':2,
                        'associativity':'L'}                   
            }

        # Main parsing function, which converts an input string equation into RPN form.
        def parse(self, inputString):
            self.refresh()
            self.tokens = self.tokenize(inputString)
            RPN = self.shuntYard(self.tokens)
            if len(RPN) == 0:
                return [inputString]
            else:
                return RPN

        # Empty (any) previously parsed information, and reset for parsing new strings
        def refresh(self):
            self.operatorStack = []
            self.outputStack = []

        # Convert input string into tokens, sliced by the operators. 
        def tokenize(self, inputString):
            # remove spaces in string
            cleanString = self.sub(r'\s+', "", inputString)
            # Convert to list of characters to isolate operators and brackets
            chars = list(cleanString)
            # Tokens
            tokens = []
            token = ""
            while len(chars) != 0:
                char = chars.pop(0)
                if char in self.operatorInfo.keys() or char in ['(', ')']:
                    if token != "":
                        tokens.append(token)
                    tokens.append(char)
                    token = ""
                else:
                    token += char
                if len(chars) == 0 and token != "":
                    tokens.append(token)
            return tokens

        # Apply the Shunting-yard algorithm to convert the tokens into RPN form. 
        def shuntYard(self, tokens):
            while len(tokens) != 0:
                token = tokens.pop(0)
                # Check if token is a known operator
                if token in self.operatorInfo.keys():
                    # Check operator priority
                    if not len(self.operatorStack) == 0:
                        sorting = True
                        while sorting:
                            push = False
                            # Check top of operator stack for brackets
                            if self.operatorStack[-1] not in ["(", ")"]:
                                if self.operatorInfo[self.operatorStack[-1]]['precedence'] > self.operatorInfo[token]['precedence']:
                                    # top operator has greater priority
                                    push = True
                                elif self.operatorInfo[self.operatorStack[-1]]['precedence'] == self.operatorInfo[token]['precedence']:
                                    if self.operatorInfo[self.operatorStack[-1]]['associativity'] == 'L':
                                        push = True
                            sorting = push and self.operatorStack[-1] != '('
                            if sorting:
                                self.outputStack.append(self.operatorStack.pop())
                            if len(self.operatorStack) == 0:
                                sorting = False
                    self.operatorStack.append(token)
                elif token == "(":
                    self.operatorStack.append(token)
                elif token == ")":
                    #Add operations to stack while in brackets
                    while True:
                        if len(self.operatorStack) == 0:
                            break
                        if self.operatorStack[-1] == "(":
                            break
                        self.outputStack.append(self.operatorStack.pop())
                    if len(self.operatorStack) != 0 and self.operatorStack[-1] == "(":
                        self.operatorStack.pop()
                else:
                    self.outputStack.append(token)
            self.outputStack.extend(self.operatorStack[::-1])
            return self.outputStack

    # Class which handles regressor evaluations. The regressor structure is stored upon initialization for efficiency. 
    class Regressor:
        def __init__(self, regressorRPN):
            self.RPN = regressorRPN
            self.numberIndices = [i for i, v in enumerate(regressorRPN) if self.isFloat(v)]
            self.knownOperators = {'+':add, '-':subtract, '/':divide, '*':multiply, '^':power}
            self.operatorIndices = [i for i, v in enumerate(regressorRPN) if v in self.knownOperators.keys()]
            self.invVariableIndices = self.numberIndices + self.operatorIndices
            self.npArange = arange
            self.npArray = array
            if len(self.invVariableIndices):
                self.variableIndices = [i for i in self.npArange(0, len(regressorRPN)) if i not in self.invVariableIndices]
            else:
                self.variableIndices = self.npArange(0, len(regressorRPN))

        def resolve(self, Data):
            # First convert RPN string into purely numbers
            RPN = self.RPN.copy()
            RPNStr = self.RPN.copy()
            for idx in self.variableIndices:
                RPN[idx] = Data[self.RPN[idx]]
            # Evaluate RPN expression
            stack = []
            if len(RPN) > 1:
                while len(RPN) > 0:
                    token = RPN.pop(0)
                    tokenStr = RPNStr.pop(0)
                    if tokenStr not in self.knownOperators.keys():
                        stack.append(token)
                    else:
                        b = self.npArray(stack.pop(), dtype=float)
                        a = self.npArray(stack.pop(), dtype=float)
                        stack.append(self.knownOperators[token](a, b))
                if len(stack) != 1:
                    raise ValueError('There are unaccounted variables in the RPN regressor stack. Please check regressor operations are parsed correctly.')
                else:
                    return stack[0]
            else:
                return self.npArray(RPN[0], dtype=float)

        def isFloat(self, string):
            try:
                float(string)
                return True
            except ValueError:
                return False



'''
Class which augments PolynomialModel with functions to transform and normalize the state and rotor speeds of the quadrotor for
use with the polynomial model. As such processing is system specific (e.g. for quadrotors), such processing is left out of the 
general PolynomialModel class (which assumes that the inputs are structured and processed the same as done during training).
Here, we further assume that the state has format:
State = 2-D array with columns (roll, pitch, yaw, u, v, w, p, q, r) in that order. Rows give the observed samples.
    - roll = Roll angle, in radians
    - pitch = Pitch angle, in radians
    - yaw = Yaw angle, in radians
    - u = Body linear velocity along x-axis, in m/s
    - v = Body linear velocity along y-axis, in m/s
    - w = Body linear velocity along z-axis, in m/s
    - p = (Roll rate) Body rotational velocity about x-axis, in rad/s
    - q = (Pitch rate) Body rotational velocity about y-axis, in rad/s
    - r = (Yaw rate) Body rotational velocity about z-axis, in rad/s
rotorSpeeds = 2-D array with columns (w1, w2, w3, w4) in that order. Rows give observed samples
    - wi = rotational speed of rotor i (1, 2, 3, 4) in erpm (electronic rpm, is equivalent to rpm scaled by number of rotor poles and thus depends on rotor)
'''      
class DronePolynomialModel(PolynomialModel):

    def __init__(self, droneConfigFilePath, isNormalized = True, hasGravity = False, usesVIN = True, metadata = None):
        self.jsonLoad = jLoad
        with open(droneConfigFilePath, 'r') as f:
            configData = self.jsonLoad(f)
        PolynomialModel.__init__(self, configData)
        # Keep track of important metadata
        self.isNormalized = isNormalized
        self.hasGravity = hasGravity
        self.VINFLAG = usesVIN
        # Other funcs        
        self.npRoots = roots
        self.npIsclose = isclose
        self.npPi = pi
        self.npAbs = abs
        self.npZeros = zeros
        self.npVstack = vstack
        self.npCos = cos
        self.npSin = sin
        self.npSum = sum
        self.LinAlgError = LinAlgError
        # self.pdDataFrame = DataFrame
        self.droneParams = {'R':float(configData['rotor radius']),
                   'b':float(configData['b']),
                   'Iv':self.npArray(self.npMatrix(configData['moment of inertia'])), 
                   'rotor configuration':configData['rotor config'],
                   'rotor 1 direction':configData['rotor1 rotation direction'],
                   'idle RPM':float(configData['idle RPM']),
                   'max RPM':float(configData['max RPM']),
                   'm':float(metadata['additional info']['droneMass']),
                   'wHover (rad/s)':metadata['additional info']['hover omega (rad/s)'],
                   'wHover (eRPM)':metadata['additional info']['hover omega (eRPM)'],
                   'g':9.81,
                   'rho':1.225,
                   'r_sign':{'CCW':-1, 'CW':1},
                   'number of rotors':4}
        # # Map normalizer
        # self.NORMFuncs = {True:self._normalizeData, False:self._skipNormalization}
        
        # Map induced velocity to true function or dummy function if unused by models
        self.VINFuncs = {True:self._getInducedVelocity_True, False:self._getInducedVelocity_Dummy}
        self._getInducedVelocity = self.VINFuncs[self.VINFLAG]
        self.columns = ['w1', 'w2', 'w3', 'w4', 'w2_1', 'w2_2', 'w2_3', 'w2_4', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'v_in', 'p', 'q', 'r']
        self.ncolumns = ['w_tot', 'w_avg', 'w1', 'w2', 'w3', 'w4', 'w2_1', 'w2_2', 'w2_3', 'w2_4',
                        'd_w1', 'd_w2', 'd_w3', 'd_w4', 'd_w_tot',
                        'p', 'q', 'r', 'u_p', 'u_q', 'u_r', 'roll', 'pitch', 'yaw', 
                        'U_p', 'U_q', 'U_r', '|U_p|', '|U_q|', '|U_r|',
                        'u', 'v', 'w', 'v_in', 'mu_x', 'mu_y', 'mu_z', 'mu_vin',
                        '|p|', '|q|', '|r|', '|u_p|', '|u_q|', '|u_r|',
                        '|u|', '|v|', '|w|', '|mu_x|', '|mu_y|', '|mu_z|',
                        'sin[roll]', 'sin[pitch]', 'sin[yaw]','cos[roll]', 'cos[pitch]', 'cos[yaw]',
                        'F_den', 'M_den']
        return None


    def droneGetModelInput(self, state, rotorSpeeds):
        organizedData = self.fasterDataFrame(len(state), self.columns, self.npZeros)

        # Rotor speeds
        organizedData['w1'] = rotorSpeeds[:, 0]
        organizedData['w2'] = rotorSpeeds[:, 1]
        organizedData['w3'] = rotorSpeeds[:, 2]
        organizedData['w4'] = rotorSpeeds[:, 3]

        organizedData['w2_1'] = rotorSpeeds[:, 0]**2
        organizedData['w2_2'] = rotorSpeeds[:, 1]**2
        organizedData['w2_3'] = rotorSpeeds[:, 2]**2
        organizedData['w2_4'] = rotorSpeeds[:, 3]**2

        # Attitude
        organizedData['roll'] = state[:, 0]
        organizedData['pitch'] = state[:, 1]
        organizedData['yaw'] = state[:, 2]

        # Body linear velocity
        organizedData['u'] = state[:, 3]
        organizedData['v'] = state[:, 4]
        organizedData['w'] = state[:, 5]

        # Induced velocity
        organizedData['v_in'] = self._getInducedVelocity(organizedData, self.droneParams)

        # Body rotational rate
        organizedData['p'] = state[:, 6]
        organizedData['q'] = state[:, 7]
        organizedData['r'] = state[:, 8]

        # Apply normalization
        if self.isNormalized:
            normalizedData = self._normalizeData(organizedData, self.droneParams)
        else:
            # Add extra and necessary columns
            N_rot = 4 # number rotors
            R = self.droneParams['R']
            r_sign = self.droneParams['r_sign']
            rotorConfig = self.droneParams['rotor configuration']
            rotorDir = self.droneParams['rotor 1 direction']
            normalizedData = self.fasterDataFrame(organizedData.shape[0], self.ncolumns, self.npZeros)

            # Pre-fill normalizedData with organizedData
            for k in self.ncolumns:
                if k in self.columns:
                    normalizedData[k] = organizedData[k]

            normalizedData['w1'] = rotorSpeeds[:, 0]*2*self.npPi/60
            normalizedData['w2'] = rotorSpeeds[:, 1]*2*self.npPi/60
            normalizedData['w3'] = rotorSpeeds[:, 2]*2*self.npPi/60
            normalizedData['w4'] = rotorSpeeds[:, 3]*2*self.npPi/60

            normalizedData['d_w1'] = normalizedData['w1'] - self.droneParams['wHover (rad/s)']
            normalizedData['d_w2'] = normalizedData['w2'] - self.droneParams['wHover (rad/s)']
            normalizedData['d_w3'] = normalizedData['w3'] - self.droneParams['wHover (rad/s)']
            normalizedData['d_w4'] = normalizedData['w4'] - self.droneParams['wHover (rad/s)']

            normalizedData['w2_1'] = normalizedData['w1']**2
            normalizedData['w2_2'] = normalizedData['w2']**2
            normalizedData['w2_3'] = normalizedData['w3']**2
            normalizedData['w2_4'] = normalizedData['w4']**2

            normalizedData['F_den'] = 1
            normalizedData['M_den'] = 1

            omega = normalizedData[['w1', 'w2', 'w3', 'w4']]
            omega_tot = self.npSum(omega, axis=1)
            omega2 = normalizedData[['w2_1', 'w2_2', 'w2_3', 'w2_4']]
            w_avg = self.npArray(self._sqrt(self.npSum(omega2, axis = 1)/N_rot)).reshape(-1)
            normalizedData['w_tot'] = omega_tot
            normalizedData['d_w_tot'] = omega_tot - self.droneParams['wHover (rad/s)']*4
            normalizedData['w_avg'] = w_avg

            # Define control moments
            normalizedData['u_p'] = (omega[:, rotorConfig['front left'] - 1] + omega[:, rotorConfig['aft left'] - 1]) - (omega[:, rotorConfig['front right'] - 1] + omega[:, rotorConfig['aft right'] - 1])
            normalizedData['u_q'] = (omega[:, rotorConfig['front left'] - 1] + omega[:, rotorConfig['front right'] - 1]) - (omega[:, rotorConfig['aft left'] - 1] + omega[:, rotorConfig['aft right'] - 1])
            normalizedData['u_r'] = r_sign[rotorDir]*((omega[:, rotorConfig['front left'] - 1] + omega[:, rotorConfig['aft right']- 1]) - (omega[:, rotorConfig['front right'] - 1] + omega[:, rotorConfig['aft left'] - 1]))

            normalizedData['U_p'] = (omega2[:, rotorConfig['front left'] - 1] + omega2[:, rotorConfig['aft left'] - 1]) - (omega2[:, rotorConfig['front right'] - 1] + omega2[:, rotorConfig['aft right'] - 1])
            normalizedData['U_q'] = (omega2[:, rotorConfig['front left'] - 1] + omega2[:, rotorConfig['front right'] - 1]) - (omega2[:, rotorConfig['aft left'] - 1] + omega2[:, rotorConfig['aft right'] - 1])
            normalizedData['U_r'] = r_sign[rotorDir]*((omega2[:, rotorConfig['front left'] - 1] + omega2[:, rotorConfig['aft right']- 1]) - (omega2[:, rotorConfig['front right'] - 1] + omega2[:, rotorConfig['aft left'] - 1]))

            # Advance ratios
            normalizedData['mu_x'] = self.npDivide(normalizedData['u'], w_avg*R)
            normalizedData['mu_y'] = self.npDivide(normalizedData['v'], w_avg*R)
            normalizedData['mu_z'] = self.npDivide(normalizedData['w'], w_avg*R)

            # Normalize the induced velocity
            normalizedData['mu_vin'] = self.npDivide(normalizedData['v_in'], w_avg*R)

            # Add extra columns
            normalizedData = self._addExtraCols(normalizedData)


        return normalizedData


    def updateDroneParams(self, key, value):
        self.droneParams.update({key:value})
        return None


    def _square(self, x):
        return self.npPower(x, 2)


    def _sqrt(self, x):
        return self.npPower(x, 0.5)


    def _getInducedVelocity_True(self, filteredData, droneParams):
        def checkIfReal(val):
            imPart = float(val.imag)
            if self.npIsclose(imPart, 0):
                rePart = float(val.real)
            else:
                rePart = self.npNan
            return rePart
            
        rotorRadius = float(droneParams['R'])
        airDensity = float(droneParams['rho'])
        g = float(droneParams['g'])
        mass = float(droneParams['m'])
        numRotors = self.droneParams['number of rotors']

        thurstHover_est = mass*g

        inducedVelocityHover = self._sqrt(thurstHover_est/(2*airDensity*numRotors*(2*self.npPi*rotorRadius**2)))

        # u = filteredData['u'].to_numpy()
        u = filteredData['u']
        # v = filteredData['v'].to_numpy()
        v = filteredData['v']
        # w = filteredData['w'].to_numpy()
        w = filteredData['w']
        V = self._sqrt(self._square(u) + self._square(v) + self._square(w))
        v_in_vals = self.npZeros(len(u))
        for i in range(len(v_in_vals)):
            coeff = [1, -2*w[i], V[i]**2, 0, -1*inducedVelocityHover**4]
            try:
                roots = self.npRoots(coeff)
            except self.LinAlgError:
                roots = []
            if len(roots):
                if len(roots) > 1:
                    val = self.npNan
                    diff = 100000
                    for j in roots:
                        _val = checkIfReal(j)
                        if not self.npIsnan(_val):
                            if i > 0:
                                _diff = self.npAbs(_val - v_in_vals[i-1])
                            else:
                                _diff = self.npAbs(_val - inducedVelocityHover)
                            if _diff < diff:
                                val = _val
                                diff = _diff
                    v_in_vals[i] = val
                else:
                    v_in_vals[i] = checkIfReal(roots[0])
            else:
                v_in_vals[i] = self.npNan
        return v_in_vals


    def _getInducedVelocity_Dummy(self, filteredData, droneParams):
        v_in_vals = self.npZeros(len(filteredData['u'])) # * self.npNan
        return v_in_vals


    def _normalizeData(self, filteredData, droneParams):
        # Extract drone params
        R = droneParams['R']
        b = droneParams['b']
        rho = droneParams['rho']
        r_sign = droneParams['r_sign']
        rotorConfig = droneParams['rotor configuration']
        rotorDir = droneParams['rotor 1 direction']
        N_rot = 4
        minRPM = float(droneParams['idle RPM'])

        # Derive normalized rotor speed
        omega = filteredData[['w1', 'w2', 'w3', 'w4']]*2*self.npPi/60
        omega_tot = self.npSum(omega, axis=1)
        omega2 = filteredData[['w2_1', 'w2_2', 'w2_3', 'w2_4']]*(2*self.npPi/60)**2
        w_avg = self.npArray(self._sqrt(self.npSum(omega2, axis = 1)/N_rot)).reshape(-1)
        # Replace 0 with self.npNan
        w_avg = self.npWhere(w_avg < minRPM*(2*self.npPi/60), self.npNan, w_avg)        

        # Normalize rotor speed
        n_omega = self.npDivide(omega, w_avg.reshape(-1, 1))
        n_omega_tot = self.npDivide(omega_tot, w_avg) * (w_avg/droneParams['max RPM'])
        n_omega2 = self.npDivide(omega2, self._square(w_avg).reshape(-1, 1))

        # Normalize rates 
        n_p = self.npDivide(filteredData['p']*b, w_avg*R)
        n_q = self.npDivide(filteredData['q']*b, w_avg*R)
        n_r = self.npDivide(filteredData['r']*b, w_avg*R)

        # Define control moments 
        u_p = (n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['aft left'] - 1]) - (n_omega[:, rotorConfig['front right'] - 1] + n_omega[:, rotorConfig['aft right'] - 1])
        u_q = (n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['front right'] - 1]) - (n_omega[:, rotorConfig['aft left'] - 1] + n_omega[:, rotorConfig['aft right'] - 1])
        u_r = r_sign[rotorDir]*((n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['aft right']- 1]) - (n_omega[:, rotorConfig['front right'] - 1] + n_omega[:, rotorConfig['aft left'] - 1]))

        U_p = (n_omega2[:, rotorConfig['front left'] - 1] + n_omega2[:, rotorConfig['aft left'] - 1]) - (n_omega2[:, rotorConfig['front right'] - 1] + n_omega2[:, rotorConfig['aft right'] - 1])
        U_q = (n_omega2[:, rotorConfig['front left'] - 1] + n_omega2[:, rotorConfig['front right'] - 1]) - (n_omega2[:, rotorConfig['aft left'] - 1] + n_omega2[:, rotorConfig['aft right'] - 1])
        U_r = r_sign[rotorDir]*((n_omega2[:, rotorConfig['front left'] - 1] + n_omega2[:, rotorConfig['aft right']- 1]) - (n_omega2[:, rotorConfig['front right'] - 1] + n_omega2[:, rotorConfig['aft left'] - 1]))


        # Normalize velocities
        va = self._sqrt(self.npSum(self._square(filteredData[['u', 'v', 'w']]), axis = 1))
        slow_va_idx = self.npWhere(va < 0.01)[0]
        va[slow_va_idx] = 0.01 # TO avoid runtime warnings

        u_bar = self.npDivide(filteredData['u'], va)
        u_bar[slow_va_idx] = 0
        v_bar = self.npDivide(filteredData['v'], va)
        v_bar[slow_va_idx] = 0
        w_bar = self.npDivide(filteredData['w'], va)
        w_bar[slow_va_idx] = 0

        # Normalize the induced velocity
        vi_bar = self.npDivide(filteredData['v_in'], va)
        vi_bar[slow_va_idx] = 0


        # Normalize velocities
        mux_bar = self.npDivide(filteredData['u'], w_avg*R)
        muy_bar = self.npDivide(filteredData['v'], w_avg*R)
        muz_bar = self.npDivide(filteredData['w'], w_avg*R)

        # Normalize the induced velocity
        mu_vi_bar = self.npDivide(filteredData['v_in'], w_avg*R)


        # Get force and moment normalizing factor 
        # F_den = 2/(rho*(N_rot*self.npPi*R**2)*(R*w_avg)**2)
        # F_den = 2/(rho*(N_rot*self.npPi*R**2)*(R**2*w_avg*droneParams['max RPM']*2*self.npPi/60))
        # F_den = 2/(rho*(N_rot*self.npPi*R**2)*(R*(droneParams['max RPM']*(2*self.npPi/60))**2/(droneParams['max RPM']*(2*self.npPi/60)+w_avg))**2)
        F_den = (0.5*rho*(N_rot*self.npPi*R**2)*(R*(droneParams['max RPM']*(2*self.npPi/60))**2/(droneParams['max RPM']*(2*self.npPi/60)+w_avg))**2)
        M_den = F_den * (1/b)

        # Replace NaN
        F_den_NaNs = self.npWhere(self.npIsnan(F_den))[0]
        F_den[F_den_NaNs] = 0
        M_den[F_den_NaNs] = 0

        # Define normalized DataFrame
        NormalizedData = self.fasterDataFrame(filteredData.shape[0], self.ncolumns, self.npZeros)
        NormalizedData['w_tot'] = n_omega_tot.reshape(-1)
        NormalizedData['d_w_tot'] = self.npDivide(omega_tot - 4*self.droneParams['wHover (rad/s)'], w_avg) * (w_avg/droneParams['max RPM'])
        NormalizedData['w_avg'] = w_avg.reshape(-1)
        NormalizedData[['w1', 'w2', 'w3', 'w4']] = n_omega
        NormalizedData[['d_w1', 'd_w2', 'd_w3', 'd_w4']] = n_omega - self.droneParams['wHover (rad/s)'] / w_avg
        NormalizedData[['w2_1', 'w2_2', 'w2_3', 'w2_4']] = n_omega2
        NormalizedData[['p', 'q', 'r']] = self.npVstack((n_p, n_q, n_r)).T
        NormalizedData[['u_p', 'u_q', 'u_r']] = self.npVstack((u_p, u_q, u_r)).T
        NormalizedData[['U_p', 'U_q', 'U_r']] = self.npVstack((U_p, U_q, U_r)).T
        NormalizedData[['roll', 'pitch', 'yaw']] = filteredData[['roll', 'pitch', 'yaw']]
        NormalizedData[['u', 'v', 'w']] = self.npVstack((u_bar, v_bar, w_bar)).T
        NormalizedData['v_in'] = vi_bar
        NormalizedData[['mu_x', 'mu_y', 'mu_z']] = self.npVstack((mux_bar, muy_bar, muz_bar)).T
        NormalizedData['mu_vin'] = mu_vi_bar
        NormalizedData['F_den'] = F_den.reshape(-1)
        NormalizedData['M_den'] = M_den.reshape(-1)

        # Replace NaNs
        # NormalizedData.fillna(0, inplace=True)

        # Add extra useful columns
        NormalizedData = self._addExtraCols(NormalizedData)

        return NormalizedData


    def _addExtraCols(self, Data):
        # Add abs(body velocity)
        Data['|u|'] = self.npAbs(Data['u'])
        Data['|v|'] = self.npAbs(Data['v'])
        Data['|w|'] = self.npAbs(Data['w'])

        # Add abs(body advance rations) 
        Data['|mu_x|'] = self.npAbs(Data['mu_x'])
        Data['|mu_y|'] = self.npAbs(Data['mu_y'])
        Data['|mu_z|'] = self.npAbs(Data['mu_z']) 

        # Add abs(rotational rates)
        Data['|p|'] = self.npAbs(Data['p'])
        Data['|q|'] = self.npAbs(Data['q'])
        Data['|r|'] = self.npAbs(Data['r'])

        # Add abs(control moments)
        Data['|u_p|'] = self.npAbs(Data['u_p'])
        Data['|u_q|'] = self.npAbs(Data['u_q'])
        Data['|u_r|'] = self.npAbs(Data['u_r'])
        Data['|U_p|'] = self.npAbs(Data['U_p'])
        Data['|U_q|'] = self.npAbs(Data['U_q'])
        Data['|U_r|'] = self.npAbs(Data['U_r'])        

        # Get trigonometric identities of attitude angles 
        Data['sin[roll]'] = self.npSin(Data['roll'])
        Data['sin[pitch]'] = self.npSin(Data['pitch'])
        Data['sin[yaw]'] = self.npSin(Data['yaw'])

        Data['cos[roll]'] = self.npCos(Data['roll'])
        Data['cos[pitch]'] = self.npCos(Data['pitch'])
        Data['cos[yaw]'] = self.npCos(Data['yaw'])        
        return Data


    class fasterDataFrame:
        def __init__(self, numRows, columns, npZeros):
            self.npZeros = npZeros
            self.shape = (numRows, len(columns))
            self.dfvalues = npZeros(self.shape)
            self.dfmapping = {k:v for v, k in enumerate(columns)}
            self.columns = columns

        # def __getitem__(self, key):
        #     return self.dfvalues[:, self.dfmapping[key]]

        def __getitem__(self, key):
            # Check if key or list is passed
            try:
                out = self.dfvalues[:, self.dfmapping[key]]
            except TypeError:
                out = self.npZeros((self.shape[0], len(key)))
                for i, k in enumerate(key):
                    out[:, i] = self.dfvalues[:, self.dfmapping[k]]
            return out


        def __setitem__(self, key, newvalue):
            try:
                self.dfvalues[:, self.dfmapping[key]] = newvalue
            except TypeError:
                for i, k in enumerate(key):
                    self.dfvalues[:, self.dfmapping[k]] = newvalue[:, i]




class DroneDiffSystem:

    def __init__(self, x, u, polyDroneModels, droneParams):
        self.matrix = matrix
        self.ones = ones
        self.x = x
        self.u = u
        self.polynomials = []
        self.coefficients = []
        self.regressors = []
        for m in polyDroneModels:
            self.polynomials.append(m.polynomial)
            # self.coefficients.append(list(m.coefficients.__array__().reshape(-1)))
            self.coefficients.append(m.coefficients)
            self.regressors.append(m.regressors)
        self.columns = polyDroneModels[0].ncolumns
        self.droneParams = droneParams
        self._models = polyDroneModels
        self._getDerivativeSystem()


    def _getDerivativeSystem(self):
        Iv = self.droneParams['Iv']
        self.invIv = inv(Iv)

        # Based on:
        #   omega_dot   = inv(Iv)*(M_B) - inv(Iv)*(omega X (Iv * omega))
        #   let omega_dot_fixed = -inv(Iv)*(omega X (Iv * omega))
        # NOTE: Here we force the inertias to be in decimal like notation to avoid errors
        #       with our parse misinterpretting 1.23e-45. However, we need to select a precision
        #       so we go with 10 here. But if we have inertias lower than this, then this will 
        #       be an issue
        Ixx = f'{Iv[0][0]:.10f}'
        Ixy = f'{Iv[0][1]:.10f}'
        Ixz = f'{Iv[0][2]:.10f}'

        Iyx = f'{Iv[1][0]:.10f}'
        Iyy = f'{Iv[1][1]:.10f}'
        Iyz = f'{Iv[1][2]:.10f}'

        Izx = f'{Iv[2][0]:.10f}'
        Izy = f'{Iv[2][1]:.10f}'
        Izz = f'{Iv[2][2]:.10f}'

        a11 = self.invIv[0][0]
        a12 = self.invIv[0][1]
        a13 = self.invIv[0][2]

        a21 = self.invIv[1][0]
        a22 = self.invIv[1][1]
        a23 = self.invIv[1][2]

        a31 = self.invIv[2][0]
        a32 = self.invIv[2][1]
        a33 = self.invIv[2][2]

        # TODO: Optimize w.r.t zeros e.g. np.isclose(a11, 0) -> p_dot_dp.pop(0)
        # NOTE: If any of the coefficients are put in scientific notation, it may screw up the NOTE: NOTE NOTE:
        # p_dot / dp [fixed]
        p_dot_dp = [
            f'{a11}*({Izx}*q - {Iyx}*p)',
            f'{a12}*({Ixx}*r - ({Izz}*r + 2*{Izx}*p + {Izy}*q))',
            f'{a13}*({Iyy}*q + 2*{Iyx}*p + {Iyz}*r - {Ixx}*q)'
        ]
        self.p_dot_dp = PolynomialModel(self.droneParams)
        self.p_dot_dp.polynomial = p_dot_dp
        self.p_dot_dp.coefficients = -1*self.matrix(self.ones(len(p_dot_dp))).T
        self.p_dot_dp.makeRegressors()

        # p_dot / dq [fixed]
        p_dot_dq = [
            f'{a11}*({Izz}*r + {Izx}*p + 2*{Izy}*q - {Iyy}*r)',
            f'{a12}*({Ixy}*r - {Izy}*p)',
            f'{a13}*({Iyy}*p - ({Ixx}*p + 2*{Ixy}*q + {Ixz}*r))'
        ]
        self.p_dot_dq = PolynomialModel(self.droneParams)
        self.p_dot_dq.polynomial = p_dot_dq
        self.p_dot_dq.coefficients = -1*self.matrix(self.ones(len(p_dot_dq))).T
        self.p_dot_dq.makeRegressors()

        # p_dot / dr [fixed]
        p_dot_dr = [
            f'{a11}*({Izz}*q - ({Iyy}*q + {Iyx}*p + 2*{Iyz}*r))',
            f'{a12}*({Ixx}*p + {Ixy}*q + 2*{Ixz}*r - {Izz}*p)',
            f'{a13}*({Iyz}*p - {Ixz}*q)'
        ]
        self.p_dot_dr = PolynomialModel(self.droneParams)
        self.p_dot_dr.polynomial = p_dot_dr
        self.p_dot_dr.coefficients = -1*self.matrix(self.ones(len(p_dot_dr))).T
        self.p_dot_dr.makeRegressors()


        # q_dot / dp [fixed]
        q_dot_dp = [
            f'{a21}*({Izx}*q - {Iyx}*r)',
            f'{a22}*({Ixx}*r - ({Izz}*r + 2*{Izx}*p + {Izy}*p))',
            f'{a23}*({Iyy}*q + 2*{Iyx}*p + {Iyx}*r - {Ixx}*q)'
        ]
        self.q_dot_dp = PolynomialModel(self.droneParams)
        self.q_dot_dp.polynomial = q_dot_dp
        self.q_dot_dp.coefficients = -1*self.matrix(self.ones(len(q_dot_dp))).T
        self.q_dot_dp.makeRegressors()


        # q_dot / dq [fixed]
        q_dot_dq = [
            f'{a21}*({Izz}*r + {Izx}*p + 2*{Izy}*q - {Iyy}*r)',
            f'{a22}*({Ixy}*r - {Izy}*p)',
            f'{a23}*({Iyy}*p - ({Ixx}*p + 2*{Ixy}*q + {Ixz}*r))'
        ]
        self.q_dot_dq = PolynomialModel(self.droneParams)
        self.q_dot_dq.polynomial = q_dot_dq
        self.q_dot_dq.coefficients = -1*self.matrix(self.ones(len(q_dot_dq))).T
        self.q_dot_dq.makeRegressors()


        # q_dot / dr [fixed]
        q_dot_dr = [
            f'{a21}*({Izz}*q - ({Iyy}*q + {Iyx}*p + 2*{Iyz}*r))',
            f'{a22}*({Ixx}*p + {Ixy}*q + 2*{Ixz}*r - {Izz}*p)',
            f'{a23}*({Iyz}*p - {Ixz}*q)'
        ]
        self.q_dot_dr = PolynomialModel(self.droneParams)
        self.q_dot_dr.polynomial = q_dot_dr
        self.q_dot_dr.coefficients = -1*self.matrix(self.ones(len(q_dot_dr))).T
        self.q_dot_dr.makeRegressors()


        # r_dot / dp [fixed]
        r_dot_dp = [
            f'{a31}*({Izx}*q - {Iyx}*r)',
            f'{a32}*({Ixx}*r - ({Izz}*r + 2*{Izx}*p + {Izy}*q))',
            f'{a33}*({Iyy}*q + 2*{Iyx}*p + {Iyz}*r - {Ixx}*q)'
        ]
        self.r_dot_dp = PolynomialModel(self.droneParams)
        self.r_dot_dp.polynomial = r_dot_dp
        self.r_dot_dp.coefficients = -1*self.matrix(self.ones(len(r_dot_dp))).T
        self.r_dot_dp.makeRegressors()        

        # r_dot / dq [fixed]
        r_dot_dq = [
            f'{a31}*({Izz}*r + {Izx}*p + 2*{Izy}*q - {Iyy}*r)',
            f'{a32}*({Ixy}*r - {Izy}*p)',
            f'{a33}*({Iyy}*p - ({Ixx}*p + 2*{Ixy}*q + {Ixz}*r))'
        ]
        self.r_dot_dq = PolynomialModel(self.droneParams)
        self.r_dot_dq.polynomial = r_dot_dq
        self.r_dot_dq.coefficients = -1*self.matrix(self.ones(len(r_dot_dq))).T
        self.r_dot_dq.makeRegressors()                

        # r_dot / dr [fixed]
        r_dot_dr = [
            f'{a31}*({Izz}*q - ({Iyy}*q + {Iyx}*p + 2*{Iyz}*r))',
            f'{a32}*({Ixx}*p + {Ixy}*q + 2*{Ixz}*r - {Izz}*p)',
            f'{a33}*({Iyz}*p - {Ixz}*q)'
        ]
        self.r_dot_dr = PolynomialModel(self.droneParams)
        self.r_dot_dr.polynomial = r_dot_dr
        self.r_dot_dr.coefficients = -1*self.matrix(self.ones(len(r_dot_dr))).T
        self.r_dot_dr.makeRegressors()        


        # Get variable
        self._DfDx_DfDu_Variable = []
        for poly, coeff, regs in zip(self.polynomials, self.coefficients, self.regressors):
            # Generate input file for [getPartialDerivative.py]
            inputFile = {}
            inputFile.update({'state':self.x})
            inputFile.update({'input':self.u})
            # Find extra variables in the poly equation
            extras = []
            for reg in regs:
                for r in array(reg.RPN)[reg.variableIndices]:
                    if r not in self.x and r not in self.u and r in self.columns and r not in extras:
                        if r not in [r'|' + _x + r'|' for _x in self.x] and r not in [r'|' + _u + r'|' for _u in self.u]:
                            extras.append(r)
            if len(extras):
                print(f'[ WARNING ] Found variables, {extras}, not in x or u. ASSUMING it is invariant of x or u.')
            inputFile.update({'extra':extras})
            inputFile.update({'poly':poly})
            # inputFile.update({'coeffs':coeff})

            with open('_getPartialDerivativeInputs.json', 'w') as f:
                jDump(inputFile, f, indent = 4)
                
            try:
                prcs = subprocess.check_call(['python', 'getPartialDerivative.py'])
            except subprocess.CalledProcessError:
                raise RuntimeError('Could not run [getPartialDerivative.py]')

            with open('dfdx_dfdu.json', 'r') as f:
                _dfdx_dfdu_elems = jLoad(f)

            _dfdx_elems = _dfdx_dfdu_elems['dfdx']
            _dfdx_variable = {}
            for _x in _dfdx_elems.keys():
                _polyModel = PolynomialModel(self.droneParams)
                _polyModel.polynomial = _dfdx_elems[_x]
                _polyModel.coefficients = coeff
                _polyModel.makeRegressors()
                _dfdx_variable.update({_x:_polyModel})
            
            _dfdu_elems = _dfdx_dfdu_elems['dfdu']
            _dfdu_variable = {}
            for _u in _dfdu_elems.keys():
                _polyModel = PolynomialModel(self.droneParams)
                _polyModel.polynomial = _dfdu_elems[_u]
                _polyModel.coefficients = coeff
                _polyModel.makeRegressors()
                _dfdu_variable.update({_u:_polyModel})
            
            self._DfDx_DfDu_Variable.append({'dfdx':_dfdx_variable, 'dfdu':_dfdu_variable})
            # import code
            # code.interact(local=locals())


    def _getDroneInputs(self, x, omega):
        return self._models[0].droneGetModelInput(x, omega)


    def getAB(self, droneInputs):
        a11 = self.invIv[0][0]
        a12 = self.invIv[0][1]
        a13 = self.invIv[0][2]

        a21 = self.invIv[1][0]
        a22 = self.invIv[1][1]
        a23 = self.invIv[1][2]
        
        a31 = self.invIv[2][0]
        a32 = self.invIv[2][1]
        a33 = self.invIv[2][2]

        Mx = self._DfDx_DfDu_Variable[0]
        My = self._DfDx_DfDu_Variable[1]
        Mz = self._DfDx_DfDu_Variable[2]

        # DfDx
        # p_ddot eqs
        A11 = (a11 * Mx['dfdx']['p'].predict(droneInputs) + self.p_dot_dp.predict(droneInputs)).__array__()[0][0]
        A12 = (a12 * My['dfdx']['q'].predict(droneInputs) + self.p_dot_dq.predict(droneInputs)).__array__()[0][0]
        A13 = (a13 * Mz['dfdx']['r'].predict(droneInputs) + self.p_dot_dr.predict(droneInputs)).__array__()[0][0]

        # q_ddot eqs 
        A21 = (a21 * Mx['dfdx']['p'].predict(droneInputs) + self.q_dot_dp.predict(droneInputs)).__array__()[0][0]
        A22 = (a22 * My['dfdx']['q'].predict(droneInputs) + self.q_dot_dq.predict(droneInputs)).__array__()[0][0]
        A23 = (a23 * Mz['dfdx']['r'].predict(droneInputs) + self.q_dot_dr.predict(droneInputs)).__array__()[0][0]

        # r_ddot eqs 
        A31 = (a31 * Mx['dfdx']['p'].predict(droneInputs) + self.r_dot_dp.predict(droneInputs)).__array__()[0][0]
        A32 = (a32 * My['dfdx']['q'].predict(droneInputs) + self.r_dot_dq.predict(droneInputs)).__array__()[0][0]
        A33 = (a33 * Mz['dfdx']['r'].predict(droneInputs) + self.r_dot_dr.predict(droneInputs)).__array__()[0][0]

        self.DfDx = self.matrix([
            [A11, A12, A13],
            [A21, A22, A23],
            [A31, A32, A33]
            ])

        # DfDu
        # p_ddot eqs 
        B11 = (a11 * Mx['dfdu'][self.u[0]].predict(droneInputs)).__array__()[0][0]
        B12 = (a12 * My['dfdu'][self.u[1]].predict(droneInputs)).__array__()[0][0]
        B13 = (a13 * Mz['dfdu'][self.u[2]].predict(droneInputs)).__array__()[0][0]

        # q_ddot eqs 
        B21 = (a21 * Mx['dfdu'][self.u[0]].predict(droneInputs)).__array__()[0][0]
        B22 = (a22 * My['dfdu'][self.u[1]].predict(droneInputs)).__array__()[0][0]
        B23 = (a23 * Mz['dfdu'][self.u[2]].predict(droneInputs)).__array__()[0][0]

        # r_ddot eqs 
        B31 = (a31 * Mx['dfdu'][self.u[0]].predict(droneInputs)).__array__()[0][0]
        B32 = (a32 * My['dfdu'][self.u[1]].predict(droneInputs)).__array__()[0][0]
        B33 = (a33 * Mz['dfdu'][self.u[2]].predict(droneInputs)).__array__()[0][0]

        self.DfDu = self.matrix([
            [B11, B12, B13],
            [B21, B22, B23],
            [B31, B32, B33]
            ])

        return self.DfDx, self.DfDu
        

    def predict(self, droneInputs, x_dot, u_dot):
        A, B = self.getAB(droneInputs)
        return A*x_dot + B*u_dot



# ================================================================================================================================ #
# Definitions
# ================================================================================================================================ #
def loadModels(modelPath, mdlID, *args):
    # Check that arguments are passed
    if len(args) == 0:
        raise ValueError('Please pass model names (e.g. Fx) to load!')
    # Load models and check that metadata is consistent
    models = {}
    path = os.path.join(modelPath, mdlID)
    # Check if actuator dynamics have been estimated
    hasTau = None
    if os.path.exists(os.path.join(path, 'taus.json')):
        print('[ INFO ] Found actuator dynamics information, adding to standalone model.')
        hasTau = os.path.join(path, 'taus.json')
    # Check if simple model is present
    simpleModel = None
    if os.path.exists(os.path.join(path, 'simpleModel.json')):
        # print('[ INFO ] Found simpleModel, adding to standalone model.')
        simpleModel = os.path.join(path, 'simpleModel.json')
    models.update({args[0]:SysID.Model.load(os.path.join(path, args[0]))})
    # Get reference metadata from first argument
    with open(os.path.join(os.path.join(path, args[0], 'processingMetadata.json')), 'rb') as f:
        referenceMetadata = jLoad(f)
    isNormalized = referenceMetadata["data normalization"]['normalize data']
    hasGravity = not referenceMetadata["data filtering"]["remove influence of gravity"]
    print('[ INFO ] [makeStandAlonePolyModel.py] Loading {} (normalized: {}, gravity: {})'.format(args[0], isNormalized, hasGravity))
    try:
        usesVIN = referenceMetadata["identification parameters"]["add induced velocity to regressors"]
        VINFlagExists = True
    except KeyError:
        print('[ INFO ] Flag for uses induced velocity could not be found in metadata, will confirm with model regressors.')
        usesVIN = True
        VINFlagExists = False
    if len(args) > 1:
        for a in args[1:]:
            # Check consistency
            with open(os.path.join(os.path.join(path, a, 'processingMetadata.json')), 'rb') as f:
                _metadata = jLoad(f)
            print('[ INFO ] [makeStandAlonePolyModel.py] Loading {} (normalized: {}, gravity: {})'.format(a, _metadata["data normalization"]['normalize data'], not _metadata["data filtering"]["remove influence of gravity"]))
            if _metadata['data filtering']["remove influence of gravity"] != referenceMetadata['data filtering']["remove influence of gravity"]:
                raise ValueError('Gravity-Error: Models appear to be inconsistent (gravity present in one model but absent in another)')
            if _metadata["data normalization"]['normalize data'] != isNormalized:
                raise ValueError('Normalization-Error: Models appear to be inconsistent (one model is dimensional and another is not)')
            if VINFlagExists:
                if _metadata["identification parameters"]["add induced velocity to regressors"]:
                    usesVIN = True
            models.update({a:SysID.Model.load(os.path.join(path, a))})
    if usesVIN:
        actuallyUsesVin = False
        for v in models.values():
            polynomial = v.CurrentModel['Model']['Regressors']
            if 'v_in' in '#'.join(polynomial) or 'mu_vin' in '#'.join(polynomial):
                actuallyUsesVin = True
                break
        usesVIN = actuallyUsesVin
        print(f'[ INFO ] Searched polynomials and updated uses induced velocity to {usesVIN}')
    return models, isNormalized, hasGravity, usesVIN, referenceMetadata, hasTau, simpleModel


def makeStandalone(savePath, mdlID, standaloneHandler, applyNormalization, hasGravity, usesVIN, mapping, metadata, hasTau, simpleModel):
    # Convert to standalone model using handler
    for a, m in mapping.items():
        polyModel = standaloneHandler(droneConfigurationFile, isNormalized = applyNormalization, hasGravity = hasGravity, usesVIN = usesVIN, metadata = metadata)
        polyModel.extractModel(m)
        if hasTau is not None:
            with open(hasTau, 'r') as f:
                taus = jLoad(f)
            polyModel.droneParams.update(taus)
        if simpleModel is not None:
            with open(simpleModel, 'r') as f:
                simpleModelData = jLoad(f)
            polyModel.droneParams.update({'simpleModel':simpleModelData})
        with open(os.path.join(savePath, mdlID + '-{}.pkl'.format(a)), 'wb') as f:
            pickle.dump(polyModel, f)
    return None


def toStandalone(modelPath, mdlID, savePath, standaloneHandler, *args):
    # Load and convert SysID polynomial models to standalone variant
    models, isNormalized, hasGravity, usesVIN, metadata, hasTau, simpleModel = loadModels(modelPath, mdlID, *args)
    makeStandalone(savePath, mdlID, standaloneHandler, isNormalized, hasGravity, usesVIN, models, metadata, hasTau, simpleModel)
    return None

# ================================================================================================================================ #
# Main script
# ================================================================================================================================ #
# Load config data
with open('standaloneConfig.json', 'r') as f:
    config = jLoad(f)

# Load models
# Path to parent folder of SysID model folder
modelPath = config['model path']
mdlID = config['model ID']

# Drone configuration directory and filename (if using DronePolynomialModel)
configPath = config['droneConfig path']
configFile = config['droneConfig name']
droneConfigurationFile = os.path.join(configPath, configFile)

# Save models
savePath = os.path.join(modelPath, mdlID, 'standaloneDiff')
if not os.path.exists(savePath):
    os.makedirs(savePath)

# If saveGeneral is true, then the drone specific processing functions will also be saved to the model file,
#   such that processing can be reproduced when working only with the filtered state and rotor speeds.
# If saveGeneral is False, then only the model will be stored. Thus, any processing required to obtain the
#   model inputs (as done for during the training of the model) must be done prior to making predictions. 
saveGeneral = False
handler = PolynomialModel
if not saveGeneral:
    handler = DronePolynomialModel

toLoad = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
toStandalone(modelPath, mdlID, savePath, handler, *toLoad)

if config['make moment DiffSys']:
    with open(os.path.join(savePath, mdlID + '-Mx.pkl'), 'rb') as f:
        MxModel = pickle.load(f)

    with open(os.path.join(savePath, mdlID + '-My.pkl'), 'rb') as f:
        MyModel = pickle.load(f)

    with open(os.path.join(savePath, mdlID + '-Mz.pkl'), 'rb') as f:
        MzModel = pickle.load(f)

    DiffSys = DroneDiffSystem(['p', 'q', 'r'], config['DiffSys u'], [MxModel, MyModel, MzModel], MxModel.droneParams)

    with open(os.path.join(savePath, 'PQR-DiffSys.pkl'), 'wb') as f:
        pickle.dump(DiffSys, f)
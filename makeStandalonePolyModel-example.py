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
import dill as pickle
from numpy import matrix, ones, hstack, dot, add, subtract, divide, multiply, power, nan, array, where, arange, isnan, roots, isclose, pi, abs, zeros, vstack, cos, sin, sum
from re import sub as reSub
from io import StringIO
from tokenize import generate_tokens
import SysID
from numpy.linalg import LinAlgError
from json import load as jLoad

# ================================================================================================================================ #
# Classes
# ================================================================================================================================ #
'''
Class bundles SysID polynomial model into an independent object that can be used for predictions without the need for local modules.
Predictions are made using the .predict(x) method, where x holds the same structure as the data used as model inputs for training. 
'''
# TODO: Add ReadMe which documents what is expected of x, or reference to other document/file/script which outlines this. 
class PolynomialModel:
    
    def __init__(self):
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
                self._inv_XtX = sysIDModel.TrainedModel['Model']['_inv(XtX)']
                self._s2 = sysIDModel.TrainedModel['Model']['_sigma2']
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
        return sysIDModel.TrainedModel['Model']['Parameters']


    def _getPolynomial(self, sysIDModel):
        return sysIDModel.TrainedModel['Model']['Regressors']
    

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

    def __init__(self, droneConfigFilePath):
        PolynomialModel.__init__(self)
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
        self.jsonLoad = jLoad
        with open(droneConfigFilePath, 'r') as f:
            configData = self.jsonLoad(f)
        self.droneParams = {'R':float(configData['rotor radius']),
                   'b':float(configData['b']),
                   'Iv':self.npArray(self.npMatrix(configData['moment of inertia'])), 
                   'rotor configuration':configData['rotor config'],
                   'rotor 1 direction':configData['rotor1 rotation direction'],
                   'idle RPM':float(configData['idle RPM']),
                   'max RPM':float(configData['max RPM']),
                   'm':float(configData['mass']),
                   'g':9.81,
                   'rho':1.225,
                   'r_sign':{'CCW':-1, 'CW':1}}
        return None


    def droneGetModelInput(self, state, rotorSpeeds):
        columns = ['w1', 'w2', 'w3', 'w4', 'w2_1', 'w2_2', 'w2_3', 'w2_4', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'v_in', 'p', 'q', 'r']
        organizedData = self.fasterDataFrame(len(state), columns, self.npZeros)

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
        normalizedData = self._normalizeData(organizedData, self.droneParams)

        return normalizedData


    def updateDroneParams(self, key, value):
        self.droneParams.update({key:value})
        return None


    def _square(self, x):
        return self.npPower(x, 2)


    def _sqrt(self, x):
        return self.npPower(x, 0.5)


    def _getInducedVelocity(self, filteredData, droneParams):
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

        thurstHover_est = mass*g

        inducedVelocityHover = self._sqrt(thurstHover_est/(2*airDensity*4*(2*self.npPi*rotorRadius**2)))

        # u = filteredData['u'].to_numpy()
        u = filteredData['u']
        # v = filteredData['v'].to_numpy()
        v = filteredData['v']
        # w = filteredData['w'].to_numpy()
        w = filteredData['w']
        V = self._sqrt(self._square(u) + self._square(v) + self._square(w))
        v_in_vals = self.npZeros(len(u))
        for i in range(len(v_in_vals)):
            coeff = [4, -8*w[i], 4*V[i]**2, 0, -1*inducedVelocityHover**2]
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
                                _diff = self.npAbs(_val - V[i])
                            if _diff < diff:
                                val = _val
                                diff = _diff
                    v_in_vals[i] = val
                else:
                    v_in_vals[i] = checkIfReal(roots[0])
            else:
                v_in_vals[i] = self.npNan
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
        w_avg = self.npArray(self._sqrt(self.npSum(omega2, axis = 1)/N_rot)).reshape(-1, 1)
        # Replace 0 with self.npNan
        w_avg = self.npWhere(w_avg < minRPM*(2*self.npPi/60), self.npNan, w_avg)        

        # Normalize rotor speed
        n_omega = self.npDivide(omega, w_avg)
        n_omega_tot = self.npDivide(omega_tot.reshape(-1, 1), w_avg) * (w_avg/droneParams['max RPM'])
        n_omega2 = self.npDivide(omega2, self._square(w_avg))

        # Normalize rates 
        n_p = self.npDivide(filteredData['p'].reshape(-1, 1)*b, w_avg*R)
        n_q = self.npDivide(filteredData['q'].reshape(-1, 1)*b, w_avg*R)
        n_r = self.npDivide(filteredData['r'].reshape(-1, 1)*b, w_avg*R)

        # Define control moments 
        u_p = (n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['aft left'] - 1]) - (n_omega[:, rotorConfig['front right'] - 1] + n_omega[:, rotorConfig['aft right'] - 1])
        u_q = (n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['front right'] - 1]) - (n_omega[:, rotorConfig['aft left'] - 1] + n_omega[:, rotorConfig['aft right'] - 1])
        u_r = r_sign[rotorDir]*((n_omega[:, rotorConfig['front left'] - 1] + n_omega[:, rotorConfig['aft right']- 1]) - (n_omega[:, rotorConfig['front right'] - 1] + n_omega[:, rotorConfig['aft left'] - 1]))

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
        mux_bar = self.npDivide(filteredData['u'].reshape(-1, 1), w_avg*R)
        muy_bar = self.npDivide(filteredData['v'].reshape(-1, 1), w_avg*R)
        muz_bar = self.npDivide(filteredData['w'].reshape(-1, 1), w_avg*R)

        # Normalize the induced velocity
        mu_vi_bar = self.npDivide(filteredData['v_in'].reshape(-1, 1), w_avg*R)


        # Get force and moment normalizing factor 
        # F_den = 2/(rho*(N_rot*self.npPi*R**2)*(R*w_avg)**2)
        # F_den = 2/(rho*(N_rot*self.npPi*R**2)*(R**2*w_avg*droneParams['max RPM']*2*self.npPi/60))
        F_den = (0.5*rho*(N_rot*self.npPi*R**2)*(R**2*w_avg*droneParams['max RPM']*2*self.npPi/60))
        M_den = F_den * (1/b)

        # Replace NaN
        F_den_NaNs = self.npWhere(self.npIsnan(F_den))[0]
        F_den[F_den_NaNs] = 0
        M_den[F_den_NaNs] = 0

        # Define normalized DataFrame
        columns=['w_tot', 'w1', 'w2', 'w3', 'w4', 'w2_1', 'w2_2', 'w2_3', 'w2_4', 
                                                    'p', 'q', 'r', 'u_p', 'u_q', 'u_r', 'roll', 'pitch', 'yaw', 
                                                    'u', 'v', 'w', 'v_in', 'mu_x', 'mu_y', 'mu_z', 'mu_vin',
                                                    '|p|', '|q|', '|r|', '|u_p|', '|u_q|', '|u_r|',
                                                    '|u|', '|v|', '|w|', '|mu_x|', '|mu_y|', '|mu_z|',
                                                    'sin[roll]', 'sin[pitch]', 'sin[yaw]','cos[roll]', 'cos[pitch]', 'cos[yaw]',
                                                    'F_den', 'M_den']
        NormalizedData = self.fasterDataFrame(filteredData.shape[0], columns, self.npZeros)
        # NormalizedData = self.pdDataFrame(columns=['w_tot', 'w1', 'w2', 'w3', 'w4', 'w2_1', 'w2_2', 'w2_3', 'w2_4', 
        #                                             'p', 'q', 'r', 'u_p', 'u_q', 'u_r', 'roll', 'pitch', 'yaw', 
        #                                             'u', 'v', 'w', 'v_in', 'mu_x', 'mu_y', 'mu_z', 'mu_vin',
        #                                             '|p|', '|q|', '|r|', '|u_p|', '|u_q|', '|u_r|',
        #                                             '|u|', '|v|', '|w|', '|mu_x|', '|mu_y|', '|mu_z|',
        #                                             'sin[roll]', 'sin[pitch]', 'sin[yaw]','cos[roll]', 'cos[pitch]', 'cos[yaw]',
        #                                             'F_den', 'M_den'], dtype=float)
        NormalizedData['w_tot'] = n_omega_tot.reshape(-1)
        NormalizedData[['w1', 'w2', 'w3', 'w4']] = n_omega
        NormalizedData[['w2_1', 'w2_2', 'w2_3', 'w2_4']] = n_omega2
        NormalizedData[['p', 'q', 'r']] = self.npHstack((n_p, n_q, n_r))
        NormalizedData[['u_p', 'u_q', 'u_r']] = self.npVstack((u_p, u_q, u_r)).T
        NormalizedData[['roll', 'pitch', 'yaw']] = filteredData[['roll', 'pitch', 'yaw']]
        NormalizedData[['u', 'v', 'w']] = self.npVstack((u_bar, v_bar, w_bar)).T
        NormalizedData['v_in'] = vi_bar
        NormalizedData[['mu_x', 'mu_y', 'mu_z']] = self.npHstack((mux_bar, muy_bar, muz_bar))
        NormalizedData['mu_vin'] = mu_vi_bar
        NormalizedData['F_den'] = F_den.reshape(-1)
        NormalizedData['M_den'] = M_den.reshape(-1)

        # Replace NaNs
        # NormalizedData.fillna(0, inplace=True)

        # Add extra useful columns
        # Add abs(body velocity)
        NormalizedData['|u|'] = self.npAbs(NormalizedData['u'])
        NormalizedData['|v|'] = self.npAbs(NormalizedData['v'])
        NormalizedData['|w|'] = self.npAbs(NormalizedData['w'])

        # Add abs(body advance rations) 
        NormalizedData['|mu_x|'] = self.npAbs(NormalizedData['mu_x'])
        NormalizedData['|mu_y|'] = self.npAbs(NormalizedData['mu_y'])
        NormalizedData['|mu_z|'] = self.npAbs(NormalizedData['mu_z']) 

        # Add abs(rotational rates)
        NormalizedData['|p|'] = self.npAbs(NormalizedData['p'])
        NormalizedData['|q|'] = self.npAbs(NormalizedData['q'])
        NormalizedData['|r|'] = self.npAbs(NormalizedData['r'])

        # Add abs(control moments)
        NormalizedData['|u_p|'] = self.npAbs(NormalizedData['u_p'])
        NormalizedData['|u_q|'] = self.npAbs(NormalizedData['u_q'])
        NormalizedData['|u_r|'] = self.npAbs(NormalizedData['u_r'])

        # Get trigonometric identities of attitude angles 
        NormalizedData['sin[roll]'] = self.npSin(NormalizedData['roll'])
        NormalizedData['sin[pitch]'] = self.npSin(NormalizedData['pitch'])
        NormalizedData['sin[yaw]'] = self.npSin(NormalizedData['yaw'])

        NormalizedData['cos[roll]'] = self.npCos(NormalizedData['roll'])
        NormalizedData['cos[pitch]'] = self.npCos(NormalizedData['pitch'])
        NormalizedData['cos[yaw]'] = self.npCos(NormalizedData['yaw'])        

        return NormalizedData


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



# ================================================================================================================================ #
# Main script
# ================================================================================================================================ #
# Load model
# Path to parent folder of SysID model folder
cwd = os.getcwd()
modelPath = os.path.join(cwd, 'exampleData')
# Name of SysID model folder
mdlID = 'exampleModel'


# Load models from folder, adds Fx_Model_Poly, Fy_Model_Poly, ..., Mz_Model_Poly to local variables. 
Fx_Model_Poly = SysID.Model.load(os.path.join(modelPath, mdlID, 'Fx', 'Poly'))
Fy_Model_Poly = SysID.Model.load(os.path.join(modelPath, mdlID, 'Fy', 'Poly'))
Fz_Model_Poly = SysID.Model.load(os.path.join(modelPath, mdlID, 'Fz', 'Poly'))
Mx_Model_Poly = SysID.Model.load(os.path.join(modelPath, mdlID, 'Mx', 'Poly'))
My_Model_Poly = SysID.Model.load(os.path.join(modelPath, mdlID, 'My', 'Poly'))
Mz_Model_Poly = SysID.Model.load(os.path.join(modelPath, mdlID, 'Mz', 'Poly'))

# Save models
savePath = os.path.join(modelPath, 'standalonePolynomials', mdlID)
if not os.path.exists(savePath):
    os.makedirs(savePath)

# If saveGeneral is true, then the drone specific processing functions will also be saved to the model file,
#   such that processing can be reproduced when working only with the filtered state and rotor speeds.
# If saveGeneral is False, then only the model will be stored. Thus, any processing required to obtain the
#   model inputs (as done for during the training of the model) must be done prior to making predictions. 
saveGeneral = True
# Drone configuration directory and filename (if using DronePolynomialModel)
droneConfigurationFile = os.path.join(modelPath, 'MetalBeetelConfig.json')


if saveGeneral:
    # Fx 
    polyModel = DronePolynomialModel(droneConfigurationFile)
    polyModel.extractModel(locals()['Fx_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Fx.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # Fy
    polyModel = DronePolynomialModel(droneConfigurationFile)
    polyModel.extractModel(locals()['Fy_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Fy.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # Fz
    polyModel = DronePolynomialModel(droneConfigurationFile)
    polyModel.extractModel(locals()['Fz_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Fz.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)


    # Mx 
    polyModel = DronePolynomialModel(droneConfigurationFile)
    polyModel.extractModel(locals()['Mx_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Mx.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # My
    polyModel = DronePolynomialModel(droneConfigurationFile)
    polyModel.extractModel(locals()['My_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-My.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # Mz
    polyModel = DronePolynomialModel(droneConfigurationFile)
    polyModel.extractModel(locals()['Mz_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Mz.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)


else:
    # Fx 
    polyModel = PolynomialModel()
    polyModel.extractModel(locals()['Fx_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Fx.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # Fy
    polyModel = PolynomialModel()
    polyModel.extractModel(locals()['Fy_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Fy.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # Fz
    polyModel = PolynomialModel()
    polyModel.extractModel(locals()['Fz_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Fz.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)


    # Mx 
    polyModel = PolynomialModel()
    polyModel.extractModel(locals()['Mx_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Mx.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # My
    polyModel = PolynomialModel()
    polyModel.extractModel(locals()['My_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-My.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)

    # Mz
    polyModel = PolynomialModel()
    polyModel.extractModel(locals()['Mz_Model_Poly'])
    with open(os.path.join(savePath, mdlID + '-Mz.pkl'), 'wb') as f:
        pickle.dump(polyModel, f)


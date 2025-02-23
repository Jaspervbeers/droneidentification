% Clear all and reset to default paths, we add necessary paths upon init 
close all; clear all; path(pathdef); clc;

% Define underlying model and add necessary paths
modelID = 'MDL-HDBeetle-NN-II-NOP008-G-Quadratic_TEST';

addpath("models");
addpath(fullfile("models", modelID));

% Initialize model
[modelParams, droneParams] = model_init(modelID);
droneParams.kappaFz = 8.720395416240164e-07; %TODO Derive this in toMATLAB.py 
%TODO Derive this in toMATLAB.py
droneParams.M2OmegaMapping = [
    [2.99e-08, 2.99e-08, 2.99e-08, 2.99e-08];
    [3.69e-08, 3.69e-08, 3.69e-08, 3.69e-08];
    [4.29e-09, 4.29e-09, 4.29e-09, 4.29e-09]
].*double(droneParams.signMask);

droneParams.FM2OmegaMapping = inv([droneParams.M2OmegaMapping; ones(1, 4)*droneParams.kappaFz]);

Tmax = 10;
dt = 0.001;
time = (0:dt:Tmax);

% Define initial values
% State = [attitude, velocity, rates, position]
%   - attitude = [roll, pitch, yaw] in rad
%   - velocity = [u, v, w] in m/s along body x, y, z respectively
%   - rates = [p, q, r] in rad/s about body x, y, z respectively
%   - position = [x, y, z] in m in the earth-frame! (All other vars in
%   body frame)
% Omega = [w1, w2, w3, w4], rotor speeds in eRPM
initialState = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]';
initialOmega = [droneParams.minRPM, droneParams.minRPM, ... 
                droneParams.minRPM, droneParams.minRPM]';

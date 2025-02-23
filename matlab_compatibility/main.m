% Clear all and reset to default paths, we add necessary paths upon init 
close all; clear all; path(pathdef); clc;

% Define underlying model and add necessary paths
modelID = 'MDL-HDBeetle-EXAMPLE';
addpath("models");
addpath(fullfile("models", modelID));

% Initialize model
[modelParams, droneParams] = model_init(modelID);


%% Simulation (simpleModelSim.slx)
% Time
Tmax = 10;
dt = 0.004;
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

% Define reference: [x, y, z, yaw]
reference_ = zeros(length(time), 4);
reference_(:, 1) = 3*sin(2*pi*time/3);
reference_(:, 2) = 5*sin(2*pi*time/5);
reference_(:, 3) = -2;
reference = struct;
reference.time = time;
reference.signals.values = reference_;
reference.signals.dimensions = 4;

% Run simulation
% If visualization does not show, go to simpleModelSim.slx, open the 
% visualization block (yellow), open "UAV Animation" and click 
% "Show animation". Run main.m again (do not close animation window!)
simout = sim('simpleModelSim', 'StartTime', '0', 'StopTime', num2str(Tmax));

yout = simout.yout(1);
simdata = yout{1}.Values;



% %% For baseModel.slx
% omega = struct;
% omega.time = simdata.omega.Time(4:end);
% omega.signals.values = simdata.omega.Data(4:end, :);
% omega.signals.dimension = 4;
% 
% state = struct;
% state.time = simdata.omega.Time(4:end);
% state.signals.values = [simdata.attitude.Data(4:end, :), ...
%     simdata.velocity.Data(4:end, :), ...
%     simdata.rate.Data(4:end, :), ...
%     simdata.position.Data(4:end, :)];
% state.signals.dimension = 12;
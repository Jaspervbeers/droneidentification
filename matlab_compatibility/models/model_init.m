function [modelParams, droneParams] = model_init(modelID)
    % Initialize a quadrotor polynomial model
    % Loads various parameters of the model and drone configuration
    
    % Add model to path
    addpath(fullfile("models", modelID));

    % Load modelParams
    modelParams = load(fullfile("models", modelID, "modelParams.mat"));
    
    % Load droneParams
    droneParams = load(fullfile("models", modelID, "droneParams.mat"));
    
    % Define mapping from forces and moments to rotor speeds
    droneParams.FM2OmegaMapping = inv([droneParams.M2OmegaMapping; ones(1, 4)*droneParams.kappaFz]);
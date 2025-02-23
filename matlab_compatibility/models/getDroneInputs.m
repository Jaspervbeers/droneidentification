function [droneInputs] = getDroneInputs(state, omega, modelParams, droneParams)
    % Get the necessary force and moment model inputs from the state and
    % input vectors.
    %
    % State vector, state
    % state = [roll, pitch, yaw, u, v, w, p, q, r, x, y, z]
    %   - Attitude: [roll, pitch, yaw] in rad
    %   - (Body) velocity: [u, v, w] in m/s
    %   - (Body) rotational rates: [p, q, r] in rad/s
    %   - Position: [x, y, z] in E-frame, in m
    % Input vector, omega
    % omega = [w1, w2, w3, w4]
    %   - wi = rotor speed, in eRPM, of rotor i. See droneParams for
    %   mapping

    % Define input structure
    droneInputs = struct;
    
    % Extract rotor speeds
    w1 = double(omega(1)*2*pi/60);
    w2 = double(omega(2)*2*pi/60);
    w3 = double(omega(3)*2*pi/60);
    w4 = double(omega(4)*2*pi/60);
    
    % Get square of rotor speeds
    w2_1 = w1^2;
    w2_2 = w2^2;
    w2_3 = w3^2;
    w2_4 = w4^2;

    % Add state information
    % Attitude
    roll = double(state(1));
    pitch = double(state(2));
    yaw = double(state(3));
    
    % Velocity
    u = double(state(4));
    v = double(state(5));
    w = double(state(6));

    % Rates
    p = double(state(7));
    q = double(state(8));
    r = double(state(9));

    % Positions
    x = double(state(10));
    y = double(state(11));
    z = double(state(12));
    

    % Define normalizing struct
    normalizers = struct;
    % Determine normalizing values
    w_avg = sqrt((w2_1 + w2_2 + w2_3 + w2_4)/double(droneParams.nRotors));
    % Replace 0 with NaN
    if w_avg < droneParams.minRPM*2*pi/60
        w_avg = NaN;
    end
    if modelParams.isNormalized
        normalizers.w_avg = w_avg;
        normalizers.w_tot = droneParams.maxRPM*2*pi/60;
        normalizers.w2_avg = w_avg^2;
        normalizers.rate = (droneParams.R * w_avg)/droneParams.b;
        normalizers.mu = droneParams.R*w_avg;
        normalizers.vel = sqrt(u^2 + v^2 + w^2);
        normalizers.force = double(0.5*modelParams.rho          ... 
            *(droneParams.nRotors*pi*droneParams.R^2)           ...
            *(droneParams.R*(droneParams.maxRPM*2*pi/60)^2/(droneParams.maxRPM*2*pi/60 + w_avg))^2);
        normalizers.moments = normalizers.force * (1/droneParams.b);
    else
        normalizers.w_avg = 1.0;
        normalizers.w_tot = 1.0;
        normalizers.w2_avg = 1.0;
        normalizers.rate = 1.0;
        normalizers.mu = droneParams.R*w_avg;
        normalizers.vel = 1.0;
        normalizers.force = 1.0;
        normalizers.moment = 1.0;
    end
    
    % Populate droneInputs
    droneInputs = struct;

    % Rotor speed information
    droneInputs.w1 = w1/normalizers.w_avg;
    droneInputs.w2 = w2/normalizers.w_avg;
    droneInputs.w3 = w3/normalizers.w_avg;
    droneInputs.w4 = w4/normalizers.w_avg;

    droneInputs.w_tot = (w1 + w2 + w3 + w4)/normalizers.w_tot;

    droneInputs.w2_1 = w2_1/normalizers.w_avg^2;
    droneInputs.w2_2 = w2_2/normalizers.w_avg^2;
    droneInputs.w2_3 = w2_3/normalizers.w_avg^2;
    droneInputs.w2_4 = w2_4/normalizers.w_avg^2;

    % State information
    droneInputs.roll = roll;
    droneInputs.pitch = pitch;
    droneInputs.yaw = yaw;

    droneInputs.u = u/normalizers.vel;
    droneInputs.v = v/normalizers.vel;
    droneInputs.w = w/normalizers.vel;
    
    % Advance ratio
    droneInputs.mu_x = u/normalizers.mu;
    droneInputs.mu_y = v/normalizers.mu;
    droneInputs.mu_z = w/normalizers.mu;

    droneInputs.p = p/normalizers.rate;
    droneInputs.q = q/normalizers.rate;
    droneInputs.r = r/normalizers.rate;

    % Force and moment normalizers
    droneInputs.Fden = normalizers.force;
    droneInputs.Mden = normalizers.moment;
    

    % Pre-allocate entries in droneInputs, otherwise Simulink complains
    droneInputs.d_w1 = 0;
    droneInputs.d_w2 = 0;
    droneInputs.d_w3 = 0;
    droneInputs.d_w4 = 0;
    droneInputs.d_w_tot = 0;
    droneInputs.d_w2_1 = 0;
    droneInputs.d_w2_2 = 0;
    droneInputs.d_w2_3 = 0;
    droneInputs.d_w2_4 = 0;
    droneInputs.u_p = 0;
    droneInputs.u_q = 0;
    droneInputs.u_r = 0;
    droneInputs.U_p = 0;
    droneInputs.U_q = 0;
    droneInputs.U_r = 0;


    % Get difference from hover
    wHover = droneParams.wHover/normalizers.w_avg;
    droneInputs.d_w1 = droneInputs.w1 - wHover;
    droneInputs.d_w2 = droneInputs.w2 - wHover;
    droneInputs.d_w3 = droneInputs.w3 - wHover;
    droneInputs.d_w4 = droneInputs.w4 - wHover;

    droneInputs.d_w_tot = (droneInputs.d_w1 + droneInputs.d_w2 + droneInputs.d_w3 + droneInputs.d_w4)/normalizers.w_tot;

    droneInputs.d_w2_1 = droneInputs.d_w1^2;
    droneInputs.d_w2_2 = droneInputs.d_w2^2;
    droneInputs.d_w2_3 = droneInputs.d_w3^2;
    droneInputs.d_w2_4 = droneInputs.d_w4^2;
    

    % Control moments
    rotorSpeeds = [droneInputs.w1, droneInputs.w2, droneInputs.w3, droneInputs.w4];
    droneInputs.u_p = rotorSpeeds(droneParams.rotorConfig.frontLeft) ...
        + rotorSpeeds(droneParams.rotorConfig.aftLeft) ...
        - rotorSpeeds(droneParams.rotorConfig.frontRight) ...
        - rotorSpeeds(droneParams.rotorConfig.aftRight);

    droneInputs.u_q = rotorSpeeds(droneParams.rotorConfig.frontLeft) ...
        - rotorSpeeds(droneParams.rotorConfig.aftLeft) ...
        + rotorSpeeds(droneParams.rotorConfig.frontRight) ...
        - rotorSpeeds(droneParams.rotorConfig.aftRight);

    droneInputs.u_r = double(droneParams.r1Sign*(rotorSpeeds(droneParams.rotorConfig.frontLeft) ...
        - rotorSpeeds(droneParams.rotorConfig.aftLeft) ...
        - rotorSpeeds(droneParams.rotorConfig.frontRight) ...
        + rotorSpeeds(droneParams.rotorConfig.aftRight)));


    % NOTE the difference between u_i and U_i (w_i and w2_i respectively)
    rotorSpeeds2 = [droneInputs.w2_1, droneInputs.w2_2, droneInputs.w2_3, droneInputs.w2_4];
    droneInputs.U_p = rotorSpeeds2(droneParams.rotorConfig.frontLeft) ...
        + rotorSpeeds2(droneParams.rotorConfig.aftLeft) ...
        - rotorSpeeds2(droneParams.rotorConfig.frontRight) ...
        - rotorSpeeds2(droneParams.rotorConfig.aftRight);

    droneInputs.U_q = rotorSpeeds2(droneParams.rotorConfig.frontLeft) ...
        - rotorSpeeds2(droneParams.rotorConfig.aftLeft) ...
        + rotorSpeeds2(droneParams.rotorConfig.frontRight) ...
        - rotorSpeeds2(droneParams.rotorConfig.aftRight);

    droneInputs.U_r = double(droneParams.r1Sign*(rotorSpeeds2(droneParams.rotorConfig.frontLeft) ...
        - rotorSpeeds2(droneParams.rotorConfig.aftLeft) ...
        - rotorSpeeds2(droneParams.rotorConfig.frontRight) ...
        + rotorSpeeds2(droneParams.rotorConfig.aftRight)));

    



    

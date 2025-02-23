function [attDot] = PQR_to_AttDot_Euler(att, pqr)
    % Function to convert body angular rates to euler angles

    s_phi = sin(att(1));
    c_phi = cos(att(1));
    c_theta = cos(att(2));
    t_theta = tan(att(2));
   
    R = [
        [1, s_phi*t_theta, c_phi*t_theta];
        [0, c_phi, -s_phi];
        [0, s_phi/c_theta, c_phi/c_theta]
    ];

    attDot = R*pqr;
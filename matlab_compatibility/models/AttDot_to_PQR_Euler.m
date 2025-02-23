function [pqr] = AttDot_to_PQR_Euler(att, att_dot)

    s_phi = sin(att(1));
    c_phi = cos(att(1));
    c_theta = cos(att(2));
    s_theta = sin(att(2));
    
    R = [
      [1, 0, -1*s_theta];
      [0, c_phi, s_phi*c_theta];
      [0, -s_phi, c_phi*c_theta];
    ];

    pqr = R*att_dot;
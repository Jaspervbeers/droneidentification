function [quat] = Eul2Quat(theta)
    % Function to convert Euler angles into quaternions
    
    quat = zeros(4, 1);

    cr = cos(theta(1)*0.5);
    sr = sin(theta(1)*0.5);
    cp = cos(theta(2)*0.5);
    sp = sin(theta(2)*0.5);
    cy = cos(theta(3)*0.5);
    sy = sin(theta(3)*0.5);

    quat(1) = cr*cp*cy + sr*sp*sy;
    quat(2) = sr*cp*cy - cr*sp*sy;
    quat(3) = cr*sp*cy + sr*cp*sy;
    quat(4) = cr*cp*sy - sr*sp*cy;
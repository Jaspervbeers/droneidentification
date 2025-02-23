function [x_rot] = QuatRot(x, eul, direction)
    % Function to rotate a vector, x, by eul in direction 
    
    % Get quaternion representation of rotation
    quat = Eul2Quat(eul);

    % Determine rotation direction and extract quaternion values
    if direction == "B2E"
        q0 = quat(1);
        q1 = quat(2);
        q2 = quat(3);
        q3 = quat(4);
    elseif direction == "E2B"
        q0 = quat(1);
        q1 = -quat(2);
        q2 = -quat(3);
        q3 = -quat(4);
    else
        error('Unrecognized rotation direction, use "B2E" or "E2B"')
    end
    
    % Build rotation matrix
    R = [
      [(q0*q0 + q1*q1 - q2*q2 -q3*q3), (2*(q1*q2 - q0*q3)), (2*(q0*q2 + q1*q3))];
      [(2*(q1*q2 + q0*q3)), (q0*q0 - q1*q1 + q2*q2 - q3*q3), (2*(q2*q3 - q0*q1))];
      [(2*(q1*q3 - q0*q2)), (2*(q0*q1 + q2*q3)), (q0*q0 - q1*q1 - q2*q2 + q3*q3)]
    ];
    
    % Rotate vector
    x_rot = R*x;
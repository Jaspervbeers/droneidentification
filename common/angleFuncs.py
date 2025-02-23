import numpy as np

def quatMul(Q1, Q2):
    '''Function to multiply two quaternion arrays. Q2 is applied first, then Q1 (i.e. Q2 in global frame and Q1 in local after Q2 is applied) 
    
    :param Q1: First quaternion, as array with shape [N, 4] where N is the number of samples
    :param Q2: Second quaternion, as array with shape [N, 4] where N is the number of samples

    :return: Product of Q1 and Q2
    '''
    Q_out = np.array([[Q1[:, 0]*Q2[:, 0] - Q1[:, 1]*Q2[:, 1] - Q1[:, 2]*Q2[:, 2] - Q1[:, 3]*Q2[:, 3]],
                    [Q1[:, 0]*Q2[:, 1] + Q1[:, 1]*Q2[:, 0] + Q1[:, 2]*Q2[:, 3] - Q1[:, 3]*Q2[:, 2]],
                    [Q1[:, 0]*Q2[:, 2] - Q1[:, 1]*Q2[:, 3] + Q1[:, 2]*Q2[:, 0] + Q1[:, 3]*Q2[:, 1]],
                    [Q1[:, 0]*Q2[:, 3] + Q1[:, 1]*Q2[:, 2] - Q1[:, 2]*Q2[:, 1] + Q1[:, 3]*Q2[:, 0]]])
    return Q_out.T.reshape(-1, 4)



def QuatRot(q, x, rot='AsIs'):
    '''Function to rotate a vector using its quaternion representation. 

    :param q: Quaternion signal, as array with shape [N, 4], where N is the number of samples
    :param x: Signal to rotate, as array with shape [N, 3]
    :param rot: String indicating the order of rotation; options are "AsIs" (or 'B2E') or "reverse" (or 'E2B'). Default = 'AsIs' 
    
    :return: Rotated x
    '''
    if rot == 'B2E' or rot.lower() == 'asis':
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif rot == 'E2B' or rot.lower() == 'reverse':
        q0, q1, q2, q3 = q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]
    else:
        raise ValueError('specified rot is unknown. Use "AsIs" ("B2E") or "reverse" ("E2B") for rotation with the quaternion or its conjugate, respectively')

    # Define rotation matrices for each axis 
    R_1 = np.array([(q0*q0 + q1*q1 - q2*q2 -q3*q3), (2*(q1*q2 - q0*q3)), (2*(q0*q2 + q1*q3))])
    R_2 = np.array([(2*(q1*q2 + q0*q3)), (q0*q0 - q1*q1 + q2*q2 - q3*q3), (2*(q2*q3 - q0*q1))])
    R_3 = np.array([(2*(q1*q3 - q0*q2)), (2*(q0*q1 + q2*q3)), (q0*q0 - q1*q1 - q2*q2 + q3*q3)])

    # Manipulate the indices of the rotation matrices above to get a vector of form
    # N x [3 x 3] such that each element corresponds to the rotation matrix for that
    # specific sample and can therefore be multiplied directly with the acceleration array
    R_1 = R_1.T
    R_2 = R_2.T
    R_3 = R_3.T
    R_stack = np.zeros((3*len(R_1), 3))
    R_stack[0:(3*len(R_1)):3] = R_1
    R_stack[1:(3*len(R_1)):3] = R_2
    R_stack[2:(3*len(R_1)):3] = R_3
    R = R_stack.reshape((len(R_1), 3, 3))
    
    x_rot = np.matmul(R, x.reshape((len(x), -1, 1)))

    return x_rot.reshape(x.shape)



def QuatConj(quat):
    quatC = np.zeros(quat.shape)
    quatC[:, 0] = quat[:, 0]
    quatC[:, 1] = -1*quat[:, 1]
    quatC[:, 2] = -1*quat[:, 2]
    quatC[:, 3] = -1*quat[:, 3]
    return quatC



def Eul2Quat(theta):
    '''Function to convert euler angles to their quaternion equivalents
    
    :param theta: Array of the euler angles with shape (N, 3) or (3, N) where N is the number of samples

    :return: Quaternion representation of euler angles
    '''

    # Reshape theta
    theta = theta.reshape(-1, 3)

    quat = np.zeros((len(theta), 4))

    cr = np.cos(theta[:, 0]*0.5)
    sr = np.sin(theta[:, 0]*0.5)
    cp = np.cos(theta[:, 1]*0.5)
    sp = np.sin(theta[:, 1]*0.5)
    cy = np.cos(theta[:, 2]*0.5)
    sy = np.sin(theta[:, 2]*0.5)

    quat[:, 0] = cr*cp*cy + sr*sp*sy
    quat[:, 1] = sr*cp*cy - cr*sp*sy
    quat[:, 2] = cr*sp*cy + sr*cp*sy
    quat[:, 3] = cr*cp*sy - sr*sp*cy

    return quat



def Quat2Eul(quat):
    '''Function to convert quaternion representation of orientation to euler angles
    
    :param quat: Quaternion representation, as array with shape (N, 4), where N is the number of samples
    
    :return: Corresponding euler angles
    '''
    quat = quat.reshape(-1, 4)
    eul = np.zeros((len(quat), 3))
    eul[:, 0] = np.arctan2(2*(quat[:, 0]*quat[:, 1] + quat[:, 2]*quat[:, 3]), 1 - 2*(quat[:, 1]**2+quat[:, 2]**2))
    # Need to round to avoid issues where arcsin(x), with x = 1, is undefined due to floating point errors
    eul[:, 1] = np.arcsin(np.around(2*(quat[:, 0]*quat[:, 2] - quat[:, 3]*quat[:, 1]), 15))
    eul[:, 2] = np.arctan2(2*(quat[:, 0]*quat[:, 3] + quat[:, 2]*quat[:, 1]), 1 - 2*(quat[:, 2]**2+quat[:, 3]**2))
    return eul



def Quat2Mat(q):
    if len(q.shape) > 2:
        raise ValueError('Dimension of quaternion is too large, expected 1-d or 2-d arrays of size [N x 4]')
    if len(q.shape) == 1:
        r1 = np.array([q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])])
        r2 = np.array([2*(q[1]*q[2] + q[0]*q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*(q[2]*q[3] - q[0]*q[1])])
        r3 = np.array([2*(q[1]*q[3] - q[0]*q[2]), 2*(q[0]*q[1] + q[2]*q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2])
        R = np.vstack((r1, r2, r3))
    else:
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # Define rotation matrices for each axis 
        R_1 = np.array([(q0*q0 + q1*q1 - q2*q2 -q3*q3), (2*(q1*q2 - q0*q3)), (2*(q0*q2 + q1*q3))])
        R_2 = np.array([(2*(q1*q2 + q0*q3)), (q0*q0 - q1*q1 + q2*q2 - q3*q3), (2*(q2*q3 - q0*q1))])
        R_3 = np.array([(2*(q1*q3 - q0*q2)), (2*(q0*q1 + q2*q3)), (q0*q0 - q1*q1 - q2*q2 + q3*q3)])

        # Manipulate the indices of the rotation matrices above to get a vector of form
        # N x [3 x 3] such that each element corresponds to the rotation matrix for that
        # specific sample and can therefore be multiplied directly with the acceleration array
        R_1 = R_1.T
        R_2 = R_2.T
        R_3 = R_3.T
        R_stack = np.zeros((3*len(R_1), 3))
        R_stack[0:(3*len(R_1)):3] = R_1
        R_stack[1:(3*len(R_1)):3] = R_2
        R_stack[2:(3*len(R_1)):3] = R_3
        R = R_stack.reshape((len(R_1), 3, 3))
    return R



def wrapPi(angles):
    '''Function to constrain angles to [-pi, pi]
    
    :param angles: Array of angles to map to [-pi, pi]

    :return: Angles transformed to [-pi, pi]
    '''
    return (angles + np.pi) % (2*np.pi) - np.pi



def unwrapPi(angles):
    '''Function to unwrap angles from [-pi, pi]
    
    :param angles: Wrapped angles, confided to [-pi, pi]

    :return: Unwrapped angles
    '''
    return np.unwrap(angles)



def EulRotX(ang):
    '''Function to obtain matrix rotation about x-axis through an arbitrary angle 
    
    :param ang: Angle of rotation, in radians

    :return: Matrix, with shape (3, 3), corresponding to this rotation. 
    '''
    R = np.array([[1, 0, 0],
                  [0, np.cos(ang), -1*np.sin(ang)],
                  [0, np.sin(ang), np.cos(ang)]])
    return np.matrix(R)



def EulRotY(ang):
    '''Function to obtain matrix rotation about y-axis through an arbitrary angle 
    
    :param ang: Angle of rotation, in radians

    :return: Matrix, with shape (3, 3), corresponding to this rotation. 
    '''
    R = np.array([[np.cos(ang), 0, np.sin(ang)],
                  [0, 1, 0],
                  [-1*np.sin(ang), 0, np.cos(ang)]])
    return np.matrix(R)



def EulRotZ(ang):
    '''Function to obtain matrix rotation about z-axis through an arbitrary angle 
    
    :param ang: Angle of rotation, in radians

    :return: Matrix, with shape (3, 3), corresponding to this rotation. 
    '''
    R = np.array([[np.cos(ang), -1*np.sin(ang), 0],
                  [np.sin(ang), np.cos(ang), 0],
                  [0, 0, 1]])
    return np.matrix(R)



def EulRotX_arr(ang_arr):
    '''Function to obtain matrix rotation about x-axis through an arbitrary sequence of angles 
    
    :param ang: Array of angle rotations with shape (N, 1) where N is the number of samples

    :return: Matrix, with shape (N, 3, 3), corresponding to this rotation where N is the number of samples. 
    '''
    zeros = np.zeros(ang_arr.shape[0])
    ones = np.ones(ang_arr.shape[0])
    R = np.array([[ones, zeros, zeros],
                  [zeros, np.cos(ang_arr), -1*np.sin(ang_arr)],
                  [zeros, np.sin(ang_arr), np.cos(ang_arr)]])
    return np.transpose(R.T, (0, 2, 1))



def EulRotY_arr(ang_arr):
    '''Function to obtain matrix rotation about y-axis through an arbitrary sequence of angles 
    
    :param ang: Array of angle rotations with shape (N, 1) where N is the number of samples

    :return: Matrix, with shape (N, 3, 3), corresponding to this rotation where N is the number of samples. 
    '''    
    zeros = np.zeros(ang_arr.shape[0])
    ones = np.ones(ang_arr.shape[0])
    R = np.array([[np.cos(ang_arr), zeros, np.sin(ang_arr)],
                  [zeros, ones, zeros],
                  [-1*np.sin(ang_arr), zeros, np.cos(ang_arr)]])
    return np.transpose(R.T, (0, 2, 1))



def EulRotZ_arr(ang_arr):
    '''Function to obtain matrix rotation about y-axis through an arbitrary sequence of angles 
    
    :param ang: Array of angle rotations with shape (N, 1) where N is the number of samples

    :return: Matrix, with shape (N, 3, 3), corresponding to this rotation where N is the number of samples. 
    '''    
    zeros = np.zeros(ang_arr.shape[0])
    ones = np.ones(ang_arr.shape[0])
    R = np.array([[np.cos(ang_arr), -1*np.sin(ang_arr), zeros],
                  [np.sin(ang_arr), np.cos(ang_arr), zeros],
                  [zeros, zeros, ones]])
    return np.transpose(R.T, (0, 2, 1))
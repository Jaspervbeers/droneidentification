from numpy import hstack, isinf, zeros, isnan, where
from scipy.interpolate import interp1d



def rk4(func, x, u, t, dt, kwargs):
    ''' Fourth order Runge-Kutta numerical integration scheme

    :param func: Derivative of unknown function to use for integration
    :param x: State vector at step n
    :param u: Input vector at step n
    :param t: Time at step n
    :param dt: Time step size
    :param kwargs: Additional keyword arguments required by <func>
    :return: State vector at step n + 1 and associated time (i.e. t + dt)
    '''
    k1 = dt * func(x, u, t, **kwargs)
    k2 = dt * func(x + k1/2, u, t, **kwargs)
    k3 = dt * func(x + k2/2, u, t, **kwargs)
    k4 = dt * func(x + k3, u, t, **kwargs)

    x_new = x + k1/6 + k2/3 + k3/3 + k4/6

    return x_new, t + dt




def _finiteDifference(x, t = None, **kwargs):
    '''Utility function to take derivative of data through finite difference
    
    :param x: Data to differentiate, as 1-D array
    :param t: Associated time corresponding to samples of x, as 1-D array

    :return: Derivative of x (dx/dt) as numpy array
    '''
    if t is None:
        raise ValueError('Expected keyword argument <t>, but None was provided instead.')
    else:
        dt = t[1:] - t[0:-1]
        dx = x[1:] - x[0:-1]
        x_dot = dx/dt
    return hstack((0, x_dot))




def derivative(x, t):
    '''Function to numerically obtain the derivative of x with respect to t
    
    :param x: Signal to differentiate
    :param t: Signal with which x should be differentiated by

    :return: dx/dt
    '''
    xdot = zeros(x.shape)
    for i in range(len(t)-2):
        xdot[i + 1, :] = 0.5 * (x[i + 2, :] - x[i, :])/(0.5 * (t[i+2] - t[i]))
        if isinf(xdot[i + 1, :]).any():
            # Correct for inf, if any
            idx_undef = where(isinf(xdot[i + 1, :]))[0]
            xdot[i + 1, idx_undef] = xdot[i-1, idx_undef]

    # Missing first and last point, set them equal to nearest point
    xdot[0, :] = xdot[1, :]
    xdot[-1, :] = xdot[-2, :]
    # Interpolate NaNs, needs to be done per index in xdot
    for i in range(xdot.shape[1]):
        nanLocs = isnan(xdot[:, i])
        nanIdxs = where(nanLocs)[0]
        if len(nanIdxs) > 0:
            # Create a function which linearly interpolates the signal, based on known data, which takes
            # index as input -> i.e. y = f(index) -< can be thought of as a proxy for time
            func = interp1d(where(~nanLocs)[0], xdot[~nanLocs, i], kind = 'slinear')
            # Interpolate NaN indexes 
            xdot[nanIdxs, i] = func(nanIdxs)

    return xdot

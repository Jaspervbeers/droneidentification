from common import latexHelper

polyFx = [
    {
        'vars':['mu_x', '|mu_y|', 'mu_z'],
        'degree':3,
        'sets':[1]
    }
]
fixedFx = ['mu_x']
xVecFx = ['mu_x', '|mu_y|', 'mu_z']

latexHelper.polyCandsToEq('Fx', fixedFx, polyFx, 'simCands_Fx', coeffPrefix='x')


polyFy = [
    {'vars':['|mu_x|', 'mu_y', 'mu_z'],
    'degree':3,
    'sets':[1]}
]
fixedFy = ['mu_y']
xVecFy = ['|mu_x|', 'mu_y', 'mu_z']

latexHelper.polyCandsToEq('Fy', fixedFy, polyFy, 'simCands_Fy', coeffPrefix='y')


polyFz = [
    {
        'vars':['w', 'mu_z', 'w_avg', 'w_tot'],
        'degree':4,
        'sets':[1]
    },
    {'vars':['|u_p|', '|u_q|', 'u_r'],
    'degree':3,
    'sets':[1, 'w_avg']},
    {'vars':['|p|', '|q|', 'r'],
    'degree':3,
    'sets':[1, 'w_avg']}
]

fixedFz = ['(mu_x^(2) + mu_y^(2))', '(-1*w - v_in)']
xVecFz = ['w', '|mu_x|', '|mu_y|', 'mu_z', 'w_avg', 'w_tot', 'v_in', '|u_p|', '|u_q|', 'u_r', '|p|', '|q|', 'r']

latexHelper.polyCandsToEq('Fz', fixedFz, polyFz, 'simCands_Fz', coeffPrefix='z')


polyMx = [
    {'vars':['|mu_x|', 'mu_y', 'mu_z'],
        'degree':3,
        'sets':[1, 'p', 'u_p']},
    {'vars':['p', 'u_p'],
        'degree':3,
        'sets':[1, 'w_avg']}
]
fixedMx = ['u_p']
xVecMx = ['mu_y', 'mu_z', 'u_p', 'p', 'w_avg']

latexHelper.polyCandsToEq('Mx', fixedMx, polyMx, 'simCands_Mx', coeffPrefix='l')


polyMy = [
    {'vars':['mu_x', '|mu_y|', 'mu_z'],
        'degree':3,
        'sets':[1, 'q', 'u_q']},
        {'vars':['q', 'u_q'],
        'degree':3,
        'sets':[1, 'w_avg']}
]
fixedMy = ['u_q']
xVecMy =  ['mu_x', 'mu_z', 'u_q', 'q', 'w_avg']

latexHelper.polyCandsToEq('My', fixedMy, polyMy, 'simCands_My', coeffPrefix='m')


polyMz = [
    {'vars':['mu_x', 'mu_y', 'mu_z'],
    'degree':3,
    'sets':[1, '|p|', '|q|', 'r', '|u_p|', '|u_q|', 'u_r', 'w_avg']},
    {'vars':['|p|', '|q|', 'r'],
    'degree':3,
    'sets':[1, 'w_avg']},
    {'vars':['|u_p|', '|u_q|', 'u_r'],
    'degree':3,
    'sets':[1, 'w_avg']}
]    
fixedMz = ['u_r']
xVecMz = ['mu_x', 'mu_y', 'mu_z', 'p', 'q', 'r', 'u_p', 'u_q', 'u_r', 'w_avg']

latexHelper.polyCandsToEq('Mz', fixedMz, polyMz, 'simCands_Mz', coeffPrefix='n')


# Make ANN input vector table
latexHelper.ANNInputs2Table(
    ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], 
    [xVecFx, xVecFy, xVecFz, xVecMx, xVecMy, xVecMz]
)
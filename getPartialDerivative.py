import json
import subprocess
import os

with open('_getPartialDerivativeInputs.json', 'r') as f:
    args = json.load(f)

tmpDfDxFile = '_dfdx_dfdu.json'
DfDxFile = 'dfdx_dfdu.json'

# TODO: Make a file with 'rules' for replacement of state and input for viable alternatives. 
x = args['state']
u = args['input']
sup = args['extra']

_poly = [i.replace('bias', '1') for i in args['poly']]
poly = [p.replace('^', '**') for p in _poly]

for i, p in enumerate(poly):
    for _x in x:
        if r'|' + f'{_x}' + r'|' in p:
            p = p.replace(r'|' + f'{_x}' r'|', f'abs({_x})')
    for _u in u:
        if r'|' + f'{_u}' + r'|' in p:
            p = p.replace(r'|' + f'{_u}' r'|', f'abs({_u})')
    if len(sup):
        for _e in sup:
            if r'|' + f'{_e}' + r'|' in p:
                p = p.replace(r'|' + f'{_e}' r'|', f'abs({_e})')
    poly[i] = p

eqs = [f'{p}' for p in poly]

# Create file to compute partial derivatives 
with open('_getPartialDerivative.py', 'w') as f:
    f.write('from sympy import symbols, diff\n')
    f.write('import json\n')

    f.write('\n# Define symbols for STATE variables\n')
    xlhs = ''
    xrhs = ''
    for _x in x:
        xlhs += '{}, '.format(_x)
        xrhs += '{} '.format(_x)
    
    varDef = xlhs[:-2] + ' = ' + f'symbols("{xrhs[:-1]}", real = True)\n'
    f.write(varDef)


    f.write('\n# Define symboles for INPUT variables\n')
    ulhs = ''
    urhs = ''
    for _u in u:
        ulhs += '{}, '.format(_u)
        urhs += '{} '.format(_u)
    
    uDef = ulhs[:-2] + ' = ' + f'symbols("{urhs[:-1]}", real = True)\n'
    f.write(uDef)


    if len(sup):
        f.write('\n# Define any extra variables, not dependent on the state or input but a part of f(x,u)\n')
        elhs = ''
        erhs = ''
        for _sup in sup:
            elhs += '{}, '.format(_sup)
            erhs += '{} '.format(_sup)
        
        eDef = elhs[:-2] + ' = ' + f'symbols("{erhs[:-1]}", real = True)\n'
        f.write(eDef)


    f.write('\n')
    frhs = '['
    for eq in eqs:
       frhs += '{}, '.format(eq)
    
    frhs = frhs[:-2] + ']'
    f.write(f'__funcs = {frhs}\n')

    f.write('\n')
    f.write('__df_x = ' r'{}' + '\n')
    f.write(f'for __x in [{xlhs[:-2]}]:\n')
    f.write('\t__df_xi = []\n')
    f.write('\tfor __f in __funcs:\n')
    f.write(f'\t\t__df_xi.append(str(diff(__f, __x)))\n')
    f.write('\t__df_xi = [i.replace("**", "^") for i in __df_xi]\n')
    f.write(f'\t__df_x.update(' + r'{' + 'str(__x):__df_xi' + r'}' + ')\n')

    f.write('\n')
    f.write('__df_u = ' r'{}' + '\n')
    f.write(f'for __u in [{ulhs[:-2]}]:\n')
    f.write('\t__df_ui = []\n')
    f.write('\tfor __f in __funcs:\n')
    f.write(f'\t\t__df_ui.append(str(diff(__f, __u)))\n')
    f.write('\t__df_ui = [i.replace("**", "^") for i in __df_ui]\n')
    f.write(f'\t__df_u.update(' + r'{' + 'str(__u):__df_ui' + r'}' + ')\n')    


    f.write('\n')
    f.write('toFile = ' + r'{' + '\n' + '\t"dfdx":__df_x,\n\t"dfdu":__df_u\n' + r'}' + '\n')
    f.write(f'with open("{tmpDfDxFile}", "w") as f:\n')
    f.write('\tjson.dump(toFile, f, indent = 4)')


# Find derivatives
try:
    prcs = subprocess.check_call(['python', '_getPartialDerivative.py'])
except subprocess.CalledProcessError:
    os.remove('_getPartialDerivative.py')
    os.remove(tmpDfDxFile)
    raise RuntimeError('Could not compute partial derivatives, check that the states, inputs, and/or extra variables are fully and properly defined.')

# Load derivatives and convert to string eqn form
with open(tmpDfDxFile, 'r') as f:
    _dfdx_dfdu = json.load(f)

dfdx = _dfdx_dfdu['dfdx']
dfdu = _dfdx_dfdu['dfdu']

for v in dfdx.values():
    for i, elem in enumerate(v):
        for _x in x:
            if f'Abs({_x})' in elem:
                elem = elem.replace(f'Abs({_x})', r'|' + f'{_x}' + r'|')
            if f'sign({_x})' in elem:
                elem = elem.replace(f'sign({_x})', r'(|' + f'{_x}' + r'|' + f'/{_x})')
        for _u in u:
            if f'Abs({_u})' in elem:
                elem = elem.replace(f'Abs({_u})', r'|' + f'{_u}' + r'|')
        if len(sup):
            for _e in sup:
                if f'Abs({_e})' in elem:
                    elem = elem.replace(f'Abs({_e})', r'|' + f'{_e}' + r'|')
                if f'sign({_e})' in elem:
                    elem = elem.replace(f'sign({_e})', r'(|' + f'{_e}' + r'|' + f'/{_e})')                
        v[i] = elem 

for v in dfdu.values():
    for i, elem in enumerate(v):
        for _u in u:
            if f'Abs({_u})' in elem:
                elem = elem.replace(f'Abs({_u})', r'|' + f'{_u}' + r'|')
            if f'sign({_u})' in elem:
                elem = elem.replace(f'sign({_u})', r'(|' + f'{_u}' + r'|' + f'/{_u})')
        for _x in x:
            if f'Abs({_x})' in elem:
                elem = elem.replace(f'Abs({_x})', r'|' + f'{_x}' + r'|')
            if f'sign({_x})' in elem:
                elem = elem.replace(f'sign({_x})', r'(|' + f'{_x}' + r'|' + f'/{_x})')
        if len(sup):
            for _e in sup:
                if f'Abs({_e})' in elem:
                    elem = elem.replace(f'Abs({_e})', r'|' + f'{_e}' + r'|')
                if f'sign({_e})' in elem:
                    elem = elem.replace(f'sign({_e})', r'(|' + f'{_e}' + r'|' + f'/{_e})')
        v[i] = elem

dfdx_dfdu = {
    'dfdx':dfdx,
    'dfdu':dfdu
}

with open(DfDxFile, 'w') as f:
    json.dump(dfdx_dfdu, f, indent = 4)

# Clean up files
os.remove(tmpDfDxFile)
os.remove('_getPartialDerivative.py')
import numpy as np

class lhParams:

    def __init__(self):
        self.mappings = {
            'mu_x':r'\mu_{x}',
            'mu_y':r'\mu_{y}',
            'mu_z':r'\mu_{z}',
            'u_p':r'U_{p}',
            'u_q':r'U_{q}',
            'u_r':r'U_{r}',    
            'mu_vin':r'\mu_{v_{in}}',
            'w1':r'\omega_{1}',
            'w2':r'\omega_{2}',
            'w3':r'\omega_{3}',
            'w4':r'\omega_{4}',
            'w1_2':r'\omega_{1}^{2}',
            'w2_2':r'\omega_{2}^{2}',
            'w3_2':r'\omega_{3}^{2}',
            'w4_2':r'\omega_{4}^{2}',
            'w_avg':r'\omega_{avg}',
            'w_tot':r'\omega_{total}',
            'sin[roll]':r'\sin{(\phi)}',
            'sin[pitch]':r'\sin{(\theta)}',
            'sin[yaw]':r'\sin{(\psi)}',
            'cos[roll]':r'\cos{(\phi)}',
            'cos[pitch]':r'\cos{(\theta)}',
            'cos[yaw]':r'\cos{(\psi)}',
            'roll':r'\phi',
            'pitch':r'\theta',
            'yaw':r'\psi',
            '*':r'\cdot ',
            '(':r'{',
            ')':r'}'
        }

    def addMap(self, key, latex):
        self.mapping.update({key:latex})

helperParams = lhParams()


def LatexifyRegressor(regressor):
    '''
    Convert string-based regressor to LaTeX code, using knowledge base from lhParams.mappings (Can add additional mappings through lhParams.addMap())

    Inputs:
        - regressor -- string -- Regressor to convert to LaTeX code

    Outputs:
        - regressor -- (raw) string -- Converted LaTeX regressor
    '''
    for m in helperParams.mappings.keys():
        if m in regressor:
            regressor = regressor.replace(m, helperParams.mappings[m])
            regressor = regressor.replace('.0', '')
    return regressor


def _latexMakeBF(text):
    '''
    Make text bold, using LaTeX syntax

    Inputs:
    - text -- string -- text to make bold

    Outputs:
    - boldedText -- (raw) string -- bolded text
    '''
    return r'\textbf' + '{' + text + '}'


def _latexMakeRow(elements):
    '''
    Convert a list of elements into a LaTex table row

    Inputs:
    - elements -- list -- list of strings where each element in the list corresponds to the string belonging to a cell

    Outputs:
    - row -- (raw) string -- elements expressed as a LaTeX table row row
    '''
    row = r"{}"
    row += r" &{}"*(len(elements)-1) + r' \\'
    row = row.format(*elements)
    return row


def PolyCands2Eq(fm, fixedRegs, cands, filename = None, eqLabel = 'eq:myLabel', charLim = 150, coeffPrefix = None):
    '''
    Convert (SysID) polynomial candidate regressors to LaTeX equations

    Inputs:
    - fm -- string -- Left-hand-side of equation (e.g. Fx, My, Y)
    - fixedRegs -- list -- List of strings where each element is a fixed regressor of the polynomial model
    - cands -- list (of dicts) -- list of dictionaries describing the candidate regressors, using the SysID.Model('Stepwise_Regression') syntax. Each dict should have the entries 'vars', 'degree', and 'sets'.
    - filename -- string -- (Path to and) Name of the output .txt file. If None, filename will be polyCands-fm.txt, default = None.
    - eqLabel -- string -- Label for the LaTeX equation. Default = 'eq:myLabel'
    - charLim -- int -- Character limit for each equation row. If the total equation exceeds this, a new line will be created to continue the eq.
    - coeffPrefix -- string -- Prefix for the fixed regressor coefficients, default is None. If None, the first character of fm will be used as the coefficient prefix.

    Outputs:
    - File with LaTeX equation
    '''
    if coeffPrefix is None:
        coeffPrefix = fm[0]
    if filename is None:
        filename = f'polyCands-{fm}.txt'
    with open(filename + '.txt', 'w') as f:
        f.write(r'\begin{equation}\label{eq:' + eqLabel + r'}' + '\n')
        fixedRegsString = _convertFixedRegs2String(fixedRegs, coeffPrefix)
        polysStrings = _convertPolyCands2String(cands)
        eqString = fixedRegsString + polysStrings
        f.write(r'\begin{array}{rl}' + '\n')
        f.write(r'\hat{' + fm + r'} = ')
        writingArray = True
        maxIter = 50
        counter = 0
        while writingArray:
            idxsPlus = np.where(np.array(list(eqString)) == '+')[0]
            if len(idxsPlus):
                idxsClosest = idxsPlus[np.where(idxsPlus - charLim < 0)[0][-1]]
                if idxsClosest == 0:
                    if len(idxsPlus) > 1:
                        idxsClosest = idxsPlus[1]
                    else:
                        idxsClosest = len(eqString) + 1
                rowString = eqString[:idxsClosest-1]
                f.write(r'& ' + rowString)
                eqString = eqString[idxsClosest:]
                counter += 1
            else:
                writingArray = False
                break
            if counter >= maxIter:
                writingArray = False
                break
            f.write(r'\\' + '\n')
        f.write(r'\end{array}' + '\n')
        f.write(r'\end{equation}')


def _convertPolyCands2String(polyCands):
    '''
    Convert list of dictionaries describing polynomial candidate regressors to a single string

    Inputs:
    - polyCands -- list (of dicts) -- list of dictionaries describing the candidate regressors, using the SysID.Model('Stepwise_Regression') syntax. Each dict should have the entries 'vars', 'degree', and 'sets'.
    
    Outputs:
    - polyCandString -- string -- Single string of polynomial candidate regressors, written in compact form. 
    '''
    string = ' + '
    for p in polyCands:
        Pstring = f'P^{p["degree"]}(' + LatexifyRegressor(','.join(p['vars'])) + ')'
        Sstring = LatexifyRegressor('*') + r'\{' + ','.join([LatexifyRegressor(str(i)) for i in p['sets']]) + r'\}'
        string += Pstring + Sstring + ' + '
    return string[:-3]


def _convertFixedRegs2String(fixedRegs, coeffPrefix):
    '''
    Convert list of fixed regressors to a single string

    Inputs:
    - fixedRegs -- list -- List of strings where each element is a fixed regressor of the polynomial model
    - coeffPrefix -- string -- Prefix for the fixed regressor coefficients.
    
    Outputs:
    - fixedString -- string -- Single string of polynomial fixed regressors.
    '''
    string = ''
    for i, fr in enumerate(fixedRegs):
        if fr == 'bias':
            string += coeffPrefix.upper() + r'_{' + f'{i}' + r'}' + ' + '
        else:
            string += coeffPrefix.upper() + r'_{' + f'{i}' + r'}' + LatexifyRegressor(f'*{fr}') + ' + '
    return string[:-3]


def PolyStructure2Table(Models, which, filename = 'PolyStructure', addTableWrapper = True, caption = None, label = None):
    '''
    Convert identified polynomial structures (from stepwise regression) to a LaTeX table with associated regressor coefficients and coefficients of determination (R2)

    Inputs:
    - Models -- dict -- Dictionary where each element is a stepwise_regression SysID.Model object. The keys correspond to the model handle (e.g. Fx). 
    - which -- list -- List of strings that correspond to the keys in <Models> that should be tabulated.
    - filename -- string -- (Path to and) Name of the output .txt file. Default = PolyStructure.txt
    - addTableWrapper -- boolean -- Indicates if the full table environment should be created. If False, only the tabular environment will be written to the file.
    - caption -- string -- Caption of the table. Only relevant if <addTableWrapper>=True. Default = None. If None, default descriptive caption is used.
    - label -- string -- label of the table. Only relevant if <addTableWrapper>=True. Default = None. If None, <filename> is used. Note the "tab:" prefix is always added. 

    Outputs:
    - A file <filename>.txt containing the LaTeX code for the table. 
    '''
    # First create table layout 
    comb = '-'.join(which)
    with open(filename + f'_{comb}.txt', 'w') as f:
        if addTableWrapper:
            f.write(r'\begin{table}[!h]' + '\n')
            f.write(r'\centering' + '\n')
            if caption is None:
                caption = f'Selected regressors, associated coefficients, and goodness-of-fit metrics for the identified polynomial models of: {", ".join(which)}. Fixed regressors are highlighted in grey. The rows indicate the order of selection.'
            f.write(r'\caption{' + caption + '} \n')
            if label is None:
                label = filename
            f.write(r'\label{tab:' + label + '}')
        f.write(r'\begin{tabular}' + '{' + ('{}|'.format('c'*3)*len(which))[:-1] + '}' + '\n')
        header = []
        subHeader = []
        for w in which:
            header.append(r'\multicolumn{3}{c}'+ '{' + _latexMakeBF(w) + '}')
            _subheader = [
                "Regressor",
                "Coefficient",
                "R2"
            ]
            subHeader += _subheader

        f.write(_latexMakeRow(header) + r' \hline' + ' \n')
        f.write(_latexMakeRow(subHeader) + r' \hline' + ' \n')
        building = True
        idx = 0
        flags = [False,]*len(which)
        while building:
            row = []
            for i, w in enumerate(which):
                try:
                    CurrentRegressor = Models[w].TrainedModel['Model']['Regressors'][idx]
                    Coefficient = '{:.3e}'.format(Models[w].TrainedModel['Model']['Parameters'][idx].__array__()[0][0])
                    AddLog = Models[w].TrainedModel['Additional']['Log']['Add Log']
                    InfoLog = Models[w].TrainedModel['Additional']['Log']['Info Log']
                    ReverseAddLog = {v:k for k, v in AddLog.items()}
                    if CurrentRegressor not in AddLog.values():
                        addColor = r'\cellcolor[HTML]{DCDCDC}'
                        R2 = '-'
                    else:
                        addColor = ' '
                        R2 = '{:.2f}'.format(InfoLog[ReverseAddLog[CurrentRegressor]]['R2'].__array__()[0][0])
                except IndexError:
                    CurrentRegressor = '-'
                    Coefficient = '-'
                    R2 = '-'
                    addColor = ' '
                    flags[i] = True
                row += [
                    addColor + r'$' + LatexifyRegressor(CurrentRegressor) + r'$',
                    addColor + Coefficient, 
                    addColor + R2
                ]
            if all(flags):
                building = False
            else:
                idx += 1
                f.write(_latexMakeRow(row) + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}')
        if addTableWrapper:
            f.write(r'\end{table}')


def ANNInputs2Table(fms, xVecs, filename = 'ANNInputs'):
    '''
    Converts ANN input vectors into a LaTeX table

    Inputs:
    - fms -- list -- list of strings which correspond to the model names to tabulate, should match <xVecs>
    - xVecs -- list -- list of lists where each sublist contains the input variables, as strings, for the associated elements of <fms>
    - filename -- string -- (Path to and) Name of the output .txt file. Default = ANNInputs.txt

    Outputs:
    - A file <filename>.txt containing the LaTeX code for the table. 
    '''
    if len(fms) != len(xVecs):
        raise ValueError(f'Dimension mismatch: Length Forces/Moments (={len(fms)}) =/= Length xVectors (={xVecs})')
    with open(f'{filename}.txt', 'w') as f:
        f.write(r'\begin{tabular}{rl}')
        header = [
            _latexMakeBF('Model'),
            _latexMakeBF('Input variables')
        ]
        f.write(_latexMakeRow(header) + r' \hline' + '\n')
        for i, fm in enumerate(fms):
            row = [r'\multicolumn{1}{r|}' + '{' + fm + '}']
            string = ''
            for x in xVecs[i]:
                string += '$' + LatexifyRegressor(x) + '$, '
            row.append(string[:-2])
            f.write(_latexMakeRow(row) + '\n')
        f.write(r'\hline')
        f.write(r'\end{tabular}')

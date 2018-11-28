'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import pandas as pd
import scipy as sp
from lifelines.utils import concordance_index

def c_index(solution, prediction):
    '''Concordance index from the lifelines library
    '''
    # the concordance_index function requires a panda dataFrame
    df = pd.DataFrame(solution, columns=['time', 'event'])
    return concordance_index(df['time'], prediction, df['event'])

def mse_metric(solution, prediction):
    '''Mean-square error.
    Works even if the target matrix has more than one column'''
    df = pd.DataFrame(solution, columns=['time', 'event'])
    mse = np.mean((solution-prediction)**2)
    return np.mean(mse)

# def mse_metric(solution, prediction):
#     '''Mean-square error.
#     Works even if the target matrix has more than one column'''
#     mse = np.mean((solution-prediction)**2)
#     return np.mean(mse)

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def z_calc(y, kdir, krev, v_f, v_b):
    """
    Calculates reversibility of all elementary reactions present in the reaction mechanism.        
    Args:
        y(nparray): Steady state surface coverage at desired reaction conditions.
        kdir,krev(list): Kinetic constants.       
    Returns:
        List with reversibility of elementary reactions.
    """
    rd = np.zeros(len(kdir))
    ri = np.zeros(len(krev))
    for reaction in range(len(rd)):
        rd[reaction] = kdir[reaction] * np.prod(y ** v_f[:, reaction])
        ri[reaction] = krev[reaction] * np.prod(y ** v_b[:, reaction])
    return ri / rd

def calc_eapp(temperature_vector, reaction_rate_vector):
    """
    Function that evaluates the apparent activation energy of a global reaction.
    Args:
        temperature_vector(ndarray): Array containing the studied temperature range in Kelvin
        reaction_rate_vector(ndarray): Array containing the reaction rate at different temperatures            
    Returns:
        Apparent reaction energy in kJ/mol in the temperature range of interest.      
    """
    lm = LinearRegression()
    x = pd.DataFrame(1 / temperature_vector)
    y = pd.DataFrame(np.log(reaction_rate_vector))
    reg = lm.fit(x, y)
    Eapp = -(8.31439 / 1000.0) * reg.coef_[0, 0]  # kJ/mol
    R2 = reg.score(x, y)
    return Eapp, R2

def calc_reac_order(partial_pressure, reaction_rate):
    """
    Function that evaluates the apparent reaction order for a specific
    gas species for the selected global reaction.
    Args:
        partial_pressure(ndarray): Partial pressure of the gas species [Pa]
        reaction_rate(ndarray): Reaction rate [1/s]            
    Returns:
        Apparent reaction order with respect to the selected species
    """
    lm = LinearRegression()
    x = pd.DataFrame(np.log(partial_pressure))
    y = pd.DataFrame(np.log(reaction_rate))
    reg = lm.fit(x, y)
    napp = reg.coef_[0, 0]
    R2 = reg.score(x, y)
    return napp, R2

def calc_tafel_slope(overpotential, current_density):
    """
    Function for evaluating the Tafel slope of an electrochemical system.
    Args:
        overpotential(ndarray): applied overpotential array [V]
        current_density(ndarray): current density array[mA cm-2]            
    Returns:
        Apparent reaction order with respect to the selected species
    """
    lm = LinearRegression()
    xx = []
    yy = []
    for i in range(len(overpotential)):
        if overpotential[i] < -0.1:
            yy.append(overpotential[i])
            xx.append(current_density[i])
    xx = np.array(xx)
    yy = np.array(yy)
    y = pd.DataFrame(yy)
    x = pd.DataFrame(np.log10(abs(xx)))
    reg = lm.fit(x, y)
    m = reg.coef_[0, 0]
    R2 = reg.score(x, y)
    return m, R2

def stoic_forward(matrix):
    """
    Filter function for the stoichiometric matrix.
    Negative elements are considered and changed of sign in order to 
    compute the direct reaction rates.
    """
    mat = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if matrix[i][j] < 0:
                mat[i][j] = - matrix[i][j]
    return mat

def stoic_backward(matrix):
    """
    Filter function for the stoichiometric matrix.
    Positive elements are considered and kept in order to compute 
    the reverse reaction rates.
    """
    mat = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if matrix[i][j] > 0:
                mat[i][j] = matrix[i][j]
    return mat

def net_rate(y, kd, ki, v_f, v_b):
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
            kd, ki(ndarray): kinetic constants of the direct/reverse elementary reactions.
        Returns:
            ndarray containing the net reaction rate for all the steps [1/s].
        """
        net_rate = np.zeros(len(kd))
        net_rate = kd * np.prod(y ** v_f, axis=1) - ki * np.prod(y ** v_b, axis=1)
        return net_rate
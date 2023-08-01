import numpy as np
from sklearn.linear_model import LinearRegression
from constants import *
from rm_parser import stoic_forward, stoic_backward

def kinetic_coeff(NR, dg_reaction, dg_barrier, temperature, reaction_type, m, a_site=1e-19):
        """
        Returns the kinetic coefficient for direct and reverse elementary reactions, 
        based on transition state theory and collision theory (for adsorption).        
        Args: 
            NR(int): Number of elementary reactions.
            dg_reaction(ndarray): Array containing the Gibbs reaction energy of the 
                                  elementary reactions of the system.
            dg_barrier(ndarray): Array containing the Gibbs energy barrier of the
                                 direct elementary reactions.
            temperature(float): Temperature in [K].
            reaction_type(list): list of the kind of elementary reactions:
                                     - ads=adsorption
                                     - des=desorption
                                     - sur=surface reaction
            A_site_0(float): Area of the catalytic ensemble in [m2]. Default: 1e-19[m2]
        Returns:
            kd, kr (ndarray): arrays with kinetic constants for direct and reverse reactions.
        """
        Keq = np.zeros(NR)  # Equilibrium constant
        kd = np.zeros(NR)   # Direct constant
        kr = np.zeros(NR)   # Reverse constant
        for reaction in range(NR):
            Keq[reaction] = np.exp(-dg_reaction[reaction] / temperature / K_B)
            if reaction_type[reaction] == 'ads':
                A = a_site / (2 * np.pi * m[reaction] * K_BU * temperature)**0.5
                kd[reaction] = A * np.exp(-dg_barrier[reaction] / K_B / temperature)
                kr[reaction] = kd[reaction] / Keq[reaction]
            elif reaction_type[reaction] == 'des':
                A = (K_B * temperature / H)
                kd[reaction] = A * np.exp(-dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            else:  # Surface reaction
                A = (K_B * temperature / H)
                kd[reaction] = A * np.exp(-dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
        return kd, kr

def net_rate(y, kd, ki, v_matrix):
    """
    Returns the net reaction rate for each elementary reaction.
    Args:
        y(ndarray): surface coverage + partial pressures array [-/Pa].
        kd, kr(ndarray): kinetic constants of the direct/reverse steps.
        v_matrix(ndarray): stoichiometric matrix of the system.
    Returns:
        (ndarray): Net reaction rate of the elementary reactions [1/s].
    """
    net_rate = np.zeros(len(kd))
    v_ff = stoic_forward(v_matrix)
    v_bb = stoic_backward(v_matrix)
    net_rate = kd * np.prod(y ** v_ff.T, axis=1) - ki * np.prod(y ** v_bb.T, axis=1)
    return net_rate
    
def z_calc(y, kd, kr, v_f, v_b):
    """
    Calculates reversibility of all elementary reactions present in the reaction mechanism.        
    Args:
        y(ndarray): Steady state surface coverage at desired reaction conditions.
        kd(ndarray): Direct kinetic constants
        kr(ndarray): Reverse kinetic constants.       
    Returns:
        (ndarray): Reversibility of the elementary reactions.
    """
    rd = np.zeros(len(kd))
    ri = np.zeros(len(kr))
    for reaction in range(len(rd)):
        rd[reaction] = kd[reaction] * np.prod(y ** v_f[:, reaction])
        ri[reaction] = kr[reaction] * np.prod(y ** v_b[:, reaction])
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
    x = np.reciprocal(temperature_vector)    
    y = np.log(reaction_rate_vector)
    reg = lm.fit(x, y)
    Eapp = -(R/1000.0) * reg.coef_[0, 0]  # kJ/mol (typical unit of measure)
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
    x = np.log(partial_pressure)
    y = np.log(reaction_rate)
    reg = lm.fit(x, y)
    napp = reg.coef_[0, 0]
    R2 = reg.score(x, y)
    return napp, R2




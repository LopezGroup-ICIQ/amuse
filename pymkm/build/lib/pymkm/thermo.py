"""Thermodynamic module of Pymkm. It contains functions used to compute the thermodynamic state functions 
(H, S, G, Keq) starting from the NASA 7-polynomial coefficients from the \" third millenium ideal gas and 
condensed phase thermochemical database for combustion(E. Goos, A. Burcat, B. Ruscic)"""

import numpy as np

# Reference conditions and constants
T_ref = 298.0           # K
P_ref = 1.0             # bar
R = 8.314462175/1000.0  # kJ/mol/K

species = ['CO2', 'H2', 'CO', 'H2O', 'CH3OH', 'CH2Br2', 'CH3Br', 'CH4', 'Br2', 'HBr','C']

# H_ref in kJ/mol, S_ref in J/mol/K. 
H_ref = [-393.52, 0.0, -110.53, -241.83, -205.0, 3.8, -37.7, -74.6, 30.91, -36.44, 716.67]  
S_ref = [213.79, 130.68, 197.66, 188.44, 239.9, 293.43, 245.8, 186.25, 245.38, 198.70, 5.833] # unsure value for C  
# NASA polynomials coefficients, range 200-1000K, version released 7/30/99
a1 = np.array([2.35677352E0, 2.34422112E0,
              3.57953347E0, 4.19864056E0, 5.71539582E0,
              3.07810878, 3.61367184, 5.14911468,
              3.34350669, 3.4894141, 0.25542395E1])
a2 = np.array([8.98459677E-3, 7.98052075E-3, -6.10353680E-4, -2.03643410E-3, -1.52309129E-2,
              1.23681783E-2, -8.86540422E-4, -1.36622009E-2,
              6.35013278E-3, 0.27295667E-3,-0.32153772E-3])
a3 = np.array([-7.12356269E-6, -1.94781510E-5,
              1.01681433E-6, 6.52040211E-6, 6.52441155E-5,
              8.40317756E-7, 2.94669395E-5, 4.91453921E-5,
              -1.36341193E-5, -0.15997163E-5,0.73379223E-6])
a4 = np.array([2.45919022E-9, 2.01572094E-8,
              9.07005884E-10, -5.48797062E-9, -7.10806889E-8,
              -1.25546148E-8, -3.76504049E-8, -4.84246767E-8,
              1.31622796E-8, 0.33659948E-8, -0.73223487E-9])
a5 = np.array([-1.43699548E-13, -7.37611761E-12, -9.04424499E-13, 1.77197817E-12, 2.61352698E-11,
              6.79189724E-12, 1.49390354E-11, 1.6660344E-11,
              -4.67916478E-12, -0.16408428E-11, 0.26652144E-12])
a6 = np.array([-4.83719697E4, -9.17935173E2, -1.43440860E4, -3.02937267E4, -2.56427656E4,
              -8.59489686E6, -5.61401651E3, -1.02465983E4,
              2.53514183E3, -0.54089034E4, 0.85442681E5])
a7 = np.array([9.90105222E0, 6.83010238E-1,
              3.50840928E0, -8.49032208E-1, -1.50409823E0,
              1.41666382E1, 8.24978857E0, -4.63848842,
              9.07866893, 0.39796907E1, 0.45313085E1])

a_matrix = np.array([a1, a2, a3, a4, a5, a6, a7]).T

def enthalpy(species_label, temperature, ref_temperature=T_ref):
    """
    Calculates the enthalpy of the selected species.
    Args:
        species_label(str): species (e.g., 'CO2').
        temperature(float): in Kelvin [K].
        ref_temperature(float): default to 298 K.
    Returns:
        h(float): Enthalpy at the defined temperature [kJ/mol].
    """
    species_index = species.index(species_label)
    a = a_matrix[species_index, :]
    T_ref = ref_temperature  # Typically 298K
    T = temperature
    hi_rtref = a[0] + a[1]*T_ref/2 + (a[2]/3)*T_ref**2 + (a[3]/4)*T_ref**3 + (a[4]/5)*T_ref**4 + a[5]/T_ref
    hi_rt = a[0] + a[1]*T/2 + (a[2]/3)*T**2 + (a[3]/4)*T**3 + (a[4]/5)*T**4 + a[5]/T
    h = (hi_rt*R*T - hi_rtref*R*T_ref) + H_ref[species_index]
    return h

def entropy(species_label, temperature, ref_temperature=T_ref):
    """
    Calculates the entropy of the selected species.
    Args:
        species_label(str): species (e.g., 'CO2').
        temperature(float): in Kelvin [K].
        ref_temperature(float): default to 298 K.
    Returns:
        s(float): Entropy at the defined temperature [kJ/mol/K].
    """
    species_index = species.index(species_label)
    a = a_matrix[species_index, :]
    T_ref = ref_temperature
    T = temperature
    si_rtref = a[0]*np.log(T_ref) + a[1]*T_ref + (a[2]/2) * T_ref**2 + (a[3]/3)*T_ref**3 + (a[4]/4)*T_ref**4 + a[6]
    si_rt = a[0]*np.log(T) + a[1]*T + (a[2]/2)*T**2 + (a[3]/3)*T**3 + (a[4]/4)*T**4 + a[6]
    s = (si_rt*R - si_rtref*R) + S_ref[species_index]/1000.0
    return s

def gibbs(species_label, temperature, ref_temperature=T_ref):
    """
    Calculates the Gibbs free energy of the selected species.
    Args:
        species_label(str): species (e.g., 'CO2').
        temperature(float): in Kelvin [K].
        ref_temperature(float): default to 298 K.
    Returns:
        g(float): Gibbs free energy at the defined temperature [kJ/mol].
    """
    species_index = species.index(species_label)
    h = enthalpy(species_label, temperature, ref_temperature)
    s = entropy(species_label, temperature, ref_temperature)
    g = h - temperature*s
    return g

def reaction_string_dict_converter(reaction_string):
    """
    Converts a reaction string to a Python dictionary representation of the reaction.
    Args:
        reaction_string(string): ex. 'CO2 + H2 -> CO + H2O'
    Returns:
        reaction_dict(dict): Dictionary assigning to each species its stoichiometric
                             coefficient in the reaction.
    """
    species = []
    reaction_list = reaction_string.split()
    integers = ['0','1','2','3','4','5','6','7','8','9']
    arrow_index = reaction_list.index('->')
    new_reaction_list = []
    for i in reaction_list:
        if i != '+':
            new_reaction_list.append(i)
    arrow_index = new_reaction_list.index('->')
    for i in new_reaction_list:
        if i == '->':
            pass
        else:
            if i[0] in integers:
                species.append(i[1:])
            else:
                species.append(i)      
    
    stoichiometric_coeff = []
    for i in reaction_list:
        if i in {'->','+'}:
            pass
        elif i[0] not in integers:
            if new_reaction_list.index(i) < arrow_index:
                stoichiometric_coeff.append(-1)
            else:
                stoichiometric_coeff.append(1)
        else:
            if new_reaction_list.index(i) < arrow_index:
                stoichiometric_coeff.append(-1*int(i[0]))
            else:
                stoichiometric_coeff.append(1*int(i[0]))             
    reaction_dict = dict(zip(species,stoichiometric_coeff))
    return reaction_dict

def reaction_enthalpy(reaction_string, temperature, ref_temperature=T_ref):
    """
    Calculates the enthalpy of the input reaction.
    Args:
        reaction_string(str): string of the reaction (e.g., 'CO2 + H2 -> CO + H2O')
        temperature(float): temperature in [K]
        ref_temperature(float): Default to 298 [K].
    Returns:
        h_reaction(float): Reaction enthalpy [kJ/mol].
    """
    h_dict = reaction_string_dict_converter(reaction_string)
    enthalpy_vector = np.zeros(len(h_dict))
    for i in range(len(enthalpy_vector)):
        enthalpy_vector[i] = enthalpy(
            list(h_dict.keys())[i], temperature, ref_temperature)
    h_reaction = np.sum(list(h_dict.values())*enthalpy_vector)
    return h_reaction

def reaction_entropy(reaction_string, temperature, ref_temperature=T_ref):
    """
    Calculates the entropy of the input reaction.
    Args:
        reaction_string(str): string of the reaction (e.g., 'CO2 + H2 -> CO + H2O')
        temperature(float): temperature in [K]
        ref_temperature(float): Default to 298 [K].
    Returns:
        s_reaction(float): Reaction entropy [kJ/mol/K].
    """
    s_dict = reaction_string_dict_converter(reaction_string)
    entropy_vector = np.zeros(len(s_dict))
    for i in range(len(entropy_vector)):
        entropy_vector[i] = entropy(
            list(s_dict.keys())[i], temperature, ref_temperature)
    s_reaction = np.sum(list(s_dict.values())*entropy_vector)
    return s_reaction

def reaction_gibbs(reaction_string, temperature, ref_temperature=T_ref):
    """
    Calculates the Gibbs free energy of the input reaction.
    Args:
        reaction_string(str): string of the reaction (e.g., 'CO2 + H2 -> CO + H2O')
        temperature(float): temperature in [K]
        ref_temperature(float): Default to 298 [K].
    Returns:
        g_reaction(float): Reaction Gibbs free energy [kJ/mol].
    """
    h_reaction = reaction_enthalpy(
        reaction_string, temperature, ref_temperature)
    s_reaction = reaction_entropy(
        reaction_string, temperature, ref_temperature)
    g_reaction = h_reaction - temperature*s_reaction
    return g_reaction

def k_eq_H(reaction_string, temperature, ref_temperature=T_ref):
    """
    Calculates the enthalpy equilibrium constant of the input reaction.
    Args:
        reaction_string(str): string of the reaction (e.g., 'CO2 + H2 -> CO + H2O')
        temperature(float): temperature in [K]
        ref_temperature(float): Default to 298 [K].
    Returns:
        k_eq(float): Enthalpy equilibrium constant [-].
    """
    h_reaction = reaction_enthalpy(reaction_string, temperature, ref_temperature)
    k_eq = np.exp(-h_reaction / (R * temperature))
    return k_eq

def k_eq_S(reaction_string, temperature, ref_temperature=T_ref):
    """
    Calculate the entropy equilibrium constant of the input reaction.
    Args:
        reaction_string(str): string of the reaction (e.g., 'CO2 + H2 -> CO + H2O')
        temperature(float): temperature in [K]
        ref_temperature(float): Default to 298 [K].
    Returns:
        k_eq(float): Entropy equilibrium constant [-].
    """
    s_reaction = reaction_entropy(reaction_string, temperature, ref_temperature)
    k_eq = np.exp(s_reaction / R)
    return k_eq

def reaction_equilibium_constant(reaction_string, temperature, ref_temperature=T_ref):
    """
    Calculate the thermodynamic equilibrium constant of the input reaction.
    Args:
        reaction_string(str): string of the reaction (e.g., 'CO2 + H2 -> CO + H2O')
        temperature(float): temperature in [K]
        ref_temperature(float): Default to 298 [K].
    Returns:
        k_eq(float): Equilibrium constant [-].
    """
    g_reaction = reaction_gibbs(reaction_string, temperature, ref_temperature)
    k_eq = np.exp(-g_reaction / (R * temperature))
    return k_eq

def equilibrium_composition(*reaction_string,
                            temperature, 
                            pressure, 
                            initial_composition, 
                            ref_temperature=T_ref, 
                            ref_pressure=P_ref):
    """ 
    Calculate the equilibrium composition at thermodynamic equilibrium based on ideal mixture of perfect gases.
    Gas: P_i/P_ref
    Liquid: x_i * gamma_i
    Solid: activity = 1
    """

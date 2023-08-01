"""Functions for parsing g.mkm input file (system energetics"""

import numpy as np

def preprocess_g(input_file):
    """
    Preprocess the plain text rm.mkm input file, removing 
    comments, blank lines and trailing white spaces.
    Args:
        input_file(path): input file to be processed.
        ws(int): number of blank lines between global and elementary reactions. 
                 Default to 3.
    Returns:
        new_lines(list): list of the important strings
    """
    x = open(input_file, "r")
    lines = x.readlines()
    new_lines = []
    total_length = len(lines)
    for string in lines:
        if "#" in string:
            index = string.find("#")
            new_lines.append(string[:index].strip("\n"))  # Remove comments 
        else:
            new_lines.append(string.strip("\n"))
        new_lines[-1] = new_lines[-1].rstrip()   # Remove trailing white spaces
    for i in range(len(new_lines)):
        if new_lines[i] == "":
            continue
        else:
            index_first = i   # index of first line after blank lines
            break
    new_lines = new_lines[index_first:]  # Remove empty lines at the beginning
    counter = 0
    index_last = len(new_lines)
    for i in range(index_last):
        if new_lines[i] == '':
            counter += 1
        if counter > 6:  # 3 blank lines between global and elementary reactions
            index_last = i  # index of first blank line at the end
            break
    new_lines = new_lines[:index_last]  # Remove trailing blank lines
    return new_lines

def ts_energy(lines: list, NR: int, tref: float):
    """Get Energy from g.mkm

    Args:
        lines (list): post-processed lines from g.mkm
        NR (int): number of elementary reactions
        tref (float): reference temperature at which entropies are calculated
    """
    E_ts = lines[:NR]
    H_ts = np.zeros(NR)
    S_ts = np.zeros(NR) 
    keys_R = [E_ts[i].split()[0] for i in range(len(E_ts))]
    for i in range(NR):
        index = keys_R.index('R{}:'.format(i+1))
        H_ts[i] = float(E_ts[index].split()[1])
        S_ts[i] = float(E_ts[index].split()[2]) / tref 
    G_ts = H_ts - tref * S_ts
    return H_ts, S_ts, G_ts

def chg_coeffient(lines: list, NR: int):
    """
    Get the charge transfer coefficients for the elementary
    reactions in the mechanism.

    Args:
        lines (list): preprocessed lines from g.mkm
        NR (int): number of elementary reactions
    """
    E_ts = lines[:NR]
    alpha = np.zeros(NR) 
    keys_R = [E_ts[i].split()[0] for i in range(len(E_ts))]
    for i in range(NR):
        index = keys_R.index('R{}:'.format(i+1))
        alpha[i] = float(E_ts[index].split()[3]) 
    return alpha
    

def species_energy(lines: list, NR: int, tref:float, species_tot: list, inerts: list):
    """Get species energy from g.mkm

    Args:
        lines (list): post-processed lines from g.mkm
        NR (int): number of elementary reactions
        tref (float): temperature at which entropy has been calculated.
    """
    NC_tot = len(species_tot)
    E_species = [line for line in lines[NR+3:] if line != ""]
    H_species = np.zeros(NC_tot)
    S_species = np.zeros(NC_tot)
    keys_species = [E_species[j].split()[0].strip(':') for j in range(len(E_species))]
    for i in range(NC_tot):
        if species_tot[i].strip('(g)') not in inerts:
            index = keys_species.index(species_tot[i])
            H_species[i] = float(E_species[index].split()[1])
            S_species[i] = float(E_species[index].split()[-1]) / tref
        else:
            H_species[i] = 0.0
            S_species[i] = 0.0
    G_species = H_species - tref * S_species
    return H_species, S_species, G_species

def reaction_energy(v_matrix: np.ndarray, h_species: np.ndarray, s_species: np.ndarray, tref: float):
    """
    Calculate the energy of the elementary reactions present
    in the reaction mechanism.

    Args:
        v_matrix (np.ndarray): stoichiometric matrix
        h_species (np.ndarray): entahlpy of the species [eV]
        s_species (np.ndarray): entropy of the species [eV]
        tref (float): reference temperature at which entropy is calculated

    Returns:
        tuple with reaction enthalpy, entropy and gibbs free energy
    """
    NR = v_matrix.shape[1]
    dh_reaction = np.zeros(NR)
    ds_reaction = np.zeros(NR)
    dg_reaction = np.zeros(NR)
    for i in range(NR):
        dh_reaction[i] = np.sum(v_matrix[:, i]*np.array(h_species))
        ds_reaction[i] = np.sum(v_matrix[:, i]*np.array(s_species))
        dg_reaction[i] = dh_reaction[i] - tref * ds_reaction[i]
    return dh_reaction, ds_reaction, dg_reaction

def h_barrier(v_matrix: np.ndarray, h_ts: np.ndarray, h_species: np.ndarray, dh_reaction: np.ndarray):
    """
    Calculate the energetic barriers for the elementary reactions 
    in the mechanism.

    Args:
        v_matrix (np.ndarray): stoichiometric matrix
        h_ts (np.ndarray): enthalpy of the transition state [eV]
        h_species (np.ndarray): enthalpy of the species [eV]
        dh_reaction (np.ndarray): reaction enthalpies [eV]

    Returns:
        Tuple with forward and backward enthalpy barriers [eV]
    """
    NR = len(h_ts)
    h_barrier = np.zeros(NR)
    h_barrier_rev = np.zeros(NR)
    for i in range(NR):
        condition1 = h_ts[i] != 0.0  
        ind = list(np.where(v_matrix[:, i] == -1)[0]) + list(np.where(v_matrix[:, i] == -2)[0])
        his = sum([h_species[j]*v_matrix[j, i]*(-1) for j in ind])
        condition2 = h_ts[i] > max(his, his+dh_reaction[i])
        if condition1 and condition2:  # Activated elementary reaction
            h_barrier[i] = h_ts[i] - his
            h_barrier_rev[i] = h_barrier[i] - dh_reaction[i]
        else:  # Unactivated elementary reaction
            if dh_reaction[i] < 0.0:
                h_barrier[i] = 0.0
                h_barrier_rev[i] = -dh_reaction[i]
            else:
                h_barrier[i] = dh_reaction[i]
                h_barrier_rev[i] = 0.0
    return h_barrier, h_barrier_rev

def s_barrier(v_matrix: np.ndarray, s_ts: np.ndarray, s_species: np.ndarray, ds_reaction: np.ndarray):
    """Calculate entropic barriers of the elementary reactions

    Args:
        v_matrix (np.ndarray): stoichiometric matrix 
        s_ts (np.ndarray): transition state entropy [eV]
        s_species (np.ndarray): species entropy [eV]
        ds_reaction (np.ndarray): reaction entropy [eV]

    Returns:
        _type_: _description_
    """
    NR = len(s_ts)
    s_barrier = np.zeros(NR)
    s_barrier_rev = np.zeros(NR)
    for i in range(NR):
        condition1 = s_ts[i] != 0.0  
        ind = list(np.where(v_matrix[:, i] == -1)[0]) + list(np.where(v_matrix[:, i] == -2)[0])
        sis = sum([s_species[j]*v_matrix[j, i]*(-1) for j in ind])
        condition2 = s_ts[i] > max(sis, sis+ds_reaction[i])
        if condition1 and condition2:  # Activated elementary reaction
            s_barrier[i] = s_ts[i] - sis
            s_barrier_rev[i] = s_barrier[i] - ds_reaction[i]
        else:  # Unactivated elementary reaction
            if ds_reaction[i] < 0.0:
                s_barrier[i] = 0.0
                s_barrier_rev[i] = -ds_reaction[i]
            else:
                s_barrier[i] = ds_reaction[i]
                s_barrier_rev[i] = 0.0
    return s_barrier, s_barrier_rev

def g_barrier(v_matrix: np.ndarray, g_ts: np.ndarray, g_species: np.ndarray, dg_reaction: np.ndarray):
    """_summary_
    """
    NR = len(g_ts)
    g_barrier = np.zeros(NR)
    g_barrier_rev = np.zeros(NR)
    for i in range(NR):
        condition1 = g_ts[i] != 0.0  
        ind = list(np.where(v_matrix[:, i] == -1)[0]) + list(np.where(v_matrix[:, i] == -2)[0])
        gis = sum([g_species[j]*v_matrix[j, i]*(-1) for j in ind])
        condition2 = g_ts[i] > max(gis, gis+dg_reaction[i])
        if condition1 and condition2:  # Activated elementary reaction
            g_barrier[i] = g_ts[i] - gis
            g_barrier_rev[i] = g_barrier[i] - dg_reaction[i]
        else:  # Unactivated elementary reaction
            if dg_reaction[i] < 0.0:
                g_barrier[i] = 0.0
                g_barrier_rev[i] = -dg_reaction[i]
            else:
                g_barrier[i] = dg_reaction[i]
                g_barrier_rev[i] = 0.0
    return g_barrier, g_barrier_rev
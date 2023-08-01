"""Functions for parsing the rm.mkm (reaction mechanism)."""

import numpy as np
from natsort import natsorted
from constants import int_set, m_dict, N_AV

def preprocess_rm(input_file):
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
        if counter > 3:  # 3 blank lines between global and elementary reactions
            index_last = i  # index of first blank line at the end
            break
    new_lines = new_lines[:index_last]  # Remove trailing blank lines
    return new_lines

def get_NGR_NR(input_list):
    """
    Get number of global and elementary reactions from rm.mkm input file
    Args:
        input_list(list): list of preprocessed lines in rm.mkm
    Returns:
        NGR(int): number of global reactions
        NR(int): number of elementary reactions
    """
    NGR = 0
    for i in range(len(input_list)):
        if input_list[i] != '':
            NGR += 1
        else:
            break
    NR = len(input_list) - NGR - 3
    return NGR, NR

def get_NC(species_sur: list, species_gas: list, species_tot: list):
    """
    Get number of surface intermediates, gas species and total species in the system 
    under study.
    Args:
        species_sur(list): list of surface species labels (strings)
        species_gas(list): list of gas species labels (strings)
        species_tot(list): list of all the species labels (including CHE)
    Returns:
         (int, int, int)
    """
    return len(species_sur), len(species_gas), len(species_tot)

def reaction_type(lines: list, NR: int, NGR: int):
    """Get type of elementary reaction (adsorption, charge transfer, etc.)

    Args:
        lines (list): post-processed lines from rm.mkm
        NR (int): number of elementary reactions
        NGR (int): number of global reactions

    Returns:
        reaction_type
    """
    reaction_type = []
    for reaction in range(NR):
        line_list = lines[reaction + 3 + NGR].split()
        arrow_index = line_list.index('->')        
        try:  # Extraction of reaction type
            gas_index = line_list.index([element for idx, element in enumerate(line_list) if '(g)' in element][0])
        except IndexError:
            reaction_type.append('sur')  # Surface reaction
        else:
            if gas_index < arrow_index:
                reaction_type.append('ads')  # Adsorption
            else:
                reaction_type.append('des')  # Desorption
        if "H(e)".format() in line_list:
            reaction_type[reaction] += "+e"  # Charge-transfer
    #print(reaction_type_list)
    return reaction_type

def get_species_label(lines, NGR, inerts):
    """
    Args:
        lines(list): lines from rm.mkm
        NGR(int): number of global reactions
        inerts(list): inerts of the system.
    Returns:
        species_label(list):
    """
    species_label = []
    for line in lines[NGR + 3:]:
        for element in line.split():
            if (element == '+') or (element == '->'):
                pass
            elif (element[0] in {'2', '3', '4'}):
                element = element[1:]
                if element in species_label:
                    pass
                else:
                    species_label.append(element)
            elif element in species_label:
                pass
            else:
                species_label.append(element)
    if inerts != None:
        for species in inerts:
                species_label.append(species +'(g)')
    return species_label
        
def stoich_matrix(lines, NR, NGR, species_label):
    """
    Generate stoichiometric matrix of the network.
    Args:
        lines(list): list containing the strings with the elementary reactions
        NR(int): number of elementary reactions
        NGR(int): number of global reactions
        species_label(list): list of species string
    Returns:
        v_matrix(ndarray): stoichiometric matrix representing the network in the rm.mkm file
    """
    NC_tot = len(species_label)
    v_matrix = np.zeros((NC_tot, NR)) 
    
    for reaction in range(NR):
            line = lines[NGR + 3 + reaction].split()
            arrow_index = line.index('->')
            for species in range(NC_tot):
                if species_label[species] in line:
                    species_index = line.index(species_label[species])
                    if species_index < arrow_index:
                        v_matrix[species, reaction] = -1
                    else:
                        v_matrix[species, reaction] = 1
                elif '2' + species_label[species] in line:
                    species_index = line.index('2' + species_label[species])
                    if species_index < arrow_index:
                        v_matrix[species, reaction] = -2
                    else:
                        v_matrix[species, reaction] = 2
    return v_matrix.astype(int)

def stoic_forward(matrix):
    """
    Filter function for the stoichiometric matrix.
    Negative elements are considered and changed of sign in order to 
    compute the direct reaction rates.
    Args:
        matrix(ndarray): Stoichiometric matrix
    Returns:
        mat(ndarray): Filtered matrix for constructing forward reaction rates.
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
    Args: 
        matrix(ndarray): stoichiometric matrix
    Returns:
        mat(ndarray): Filtered matrix for constructing reverse reaction rates.
    """
    mat = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if matrix[i][j] > 0:
                mat[i][j] = matrix[i][j]
    return mat

def classify_species(species_label):
    """
    Classify species based on label string.
    """
    species_sur_label = []
    species_gas_label = []
    for element in species_label:  # Classification of species (sur/gas)
        if '(g)' in element:
            species_gas_label.append(element)
        elif element == "H(e)":
            CHE_switch = 1
        else:
            species_sur_label.append(element)
    species_sur_label = natsorted(species_sur_label)
    if CHE_switch == 1:
        species_tot = species_sur_label + ["H(e)"] + species_gas_label
    else:
        species_tot = species_sur_label + species_gas_label
    return species_sur_label, species_gas_label, species_tot

def global_v_matrix(NC_tot, NGR, gr_strings, species_tot, NC_sur, species_gas):
    """
    
    """
    v_global = np.zeros((NC_tot, NGR))
    for i in range(NC_tot):
        for j in range(NGR):
            reaction_list = gr_strings[j].split()
            arrow_index = reaction_list.index('->')
            if species_tot[i].replace('(g)', "") in reaction_list:
                if reaction_list.index(species_tot[i].replace('(g)', "")) < arrow_index:
                    v_global[i, j] = -1
                else:
                    v_global[i, j] = 1
            else:
                if '2'+species_tot[i].replace('(g)', "") in reaction_list:
                    if reaction_list.index('2'+species_tot[i].replace('(g)', "")) < arrow_index:
                        v_global[i, j] = -2
                    else:
                        v_global[i, j] = 2
                elif '3'+species_tot[i].replace('(g)', "") in reaction_list:
                    if reaction_list.index('3'+species_tot[i].replace('(g)', "")) < arrow_index:
                        v_global[i, j] = -3
                    else:
                        v_global[i, j] = 3
    # for i in range(NC_tot):
    #     for j in range(NGR):
    #         if (i < NC_sur) and (species_tot[i]+'(g)' in species_gas):
    #             v_global[i, j] = 0
    return v_global

def stoich_numbers(NR, NGR, v_matrix, v_global):
    """
    Find stoichiometric factors for each elementary reaction in the network.
    Args:
        NR(int): number of reactions
        NGR(int): number of global reactions
        v_matrix(ndarray): stoichiometric matrix
    Returns:
        stoich_numbers(ndarray): NR * NGR matrix with stoichiometric coefficients.
             [i, j] is the stoichiometric coefficient of elementary reaction i for 
             global reaction j.
    """
    stoich_numbers = np.zeros((NR, NGR))
    for i in range(NGR):
        sol = np.linalg.lstsq(v_matrix, v_global[:, i], rcond=None)
        stoich_numbers[:, i] = np.round(sol[0], decimals=2)
    return stoich_numbers

def gas_MW(species_gas, ):
    """

    Args:

    Returns:

    """
    masses = []
    for i in species_gas:
        mw = 0.0
        MWW = []
        i = i.strip('(g)')
        for j in range(len(i)):
            if j != (len(i)-1):
                if i[j+1].islower():  # next char is lower case (example: Br, Ar)
                    x = i[j:j+2][0] + i[j:j+2][1]
                    MWW.append(x)
                else:
                    if i[j] in int_set:  # CH3, NH2
                        for k in range(int(i[j]) - 1):
                            MWW.append(MWW[-1])
                    elif i[j].islower():
                        pass
                    else:
                        MWW.append(i[j])
            else:  # last string char
                if i[j] in int_set:  # CH3
                    for k in range(int(i[j]) - 1):
                        MWW.append(MWW[-1])
                elif i[j].islower():  # CH3Br
                    pass
                else:  # H3N
                    MWW.append(i[j])
        for i in MWW:
            mw += m_dict[i]
        masses.append(mw)
    return dict(zip(species_gas, masses))

def ads_mass(v_matrix, reaction_type, NC_sur, masses):
    """
    Return the mass of the adsorbates for each elementary reaction.
    Args:
        v_matrix(ndarray): stoichiometric matrix
        reaction_type(list): reaction type list
        NC_sur(int): number of surface intermediates
        masses(list): list of mass of gaseous species
    Returns:
        m(list): when not zero, item i represents the MW of the adsorbate in reaction R[i+1]
    """
    NR = v_matrix.shape[1]
    NC_gas = v_matrix.shape[0] - NC_sur - 1
    m = [0] * NR   
    for i in range(NR):
        if reaction_type[i] == 'sur':
            pass
        else:
            for j in range(NC_gas):
                if v_matrix[NC_sur+j, i] == 0:
                    pass
                else:
                    m[i] = masses[j] / (N_AV*1000)
    return m

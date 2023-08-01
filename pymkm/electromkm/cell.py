"""Classes useful for electrocatalysis """

class Electrode():
    def __init__(self,
                 area,
                 roughness_factor,
                 area_active_site=1e-19,
                 composition=None,
                 surface_def=None):
        """
        Args:
            area(float): geometric area of the electrode
            roughness_factor(float): roughness factor of the electrode
            composition(dict): composition of the electrode
            surface_def(dict): composition of the surface in terms of cristalline planes
        """
        self.area = area  # geometric area [m2]
        self.RF = roughness_factor  # to account for real available area
        self.area_active_site = area_active_site 
        self.composition = composition # e.g.: {'Cu':0.80, 'Sn':0.20}
        self.hkl = surface_def # e.g.: {'100': 0.40, '111': 0.50, '110': 0.10}
        self.ECSA = self.area * self.RF # Electrochemical active Surface Area [m2]
        self.N_as = self.ECSA / self.area_active_site # total number of active sites [-]

class Electrolyte():
    def __init__(self, composition, pH):
        self.composition = composition
        self.pH = pH

class Reaction():
    def __init__(self, reaction_string):
        self.v = reaction_parser['v']
        self.kind = reaction_parser['type']

def reaction_parser(reaction_string):
    x = reaction_string
    species = []
    reaction_list = reaction_string.split()
    integers = ['0', '1','2','3','4','5','6','7','8','9']
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

print(reaction_parser("CO2 + 2e- + 2H+ -> CO + H2O"))

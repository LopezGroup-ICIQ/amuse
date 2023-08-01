"""Set of functions for including linear scaling relationships in microkinetic models.
NEED TO BE MODIFIED SINCE PREVIOUSLY WERE WITHIN THE CLASS"""

def apply_bep(reaction,
              q,
              m):
    """
    Function that applies a BEP relation to the selected elementary reaction.
        dh_barrier = q + m * dh_reaction [eV]
    Args:
        reaction(str): reaction string. Ex: "R12"
        q(float): intercept of the line
        m(float): slope of the line
    """
    i = int(reaction[1:]) - 1
    dh_barrier[i] = q + m * dh_reaction[i]

def apply_ts_scaling(reaction,
                     q,
                     m,
                     initial_state=True):
    """
    Function that applies a TS scaling relation to the selected elementary reaction.
    Apply only if in g.mkm the intermediate states species enthalpies are from LSR
        dh_barrier = q + m * dh_ads(initial/final state)        
    Args:
        reaction(str): reaction string. Ex: "R12"
        q(float): intercept of the line
        m(float): slope of the line
        initial_state(bool): if scaling is wrt initial(True) or final(state)
    """
    i = int(reaction[1:]) - 1
    if initial_state:
        ind = list(np.where(
            self.v_matrix[:, i] == -1)[0]) + list(np.where(self.v_matrix[:, i] == -2)[0])
        his = sum([self.h_species[j]*self.v_matrix[j, i]*(-1)
                  for j in ind])
        self.dh_barrier[i] = q + m * his
    else:
        ind = list(np.where(self.v_matrix[:, i] == 1)[
                   0]) + list(np.where(self.v_matrix[:, i] == 2)[0])
        hfs = sum([self.h_species[j]*self.v_matrix[j, i]*(-1)
                  for j in ind])
        self.dh_barrier[i] = q + m * hfs

def apply_lsr_1d(self,
                 descriptor_name,
                 descriptor_value,
                 scaling_matrix_h,
                 scaling_matrix_ts,
                 bep,
                 initial_state=True):
    """
    Function that builds the whole enthalpy reaction profile
    based on linear scaling relations (LSR).
    Args:
        descriptor_name(str): name of the descriptor (ads. energy of specific species)
        descriptor_value(float): value of the descriptor in eV.
        scaling_matrix_h(ndarray): array with dimension (NC_sur-1)*2.
        scaling_matrix_ts(ndarray): array with dimension NR*2.
    """
    q = scaling_matrix_h[:, 0]
    m = scaling_matrix_h[:, 1]
    for i in range(1, self.NC_sur-1):
        self.h_species[i] = q[i] + m[i] * descriptor_value
    self.h_species[0] = 0.0  # surface
    self.h_species[self.NC_sur:] = 0.0  # gas species
    for i in range(self.NR):
        self.dh_reaction[i] = np.sum(
            self.v_matrix[:, i]*np.array(self.h_species))
    if bep:
        q = scaling_matrix_ts[:, 0]
        m = scaling_matrix_ts[:, 1]
        for j in range(self.NR):
            self.apply_bep("R{}".format(j+1), q[j], m[j])
        self.bep = True
    else:
        q = scaling_matrix_ts[:, 0]
        m = scaling_matrix_ts[:, 1]
        for j in range(self.NR):
            self.apply_ts_scaling("R{}".format(
                j+1), q[j], m[j], initial_state=initial_state)
        self.bep = False
    self.lsr_mode = True  
    self.scaling_matrix_h = scaling_matrix_h
    self.scaling_matrix_ts = scaling_matrix_ts
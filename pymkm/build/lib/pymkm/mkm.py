"""
MKM class for heterogeneous catalysis.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from thermo import k_eq_H, k_eq_S, reaction_enthalpy, reaction_entropy
from constants import *
from functions import *
from reactor import *
from rm_parser import *
from g_parser import *
import graphviz

reactor_dict = {"differential" : DifferentialPFR(), "dynamic": DynamicCSTR()}

class MKM:
    """
    A class for representing microkinetic models for heterogeneous catalytic systems, providing
    functionalities to obtain information like reaction rates, steady-state surface coverage, apparent
    activation energy and reaction orders. Moreover, it provides tools for identifying the descriptors
    of the global process, like the reversibility and degree of rate control analysis.
    Attributes: 
        name(string): Name of the system under study.
        rm_input(string): .mkm file containing the reaction mechanism.
        g_input(string): .mkm file containing the energetics of the system.
        t_ref(float): Reference temperature at which entropic contributions have been calculated [K].
        inerts(list): Inert species in the system under study. The list contains the species name as strings.
    """
    def __init__(self,
                 name,
                 rm_input,
                 g_input,
                 t_ref=273.15,
                 inerts=[]):
        self.name = name
        self.t_ref = t_ref
        self.reactor_model = "differential"
        self.reactor = reactor_dict["differential"]
        self.inerts = inerts
        self.ODE_params = {'reltol': 1e-12, "abstol": 1e-64, "tfin": 1e3}  # ODE solver parameters 
        # Reaction mechanism extraction from rm.mkm
        rm_lines = preprocess_rm(rm_input)
        self.NGR, self.NR = get_NGR_NR(rm_lines)  # Number of global (NGR) and elementary (NR) reactions
        self.r = ['R{}'.format(i+1) for i in range(self.NR)]
        self.gr = ['GR{}'.format(i+1) for i in range(self.NGR)]
        global_reaction_label = [rm_lines[reaction].split()[0] for reaction in range(self.NGR)]
        global_reaction_index = [int(rm_lines[reaction].split()[1][:-1]) for reaction in range(self.NGR)]
        self.gr_string = [" ".join(rm_lines[reaction].split()[2:]) for reaction in range(self.NGR)]
        self.target = global_reaction_index[0]
        self.target_label = global_reaction_label[0]                        # ex: MeOH
        self.by_products = global_reaction_index[1:]                        
        self.by_products_label = global_reaction_label[1:]                  # ex: RWGS
        self.grl = dict(zip(global_reaction_label, global_reaction_index))  # ex: "MeOH": 15
        self.gr_dict = dict(zip(global_reaction_label, self.gr_string))     # ex: "MeOH": "CO2 + 3H2 -> CH3OH + H2O"
        self.reaction_type = reaction_type(rm_lines, self.NR, self.NGR)
        self.species_sur, self.species_gas, self.species_tot = classify_species(get_species_label(rm_lines, self.NGR, inerts))
        self.NC_sur, self.NC_gas, self.NC_tot = get_NC(self.species_sur, self.species_gas)      
        self.v_matrix = stoich_matrix(rm_lines, self.NR, self.NGR, self.species_tot)            
        self.v_f = stoic_forward(self.v_matrix)
        self.v_b = stoic_backward(self.v_matrix)        
        self.MW = gas_MW(self.species_gas)                                                      
        self.m = ads_mass(self.v_matrix, self.reaction_type, self.NC_sur, list(self.MW.values()))
        self.v_global = global_v_matrix(self.gr_string, self.species_tot, self.NC_sur)
        self.stoich_numbers = stoich_numbers(self.v_matrix, self.v_global)
        # g.mkm parsing -> Energetics
        self.h_species, self.s_species, self.g_species = species_energy(preprocess_g(g_input), self.NR, self.NC_tot, t_ref, self.species_tot, inerts)
        self.h_ts, self.s_ts, self.g_ts= ts_energy(preprocess_g(g_input), self.NR, t_ref)
        self.dh_reaction, self.ds_reaction, self.dg_reaction = reaction_energy(self.v_matrix, self.h_species, self.s_species, t_ref)
        self.dh_barrier, self.dh_barrier_rev = h_barrier(self.v_matrix, self.h_ts, self.h_species, self.dh_reaction)
        self.ds_barrier, self.ds_barrier_rev = s_barrier(self.v_matrix, self.s_ts, self.s_species, self.ds_reaction)
        self.dg_barrier, self.dg_barrier_rev = g_barrier(self.v_matrix, self.g_ts, self.g_species, self.dg_reaction)
        # DataFrames
        self.df_system = pd.DataFrame(self.v_matrix,
                                      index=self.species_tot,
                                      columns=[self.r, self.reaction_type])
        self.df_system.index.name = 'species'
        self.df_gibbs = pd.DataFrame(np.array([self.dg_reaction, self.dg_barrier, self.dg_barrier_rev]).T,
                                     index=[self.r, self.reaction_type],
                                     columns=['DGR / eV', 'G_act / eV', 'G_act,rev / eV']).round(2)
        self.df_gibbs.index.name = 'reaction'

    def set_reactor(self, reactor="differential"):
        """
        Define the reactor model. 
        Args:
            reactor(string): "differential" or "dynamic".
        """
        if (reactor not in list(reactor_dict.keys())):
            raise "Wrong reactor model definition. Please choose between 'differential' or 'dynamic'."
        self.reactor_model = reactor
        self.reactor = reactor_dict[reactor]
        return "Reactor model: {}".format(reactor)

    def set_CSTR_params(self,
                        radius,
                        length,
                        Q,
                        m_cat,
                        S_BET,
                        A_site=1.0E-19,
                        verbose=0):
        """ 
        Method for defining the parameters of the 
        Dynamic CSTR reactor.
        Args:
            radius(float): Reactor inner radius in [m]
            length(float): reactor length in [m]
            Q(float): inlet volumetric flowrate in [m3/s]
            m_cat(float): catalyst mass in [kg]
            S_BET(float): BET surface in [m2/kg_cat]
            A_site(float): Area of the active site in [m2]. Default to 1.0E-19
        """
        # TO FILL
        if verbose == 0:
            print("Reactor volume: {:0.2e} [m3]".format(self.CSTR_V))
            print(
                "Inlet volumetric flowrate : {:0.2e} [m3/s]".format(self.CSTR_Q))
            print("Residence time: {:0.2e} [s]".format(self.CSTR_tau))
            print("Catalyst mass: {:0.2e} [kg]".format(self.CSTR_mcat))
            print(
                "Catalyst surface: {:0.2e} [m2/kg_cat]".format(self.CSTR_sbet))
            print("Active site surface: {:0.2e} [m2]".format(self.CSTR_asite))
            return None
        else:
            return None

    def set_ODE_params(self, t_final=1e3, reltol=1e-12, abstol=1e-64):
        """
        Set parameters for ODE numerical integration with scipy solve_ivp solver.
        Args:
            t_final(float): final integration time [s]. Default to 1000 s.
            reltol(float): relative tolerance. Default to 1e-12.
            abstol(float): absolute tolerance. Default to 1e-64.
        """
        self.ODE_params["reltol"] = reltol
        self.ODE_params["abstol"] = abstol
        self.ODE_params["tfin"] = t_final
        print("Integration time = {} s".format(t_final))
        print("Relative tolerance = {}".format(reltol))
        print("Absolute tolerance = {}".format(abstol))
        return "Changed ODE parameters."

    def get_ODE_params(self):
        """Print ODE parameters used in scipy solver solve_ivp."""
        print("Integration time = {}s".format(self.ODE_params["tfin"]))
        print("Relative tolerance = {}".format(self.ODE_params["reltol"]))
        print("Absolute tolerance = {}".format(self.ODE_params["abstol"]))
        return None

    def __str__(self):
        print("System: {}".format(self.name))
        print("")
        for i in self.gr_string:
            print(i)
        print("")
        print("Number of global reactions: {}".format(self.NGR))
        print("Number of elementary reactions: {}".format(self.NR))
        print("Number of surface species: {}".format(self.NC_sur))
        print("Number of gas species: {}".format(self.NC_gas))
        return ""

    @staticmethod
    def methods():
        """Prints all current available MKM methods"""
        functions = ['thermodynamic_consistency_analysis',
                     'kinetic_run',
                     'map_reaction_rate',
                     'apparent_activation_energy',
                     'apparent_activation_energy_local',
                     'apparent_reaction_order',
                     'degree_of_rate_control',
                     'drc_t',
                     'drc_full',
                     'reversibility']
        for method in functions:
            print(method)

    def rxn_network(self):
        """Render reaction network with GraphViz."""
        rn = graphviz.Digraph(name=self.name,
                              comment='Reaction mechanism',
                              format='png',
                              engine='dot')
        rn.attr('graph', nodesep='1.0')
        rn.attr('graph', ratio='1.2')
        rn.attr('graph', dpi='300')
        for species in self.species_sur:
            rn.node(species, color='green', style='bold', fill='red')
        #for species in self.species_gas:
        #    rn.node(species, color='black', style='bold', fill='red')
        for j in range(self.NR):
            a = np.where(self.v_matrix[:, j] < 0)[0]
            b = np.where(self.v_matrix[:, j] > 0)[0]
            if (self.reaction_type[j] == 'ads') or (self.reaction_type[j] == 'des'):
                for i in range(len(a)):
                    for k in range(len(b)):
                        rn.edge(self.species_tot[a[i]],
                                self.species_tot[b[k]])
            else:
                for i in range(len(a)):
                    for k in range(len(b)):
                        rn.edge(self.species_tot[a[i]], self.species_tot[b[k]])
        rn.render(view=True)

    def thermodynamic_consistency(self, temperature):
        """
        Analyze the thermodynamic consistency of the microkinetic 
        model based on the provided energetics and reaction mechanism.
        It compares the equilibrium constants of the global reactions extracted from
        a thermochemistry database with the equilibrium constant from the
        DFT-derived microkinetic model.        
        Args:
            temperature(float): temperature [K].
        """
        k_h = np.exp(-self.dh_reaction / (K_B * temperature))
        k_s = np.exp(self.ds_reaction / K_B)
        DHR_model = np.zeros(self.NGR)
        DSR_model = np.zeros(self.NGR)
        DGR_model = np.zeros(self.NGR)
        DHR_database = np.zeros(self.NGR)
        DSR_database = np.zeros(self.NGR)
        DGR_database = np.zeros(self.NGR)
        keq_H_model = np.zeros(self.NGR)
        keq_S_model = np.zeros(self.NGR)
        keq_model = np.zeros(self.NGR)
        keq_H_database = np.zeros(self.NGR)
        keq_S_database = np.zeros(self.NGR)
        keq_database = np.zeros(self.NGR) 
        for i in range(self.NGR):
            DHR_model[i] = np.sum(self.dh_reaction * self.stoich_numbers[:, i]) * cf
            DSR_model[i] = np.sum(self.ds_reaction * self.stoich_numbers[:, i]) * cf
            DGR_model[i] = DHR_model[i] - temperature * DSR_model[i] * cf
            DHR_database[i] = reaction_enthalpy(self.gr_string[i], temperature)
            DSR_database[i] = reaction_entropy(self.gr_string[i], temperature)
            DGR_database[i] = DHR_database[i] - temperature * DSR_database[i]
            keq_H_model[i] = np.prod(k_h ** self.stoich_numbers[:, i])
            keq_H_database[i] = k_eq_H(self.gr_string[i], temperature)
            keq_S_model[i] = np.prod(k_s ** self.stoich_numbers[:, i])
            keq_S_database[i] = k_eq_S(self.gr_string[i], temperature)
            keq_model[i] = keq_H_model[i] * keq_S_model[i]
            keq_database[i] = keq_H_database[i] * keq_S_database[i]
        print(" {}: Thermodynamic consistency analysis".format(self.name))
        print(" Temperature = {} K\n".format(temperature))
        print("----------------------------------------------------------------------------------\n")
        for global_reaction in range(self.NGR):
            print(self.gr_string[global_reaction])
            print("Model:    DHR={:0.2e} kJ/mol    DSR={:0.2e} kJ/mol/K     DGR={:0.2e} kJ/mol".format(
                DHR_model[global_reaction], DSR_model[global_reaction], DGR_model[global_reaction]))
            print("Database: DHR={:0.2e} kJ/mol    DSR={:0.2e} kJ/mol/K     DGR={:0.2e} kJ/mol".format(
                DHR_database[global_reaction], DSR_database[global_reaction], DGR_database[global_reaction]))
            print("")
            print("Model:    keqH={:0.2e}    keqS={:0.2e}    Keq={:0.2e}".format(
                keq_H_model[global_reaction], keq_S_model[global_reaction], keq_model[global_reaction]))
            print("Database: keqH={:0.2e}    keqS={:0.2e}    Keq={:0.2e}".format(
                keq_H_database[global_reaction], keq_S_database[global_reaction], keq_database[global_reaction]))
            print(
                "----------------------------------------------------------------------------------\n")
        return None

    def __ode_solver_solve_ivp(self,
                               y_0,
                               dy, 
                               reltol, 
                               abstol, 
                               t_final,
                               ode_params,
                               end_events=None,
                               jacobian_matrix=None):
        """
        Helper function for numerical integration of ODE models.
        """        
        r = solve_ivp(dy,
                      (0.0, t_final),
                      y_0,
                      method='BDF', 
                      events=end_events,
                      jac=jacobian_matrix,
                      args=tuple(ode_params),
                      atol=abstol,
                      rtol=reltol)
        return r

    def kinetic_run(self,
                   temperature,
                   pressure,
                   gas_composition,
                   initial_conditions=None,
                   verbose=0):
        """
        Simulate a catalytic run at the defined operating conditions.        
        Args:
            temperature(float): Temperature of the experiment [K].
            pressure(float): Total abs. pressure of gaseous species [Pa].
            gas_composition(list): molar fraction of gas species [-].
            initial_conditions(ndarray): initial fractional coverage of the catalyst surface [-].
            verbose(int): 0=print all output; 1=print nothing.        
        Returns:
            output_dict(dict): Report of the simulation.        
        """
        if verbose == 0:
            print('{}: Microkinetic run'.format(self.name))
            print("Reactor model: {}".format(self.reactor.reactor_type))
            print('Temperature = {}K    Pressure = {:.1f}bar'.format(temperature, pressure/1e5))
            sgas = []
            for i in self.species_gas:
                sgas.append(i.strip('(g)'))
            str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                    list(np.array(gas_composition)*100.0))]
            gas_string = 'Gas composition: '
            for i in str_list:
                gas_string += i
            print(gas_string)
        y_0 = np.zeros(self.NC_tot)
        if initial_conditions is None:  # First surface species is the active site
            y_0[0] = 1.0
        else:
            if sum([1 for i in initial_conditions if i < 0]) != 0:
                raise ValueError("Fatal Error: at least one negative initial surface coverage.")
            y_0[:self.NC_sur] = initial_conditions
        y_0[self.NC_sur:] = pressure * np.array(gas_composition)       
        
        if np.sum(y_0[:self.NC_sur]) != 1.0:
            raise ValueError('Fatal Error: the sum of the provided surface coverage is not equal 1.')
        if sum(gas_composition) != 1.0:
            raise ValueError('Fatal Error: the sum of the provided molar fractions is not equal to 1.')
        for molar_fraction in gas_composition:
            if (molar_fraction < 0.0) or (molar_fraction > 1.0):
                raise ValueError('Fatal Error: molar fractions must be between 0 and 1.')
        if temperature < 0.0:
            raise ValueError('Fatal Error: provided temperature is < 0 K.')
        if pressure < 0.0:
            raise ValueError('Fatal Error: provided pressure is < 0 Pa).')

        kd, ki = kinetic_coeff(self.NR,
                               self.dg_reaction, 
                               self.dg_barrier,
                               temperature, 
                               self.reaction_type,
                               self.m)
        ode_params = [kd, ki, self.v_matrix, self.NC_sur]
        if self.reactor_model == "dynamic":
            ode_params.append(temperature)
            ode_params.append(pressure * gas_composition)
        t0 = time.time()
        results = self.__ode_solver_solve_ivp(y_0,
                                              self.reactor.ode,
                                              *list(self.ODE_params.values()),
                                              ode_params,
                                              end_events=self.reactor.termination_event,
                                              jacobian_matrix=self.reactor.jacobian)
        final_y = results.y[:, -1]
        final_ddt = self.reactor.ode(results.t[-1],
                                     final_y,
                                     *ode_params)
        final_r = net_rate(final_y, kd, ki, self.v_matrix)
        P_in = y_0[self.NC_sur:]
        P_out = final_y[self.NC_sur:]
        reactants = []
        products = []
        conv = []
        for species in self.species_gas:
            index = self.species_gas.index(species)
            if P_in[index] == 0.0:
                products.append(species)
            else:
                if species.strip('(g)') in self.inerts:
                    pass
                else:
                    reactants.append(species)
                    conv.append(self.reactor.conversion(P_in[index], P_out[index]))
        RR = []
        for reaction in range(self.NGR):
            try:
                x = self.species_gas.index(
                    self.gr_string[reaction].split()[-3]+'(g)')
            except:
                x = 0
            if self.reactor_model == "dynamic": 
                RR.append(self.reactor.reaction_rate(P_in[x], P_out[x], temperature))  
            else:
                RR.append(final_r[list(self.grl.values())[reaction]])
        s_target = RR[0] / np.sum(RR)
        masi = {self.species_sur[np.argmax(final_y[:self.NC_sur-1])] : max(final_y[:self.NC_sur-1])}
        coverage_dict = dict(zip(self.species_sur, final_y[:self.NC_sur]))
        r_dict = dict(zip(self.r, final_r))
        y_gas_out = P_out / np.sum(P_out)
        ddt_dict = dict(zip(self.species_tot, final_ddt))
        gas_out = dict(zip(self.species_gas, y_gas_out))
        conv_dict = dict(zip(reactants, conv))
        gas_comp_dict = dict(zip(self.species_gas, gas_composition))
        keys = ['T', 'P', 'y_in',
                'y_out', 'theta', 'ddt',
                'r', *['r_{}'.format(i) for i in list(self.grl.keys())],
                'conversion', 'S_{}'.format(self.target_label),
                'MASI', 'solver']
        values = [temperature, pressure/1e5, gas_comp_dict, 
                  gas_out, coverage_dict, ddt_dict,
                  r_dict, *RR,
                  conv_dict, s_target,
                  masi, results]
        output_dict = dict(zip(keys, values))
        if verbose == 0:
            print('')
            print('{} Reaction Rate: {:0.2e} 1/s'.format(self.target_label, final_r[self.target]))
            print('{} Selectivity: {:.2f}%'.format(self.target_label, s_target*100.0))
            print('Most Abundant Surface Intermediate (MASI): {} Coverage: {:.2f}% '.format(
                list(masi.keys())[0], list(masi.values())[0] * 100.0))
            print('CPU time: {:.2f} s'.format(time.time() - t0))
        return output_dict

    def map_reaction_rate(self,
                          temp_vector,
                          p_vector,
                          composition,
                          global_reaction_label,
                          initial_conditions=None):
        """
        Wrapper function of single_run to map the reaction rate 
        over the desired temperature/pressure domain.
        Args:
           temp_vector(nparray or list): temperature range of interest [K]
           p_vector(nparray or list): pressure range of interest [Pa]
           composition(list): composition of the gas phase in molar fraction [-]
           global_reaction_label(str): string of the selected global reaction rate.
                                      Available labels are listed in self.grl
        Returns:
           r_matrix(ndarray)           
        """
        r_matrix = np.zeros((len(temp_vector), len(p_vector)))
        for i in range(len(temp_vector)):
            for j in range(len(p_vector)):
                run = self.kinetic_run(temp_vector[i],
                                      p_vector[j],
                                      composition,
                                      initial_conditions=initial_conditions,
                                      verbose=0)
                r_matrix[i, j] = list(run['r'].values())[
                    self.grl[global_reaction_label]]
        return r_matrix

    def apparent_activation_energy(self,
                                   temp_range,
                                   pressure,
                                   gas_composition,
                                   global_reaction_label,
                                   initial_conditions=None):
        """
        Calculate the apparent activation energy of the defined global reaction.   
        Args:
            temp_range(list): List containing 3 items: 
                T_range[0]: Lower temperarure range bound [K]
                T_range[1]: Upper temperature range bound [K]
                T_range[2]: Delta of temperature between each point [K]
            pressure(float): Total abs. pressure of the experiment [Pa]
            gas_composition(list): Molar fraction of gas species [-]
            global_reaction_label(str): Label of the global reaction
            initial_conditions(nparray): Array containing initial surface coverage [-]  
        Returns:
            Apparent activation energies for the selected reaction [kJ/mol].      
        """
        print('{}: Apparent activation energy for {} reaction'.format(
            self.name, global_reaction_label))
        print('')
        print('Temperature range: {}-{}K    Pressure = {:.1f}bar'.format(temp_range[0],
                                                                         temp_range[1] -
                                                                         temp_range[2],
                                                                         pressure/1e5))
        sgas = []
        for i in self.species_gas:
            sgas.append(i.strip('(g)'))
        str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                list(np.array(gas_composition)*100.0))]
        gas_string = 'Gas composition: '
        for i in str_list:
            gas_string = gas_string + i
        print(gas_string)
        print('')
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('Unexisting global process string!')
        temperature_vector = list(
            range(temp_range[0], temp_range[1], temp_range[2]))
        r_ea = np.zeros((len(temperature_vector), 1))     
        for i in range(len(temperature_vector)):
            t0 = time.time()
            run = self.kinetic_run(temperature_vector[i],
                                  pressure,
                                  gas_composition,
                                  initial_conditions=initial_conditions,
                                  verbose=1)
            if self.reactor_model == "differential":
                r_ea[i] = list(run['r'].values())[self.grl[global_reaction_label]]
            else: # Dynamic CSTR
                r_ea[i] = self.reactor.reaction_rate(pressure*gas_composition[0],
                                                     pressure*run["y_out"]["i"],  # TO SOLVE
                                                     temperature_vector[i])
            print('Temperature = {}K    CPU Time: {:.2f}s'.format(temperature_vector[i],
                                                                  time.time() - t0))
        eapp, r_squared = calc_eapp(np.asarray(temperature_vector), r_ea)
        keys = ['Tmin',
                'Tmax',
                'N',
                'P',
                'y_gas',
                'Eapp_{}'.format(global_reaction_label),
                'R2']
        gas_comp_dict = dict(zip(self.species_gas, gas_composition))
        values = [temperature_vector[0],
                  temperature_vector[-1],
                  len(temperature_vector),
                  pressure/1e5,
                  gas_comp_dict,
                  eapp,
                  r_squared]
        output_dict = dict(zip(keys, values))
        fig_ea = plt.figure(1, dpi=500)
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.scatter(1000/np.array(temperature_vector), np.log(r_ea), lw=2)
        m, b = np.polyfit(1000/np.array(temperature_vector), np.log(r_ea), 1)
        plt.plot(1000/np.array(temperature_vector),
                 m*1000/np.array(temperature_vector) + b,
                 lw=1, ls='--')
        plt.grid()
        plt.xlabel('1000/T / $K^{-1}$')
        plt.ylabel('$ln(r_{{{}}})$ / $s^{{-1}}$'.format(global_reaction_label))
        plt.title('{}: {} Apparent Activation Energy'.format(
            self.name, global_reaction_label))
        plt.text(0.65,
                 0.75,
                 '$E_{{app}}$={:.0f} kJ/mol\n$R^2$={:.2f}'.format(
                     eapp, r_squared),
                 transform=fig_ea.transFigure,
                 bbox=dict(facecolor='white', alpha=1.0))
        plt.savefig('{}_Eapp_{}_{}{}K_{}bar.png'.format(self.name,
                                                        global_reaction_label,
                                                        temp_range[0],
                                                        temp_range[1],
                                                        int(pressure/1e5)))
        plt.show()
        return output_dict

    def apparent_activation_energy_local(self,
                                         temperature,
                                         pressure,
                                         gas_composition,
                                         global_reaction_label,
                                         delta_temperature=0.1,
                                         initial_conditions=None):
        """
        Calculate the apparent activation energy of the selected reaction.
        It solves an ODE stiff system for each temperature studied until the steady state convergence.
        From the steady state output, the global reaction rates are evaluated.        
        Args:
            temp_range(list): List containing 3 items: 
                T_range[0]: Lower temperarure range bound [K]
                T_range[1]: Upper temperature range bound [K]
                T_range[2]: Delta of temperature between each point [K]
            pressure(float): Total abs. pressure of the experiment [Pa]
            gas_composition(list): Molar fraction of gas species [-]
            global_reaction_label(str): Label of the global reaction
            initial_conditions(nparray): Array containing initial surface coverage [-]  
        Returns:
            Apparent activation energies for the selected reaction in kJ/mol.      
        """
        print('{}: Apparent activation energy for {} reaction'.format(self.name,
                                                                      global_reaction_label))
        print('')
        print('Temperature = {}K    Pressure = {:.1f}bar'.format(temperature,
                                                                 pressure/1e5))
        sgas = []
        for i in self.species_gas:
            sgas.append(i.strip('(g)'))
        str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                list(np.array(gas_composition)*100.0))]
        gas_string = 'Gas composition: '
        for i in str_list:
            gas_string = gas_string + i
        print(gas_string)
        print('')
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('Unexisting global process string!')

        temperature_vector = [
            temperature - delta_temperature, temperature + delta_temperature]
        r_ea = np.zeros((len(temperature_vector), 1))
        for i in range(len(temperature_vector)):
            t0 = time.time()
            run = self.kinetic_run(temperature_vector[i],
                                  pressure,
                                  gas_composition,
                                  initial_conditions=initial_conditions,
                                  verbose=1)
            r_ea[i] = list(run['r'].values())[self.grl[global_reaction_label]]
            print('Temperature = {}K    CPU Time: {:.2f}s'.format(temperature_vector[i],
                                                                  time.time() - t0))
        eapp = (R/1000.0) * temperature**2 * \
            (np.log(r_ea[1]) - np.log(r_ea[0])) / (2*delta_temperature)
        keys = []
        values = []
        output_dict = dict(zip(keys, values))
        return eapp[0]

    def apparent_reaction_order(self,
                                temperature,
                                pressure,
                                composition_matrix,
                                species_label,
                                global_reaction_label,
                                initial_conditions=None):
        """
        Calculate the apparent reaction order of the selected species wrt the defined global reaction.
        Args:
            temperature(float): Temperature of the experiment [K]
            pressure(float): Total pressure of the experiment [Pa]
            composition_matrix(nparray): Matrix containing gas composition at each run.
                                         Dimension in Nruns*NC_gas, where Nruns is the number
                                         of experiments with different composition
            species_label(str): Gas species for which the apparent reaction order is computed
            global_reaction_label(str): Selected global reaction
            initial_conditions(nparray): Array containing initial surface coverage [-]                            
        Returns:
            Apparent reaction order of the selected species for the selected global reaction.        
        """
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('unexisting global process string!')
        if species_label+'(g)' not in self.species_gas:
            raise Exception('Undefined gas species!')
        index = self.species_gas.index(species_label+'(g)')
        for i in composition_matrix[:, index]:
            if i == 0.0:
                raise ValueError(
                    'Provide non-zero molar fraction for the gaseous species for which the apparent reaction order is computed.')
        n_runs = composition_matrix.shape[0]
        r_napp = np.zeros((n_runs, 1))    # Reaction rate [1/s]

        print('{}: {} Apparent reaction order for {} reaction'.format(self.name,
                                                                      species_label,
                                                                      global_reaction_label))
        print('')
        print('Temperature = {}K    Pressure = {:.1f}bar'.format(
            temperature, pressure/1e5))
        print('')
        for i in range(n_runs):
            t0 = time.time()
            run = self.kinetic_run(temperature,
                                  pressure,
                                  composition_matrix[i, :],
                                  initial_conditions=initial_conditions,
                                  verbose=1)
            r_napp[i] = list(run['r'].values())[
                self.grl[global_reaction_label]]
            print('y_{} = {:.2f}    CPU Time: {:.2f}s'.format(species_label,
                                                              composition_matrix[i,
                                                                                 index],
                                                              time.time() - t0))

        napp, r_squared = calc_reac_order(
            pressure*composition_matrix[:, index], r_napp)
        keys = ['T',
                'P',
                'N',
                'y_{}'.format(species_label),
                'r_{}'.format(global_reaction_label),
                'napp_{}'.format(species_label),
                'R2']
        values = [temperature,
                  pressure/1e5,
                  n_runs,
                  composition_matrix[:, index],
                  r_napp,
                  napp,
                  r_squared]
        output_dict = dict(zip(keys, values))
        fig_na = plt.figure(2, dpi=500)
        plt.scatter(
            np.log(pressure*composition_matrix[:, index]), np.log(r_napp), lw=2)
        m, b = np.polyfit(
            np.log(pressure*composition_matrix[:, index]), np.log(r_napp), 1)
        plt.plot(np.log(pressure*composition_matrix[:, index]),
                 m*np.log(pressure*composition_matrix[:, index]) + b,
                 lw=1, ls='--')
        plt.grid()
        plt.xlabel('ln($P_{{{}}}$ / Pa)'.format(species_label))
        plt.ylabel('ln($r_{{{}}}$) / $s^{{-1}}$'.format(global_reaction_label))
        plt.title('{}: {} apparent reaction order for {}'.format(self.name,
                                                                 species_label,
                                                                 global_reaction_label))
        plt.text(0.75, 0.75,
                 '$n_{{app}}$={:.2f}\n$R^2={:.2f}$'.format(napp, r_squared),
                 transform=fig_na.transFigure,
                 bbox=dict(facecolor='white', alpha=1.0))
        plt.savefig('{}_napp_{}_{}_{}K_{}bar.png'.format(self.name,
                                                         global_reaction_label,
                                                         species_label,
                                                         temperature,
                                                         int(pressure/1e5)))
        plt.show()
        return output_dict

    def degree_of_rate_control(self,
                               temperature,
                               pressure,
                               gas_composition,
                               global_reaction_label,
                               ts_int_label,
                               initial_conditions=None,
                               dg=1.0E-6,
                               verbose=0):
        """
        Calculates the degree of rate control(DRC) and selectivity control(DSC)
        for the selected transition state or intermediate species.        
        Args:
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            ts_int_label(str): Transition state/surface intermediate for which DRC/DSC are computed
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            verbose(int): 1= Print essential info
                          0= Print additional info                       
        Returns:
            List with DRC and DSC of TS/intermediate for the selected reaction [-] 
        """
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception(
                'Reaction label must be related to a global reaction!')
        switch_ts_int = 0  # 0=TS 1=intermediate species
        if 'R' not in ts_int_label:
            switch_ts_int = 1
        index = 0
        if switch_ts_int == 0:
            index = int(ts_int_label[1:]) - 1
        else:
            index = self.species_sur.index(ts_int_label)

        if verbose == 0:
            if switch_ts_int == 0:
                print('{}: DRC analysis for elementary reaction R{} wrt {} reaction'.format(self.name,
                                                                                            index+1,
                                                                                            global_reaction_label))
            else:
                print('{}: DRC and DSC for intermediate {} wrt {} reaction'.format(self.name,
                                                                                   self.species_sur[index],
                                                                                   global_reaction_label))
            print('Temperature = {}K    Pressure = {:.1f}bar'.format(
                temperature, pressure/1e5))
            sgas = []
            for i in self.species_gas:
                sgas.append(i.strip('(g)'))
            str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                    list(np.array(gas_composition)*100.0))]
            gas_string = 'Gas composition: '
            for i in str_list:
                gas_string += i
            print(gas_string)
            print('')
        r = np.zeros(2)
        s = np.zeros(2)
        if switch_ts_int == 0:    # Transition state
            if self.g_ts[index] != 0.0:  # Originally activated reaction
                for i in range(2):
                    mk_object = MKM('i', 
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    reactor=self.reactor_model,
                                    inerts=self.inerts)
                    if mk_object.reactor_model == 'dynamic':
                        mk_object.set_CSTR_params(volume=self.CSTR_V,
                                              Q=self.CSTR_Q,
                                              m_cat=self.CSTR_mcat,
                                              S_BET=self.CSTR_sbet,
                                              verbose=1)
                    mk_object.dg_barrier[index] += dg*(-1)**(i)
                    mk_object.dg_barrier_rev[index] += dg*(-1)**(i)
                    run = mk_object.kinetic_run(temperature,
                                               pressure,
                                               gas_composition,
                                               initial_conditions=initial_conditions,
                                               verbose=1)
                    if mk_object.reactor_model == 'differential':
                        r[i] = list(run['r'].values())[
                            self.grl[global_reaction_label]]
                        r_tot = list(run['r'].values())
                        r_tot = [r_tot[i] for i in range(
                            self.NR) if i in list(self.grl.values())]
                        s[i] = r[i] / sum(r_tot)
                    else:  # dynamic CSTR
                        r[i] = run['R_' + global_reaction_label]
                drc = (-K_B*temperature) * (np.log(r[0])-np.log(r[1])) / (2*dg)
                dsc = (-K_B*temperature) * (np.log(s[0])-np.log(s[1])) / (2*dg)
            else:  # Originally unactivated reaction
                for i in range(2):
                    mk_object = MKM('i',
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    reactor=self.reactor_model,
                                    inerts=self.inerts)
                    if mk_object.reactor_model == 'dynamic':
                        mk_object.set_CSTR_params(volume=self.CSTR_V,
                                                  Q=self.CSTR_Q,
                                                  m_cat=self.CSTR_mcat,
                                                  S_BET=self.CSTR_sbet,
                                                  verbose=1)
                    if mk_object.dg_reaction[index] < 0.0:
                        mk_object.dg_barrier[index] = dg * i
                        mk_object.dg_barrier_rev[index] += dg * i
                    else:
                        mk_object.dg_barrier[index] = mk_object.dg_reaction[index] + dg * i
                    run = mk_object.kinetic_run(temperature,
                                               pressure,
                                               gas_composition,
                                               initial_conditions=initial_conditions,
                                               verbose=1)
                    if mk_object.reactor_model == 'differential':
                        r[i] = list(run['r'].values())[
                            self.grl[global_reaction_label]]
                        r_tot = list(run['r'].values())
                        r_tot = [r_tot[i] for i in range(
                            self.NR) if i in list(self.grl.values())]
                        s[i] = r[i] / sum(r_tot)
                    else:  # dynamic CSTR
                        r[i] = run['R_'+global_reaction_label]
                drc = (-K_B*temperature) * (np.log(r[1])-np.log(r[0])) / dg
                dsc = (-K_B*temperature) * (np.log(s[1])-np.log(s[0])) / dg
        else:  # Surface intermediate
            for i in range(2):
                mk_object = MKM('i',
                                self.input_rm,
                                self.input_g,
                                t_ref=self.t_ref,
                                reactor=self.reactor_model,
                                inerts=self.inerts)
                if mk_object.reactor_model == 'dynamic':
                    mk_object.set_CSTR_params(volume=self.CSTR_V,
                                          Q=self.CSTR_Q,
                                          m_cat=self.CSTR_mcat,
                                          S_BET=self.CSTR_sbet,
                                          verbose=1)
                mk_object.g_species[index] += dg * (-1) ** (i)
                for j in range(mk_object.NR):
                    mk_object.dg_reaction[j] = np.sum(
                        mk_object.v_matrix[:, j]*np.array(mk_object.g_species))
                    condition1 = mk_object.g_ts[j] != 0.0
                    ind = list(np.where(mk_object.v_matrix[:, j] == -1)[0]) + list(
                        np.where(mk_object.v_matrix[:, j] == -2)[0])
                    gis = sum([mk_object.g_species[k] *
                              mk_object.v_matrix[k, j]*(-1) for k in ind])
                    condition2 = mk_object.g_ts[j] > max(
                        gis, gis+mk_object.dg_reaction[j])

                    if condition1 and condition2:  # Activated elementary reaction
                        mk_object.dg_barrier[j] = mk_object.g_ts[j] - gis
                        mk_object.dg_barrier_rev[j] = mk_object.dg_barrier[j] - \
                            mk_object.dg_reaction[j]
                    else:  # Unactivated elementary reaction
                        if mk_object.dg_reaction[j] < 0.0:
                            mk_object.dg_barrier[j] = 0.0
                            mk_object.dg_barrier_rev[j] = - \
                                mk_object.dg_reaction[j]
                        else:
                            mk_object.dg_barrier[j] = mk_object.dg_reaction[j]
                            mk_object.dg_barrier_rev[j] = 0.0
                run = mk_object.kinetic_run(temperature,
                                           pressure,
                                           gas_composition,
                                           initial_conditions=initial_conditions,
                                           verbose=1)
                if mk_object.reactor_model == 'differential':
                    r[i] = list(run['r'].values())[
                        self.grl[global_reaction_label]]
                    r_tot = list(run['r'].values())
                    r_tot = [r_tot[i] for i in range(
                        self.NR) if i in list(self.grl.values())]
                    s[i] = r[i] / sum(r_tot)
                else:  # dynamic CSTR
                    r[i] = run['R_'+global_reaction_label]
            drc = (-K_B*temperature) * (np.log(r[0])-np.log(r[1])) / (2*dg)
            dsc = (-K_B*temperature) * (np.log(s[0])-np.log(s[1])) / (2*dg)
        print('DRC = {:0.2f}'.format(drc))
        return drc, dsc

    def drc_t(self,
              temp_vector,
              pressure,
              gas_composition,
              global_reaction_label,
              ts_int_label,
              initial_conditions=None,
              dg=1.0E-6,
              verbose=1):
        """
        Calculate the degree of rate control(DRC) and selectivity control(DSC)
        for the selected transition states or intermediate species as function of temperature.        
        Args:
            temp_vector(nparray): Temperature vector [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            ts_int_label(str or list of str): Transition state/surface intermediate for which DRC/DSC are computed
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            verbose(int): 1= Print essential info
                          0= Print additional info                       
        Returns:
            List with DRC and DSC of TS/intermediate for the selected reaction [-] 
        """
        if (type(ts_int_label) == list):  # multiple species/ts
            drc_array = np.zeros((len(temp_vector), len(ts_int_label)))
            dsc_array = np.zeros((len(temp_vector), len(ts_int_label)))
            for i in range(len(temp_vector)):
                for j in range(len(ts_int_label)):
                    drc_array[i, j], dsc_array[i, j] = self.degree_of_rate_control(temp_vector[i],
                                                                                   pressure,
                                                                                   gas_composition,
                                                                                   global_reaction_label,
                                                                                   ts_int_label[j],
                                                                                   initial_conditions=initial_conditions,
                                                                                   verbose=verbose,
                                                                                   dg=dg)
            drsc_temp = np.concatenate((np.array([temp_vector]).T,
                                        drc_array, dsc_array), axis=1)
            col = ['T[K]']
            for i in range(len(ts_int_label)):
                col.append('DRC_{}'.format(ts_int_label[i]))
            for i in range(len(ts_int_label)):
                col.append('DSC_{}'.format(ts_int_label[i]))

            df_drsc_temp = pd.DataFrame(np.round(drsc_temp, decimals=2),
                                        columns=col)
            fig = plt.figure(dpi=500)
            for i in range(len(ts_int_label)):
                plt.plot(temp_vector, drc_array[:, i], label=ts_int_label[i])
            plt.grid()
            plt.legend()
            plt.xlabel('Temperature / K')
            plt.ylabel('DRC')
            plt.ylim([0.0, 1.0])
            title = "-".join(ts_int_label)
            plt.title(title)
            plt.savefig('{}_drc_{}_{}_{}{}K_{}bar.png'.format(self.name,
                                                              global_reaction_label,
                                                              title,
                                                              temp_vector[0],
                                                              temp_vector[-1],
                                                              int(pressure/1e5)))
            plt.show()
            return df_drsc_temp
        else:  # single species/ts
            drc_array = np.zeros(len(temp_vector))
            dsc_array = np.zeros(len(temp_vector))
            for i in range(len(temp_vector)):
                drc_array[i], dsc_array[i] = self.degree_of_rate_control(temp_vector[i],
                                                                         pressure,
                                                                         gas_composition,
                                                                         global_reaction_label,
                                                                         ts_int_label,
                                                                         verbose=1)
            drsc_temp = np.concatenate((np.array([temp_vector]).T,
                                        np.array([drc_array]).T,
                                        np.array([dsc_array]).T),
                                       axis=1)
            df_drsc_temp = pd.DataFrame(np.round(drsc_temp, decimals=2),
                                        columns=['T[K]', 'DRC', 'DSC'])
            fig = plt.figure(dpi=400)
            plt.plot(temp_vector, drc_array)
            plt.grid()
            plt.xlabel('Temperature / K')
            plt.ylabel('DRC')
            plt.ylim([0, 1])
            plt.title('{}'.format(ts_int_label))
            plt.savefig('{}_drc_{}_{}_{}{}K_{}bar.png'.format(self.name,
                                                              global_reaction_label,
                                                              ts_int_label,
                                                              temp_vector[0],
                                                              temp_vector[-1],
                                                              int(pressure/1e5)))
            plt.show()
            return df_drsc_temp

    def drc_full(self,
                 temperature,
                 pressure,
                 gas_composition,
                 global_reaction_label,
                 initial_conditions=None,
                 dg=1.0E-6):
        """
        Wrapper function that calculates the degree of rate control of all
        intermediates and transition states at the desired conditions.        
        Args:
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(nparray): Molar fraction of the gaseous species [-]
            global_reaction_label(str): Global reaction for which all DRC/DSC are computed
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Applied perturbation to the Gibbs energy of the TS/intermediates.
                       Default=1E-6 eV                       
        Returns:
            Two Pandas DataFrames with final all results.        
        """
        print('{}: Full DRC and DSC analysis wrt {} global reaction'.format(self.name,
                                                                            global_reaction_label))
        print('Temperature = {}K    Pressure = {:.1f}bar'.format(
            temperature, pressure/1e5))
        sgas = []
        for i in self.species_gas:
            sgas.append(i.strip('(g)'))
        str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                list(np.array(gas_composition)*100.0))]
        gas_string = 'Gas composition: '
        for i in str_list:
            gas_string = gas_string + i
        print(gas_string)
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('Unexisting global reaction string.')
        if dg > 0.1:
            raise Exception(
                'Too high perturbation (recommended lower than 1e-6 eV)')
        drc_ts = np.zeros(self.NR)
        dsc_ts = np.zeros(self.NR)
        drc_int = np.zeros(self.NC_sur)
        dsc_int = np.zeros(self.NC_sur)
        for reaction in range(self.NR):
            print('')
            print('R{}'.format(reaction+1))
            drc_ts[reaction], dsc_ts[reaction] = self.degree_of_rate_control(temperature,
                                                                             pressure,
                                                                             gas_composition,
                                                                             global_reaction_label,
                                                                             'R{}'.format(reaction+1),
                                                                             verbose=1,
                                                                             initial_conditions=initial_conditions)
        for species in range(self.NC_sur):
            print('')
            print('{}'.format(self.species_sur[species]))
            drc_int[species], dsc_int[species] = self.degree_of_rate_control(temperature,
                                                                             pressure,
                                                                             gas_composition,
                                                                             global_reaction_label,
                                                                             self.species_sur[species],
                                                                             verbose=1,
                                                                             initial_conditions=initial_conditions)
        r = []
        for i in range(self.NR):
            r.append('R{}'.format(i+1))
        drsc_ts = np.concatenate((np.array([drc_ts]).T,
                                  np.array([dsc_ts]).T),
                                 axis=1)
        df_drsc_ts = pd.DataFrame(np.round(drsc_ts, decimals=2),
                                  index=r,
                                  columns=['DRC', 'DSC'])
        df_drsc_ts.to_csv("X_{}_{}_{}_ts.csv".format(global_reaction_label,
                                                     int(temperature),
                                                     int(pressure/1e5)))

        def style_significant(v, props=''):
            return props if abs(v) >= 0.01 else None

        df_drsc_ts = df_drsc_ts.style.applymap(style_significant, props='color:red;').applymap(
            lambda v: 'opacity: 50%;' if abs(v) < 0.01 else None)
        df_drsc_ts.format({'DRC': '{:,.2f}'.format, 'DSC': '{:,.2f}'.format})
        drsc_int = np.concatenate((np.array([drc_int]).T,
                                   np.array([dsc_int]).T),
                                  axis=1)
        df_drsc_int = pd.DataFrame(np.round(drsc_int, decimals=2),
                                   index=self.species_sur,
                                   columns=['DRC', 'DSC'])
        df_drsc_int.to_csv("X_{}_{}_{}_int.csv".format(global_reaction_label,
                                                       int(temperature),
                                                       int(pressure/1e5)))
        df_drsc_int = df_drsc_int.style.applymap(style_significant, props='color:red;').applymap(
            lambda v: 'opacity: 50%;' if abs(v) < 0.01 else None)
        df_drsc_int.format({'DRC': '{:,.2f}'.format, 'DSC': '{:,.2f}'.format})
        return df_drsc_ts, df_drsc_int

    def reversibility(self,
                      temperature,
                      pressure,
                      gas_composition,
                      initial_conditions=None):
        """
        Function that provides the reversibility of all elementary reaction at the desired
        reaction conditions.

        Args:
            temperature(float): Temperature in [K]
            pressure(float): Pressure in [Pa]
            gas_composition(list): Gas species molar fractions [-]
            initial_conditions(nparray): Initial surface coverage [-]        
        Returns:
            List containing reversibility of all elementary reactions [-]        
        """
        print("{}: Reversibility analysis".format(self.name))
        print('Temperature = {}K    Pressure = {:.1f}bar'.format(
            temperature, pressure/1e5))
        sgas = []
        for i in self.species_gas:
            sgas.append(i.strip('(g)'))
        str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                list(np.array(gas_composition)*100.0))]
        gas_string = 'Gas composition: '
        for i in str_list:
            gas_string = gas_string + i
        print(gas_string)
        run = self.kinetic_run(temperature,
                              pressure,
                              gas_composition,
                              initial_conditions=initial_conditions,
                              verbose=1)
        k = self.kinetic_coeff(temperature)
        composition_ss = list(run['theta'].values())
        for i in range(self.NC_gas):
            composition_ss.append(pressure*gas_composition[i])
        reversibility = z_calc(composition_ss, *k, self.v_f, self.v_b)
        r = []
        for i in range(self.NR):
            r.append('R{}'.format(i+1))
        df_reversibility = pd.DataFrame(np.round(np.array([reversibility]).T, decimals=2),
                                        index=r,
                                        columns=['Reversibility [-]'])

        def style_significant(v, props=''):
            return props if ((v >= 0.01) and (v <= 0.99)) else None
        df_reversibility.to_csv("Z_{}_{}_{}.csv".format(
            self.name, int(temperature), int(pressure/1e5)))
        df_reversibility = df_reversibility.style.applymap(
            style_significant, props='color:red;')
        df_reversibility.format({'Reversibility [-]': '{:,.2f}'.format})
        return df_reversibility
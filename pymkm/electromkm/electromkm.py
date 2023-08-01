"""electroMKM class for electrocatalysis."""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import *
from functions import *
from rm_parser import *
from g_parser import *

class electroMKM:
    """
    A class for representing microkinetic models for electrocatalysis. 
    It provides functions for obtaining information as reaction rates (current density),
    steady-state surface coverage.
    Attributes: 
        name(string): Name of the system under study.
        rm_input_file(string): Path to the .mkm file containing the reaction mechanism.
        g_input_file(string): Path to the .mkm file containing the energy values for TS, surface
                              and gas species.   
        reactor_model(string): reactor model used for representing the system under study. 
                               Two available options:
                                   "differential": differential PFR, zero conversion model
                                   "dynamic": dynamic CSTR, integral model (finite conversion)
        t_ref(float): Reference temperature at which entropic contributions have been calculated [K].
        inerts(list): Inert species in the system under study. The list contains the species name as strings.
    """
    def __init__(self,
                 name: int,
                 rm_input,
                 g_input,
                 t_ref: float=298.15,
                 inerts: list=[]):

        self.name = name
        self.input_rm = rm_input
        self.input_g = g_input
        self.t_ref = t_ref
        self.reactor_model = 'differential'
        self.inerts = inerts
        self.ODE_params = {'reltol':1e-12, 'abstol': 1e-64, 'tfin': 1e3}
        # rm.mkm parsing -> Reaction mechanism
        rm_lines = preprocess_rm(rm_input)
        self.NGR, self.NR = get_NGR_NR(rm_lines)
        self.r = ['R{}'.format(i+1) for i in range(self.NR)]
        self.gr = ['GR{}'.format(i+1) for i in range(self.NGR)]
        global_reaction_label = [rm_lines[reaction].split()[0] for reaction in range(self.NGR)]
        global_reaction_index = [int(rm_lines[reaction].split()[1][:-1]) for reaction in range(self.NGR)]
        self.gr_string = [" ".join(rm_lines[reaction].split()[2:]) for reaction in range(self.NGR)]
        self.target = global_reaction_index[0]
        self.target_label = global_reaction_label[0]
        self.by_products = global_reaction_index[1:]
        self.by_products_label = global_reaction_label[1:]
        self.grl = dict(zip(global_reaction_label, global_reaction_index))
        self.gr_dict = dict(zip(global_reaction_label, self.gr_string))
        self.reaction_type = reaction_type(rm_lines, self.NR, self.NGR)
        self.species_sur, self.species_gas, self.species_tot = classify_species(get_species_label(rm_lines, self.NGR, inerts))
        self.NC_sur, self.NC_gas, self.NC_tot = get_NC(self.species_sur, self.species_gas, self.species_tot)
        self.v_matrix = stoich_matrix(rm_lines, self.NR, self.NGR, self.species_tot)
        self.v_f = stoic_forward(self.v_matrix).T
        self.v_b = stoic_backward(self.v_matrix).T
        self.MW = gas_MW(self.species_gas)                                                      
        self.m = ads_mass(self.v_matrix, self.reaction_type, self.NC_sur, list(self.MW.values()))
        self.v_global = global_v_matrix(self.NC_tot, self.NGR, self.gr_string, self.species_tot, self.NC_sur, self.species_gas)
        self.stoich_numbers = stoich_numbers(self.NR, self.NGR, self.v_matrix, self.v_global)
        # g.mkm parsing -> Energetics
        self.h_species, self.s_species, self.g_species = species_energy(preprocess_g(g_input), self.NR, t_ref, self.species_tot, inerts)
        self.alfa = chg_coeffient(preprocess_g(g_input), self.NR)
        self.h_ts, self.s_ts, self.g_ts= ts_energy(preprocess_g(g_input), self.NR, t_ref)
        self.dh_reaction, self.ds_reaction, self.dg_reaction = reaction_energy(self.v_matrix, self.h_species, self.s_species, t_ref)
        self.dh_barrier, self.dh_barrier_rev = h_barrier(self.v_matrix, self.h_ts, self.h_species, self.dh_reaction)
        self.ds_barrier, self.ds_barrier_rev = s_barrier(self.v_matrix, self.s_ts, self.s_species, self.ds_reaction)
        self.dg_barrier, self.dg_barrier_rev = g_barrier(self.v_matrix, self.g_ts, self.g_species, self.dg_reaction)
        # Pandas DataFrames
        self.df_system = pd.DataFrame(self.v_matrix, index=self.species_sur+["H(e)"]+self.species_gas,
                                      columns=[self.r, self.reaction_type])
        self.df_system.index.name = 'species'
        self.df_gibbs = pd.DataFrame(np.array([self.dg_reaction,
                                               self.dg_barrier,
                                               self.dg_barrier_rev]).T,
                                     index=[self.r, self.reaction_type],
                                     columns=['DGR / eV',
                                              'DG barrier / eV',
                                              'DG reverse barrier / eV'])
        self.df_gibbs.index.name = 'reaction'

    def __str__(self):
        info = "System: {}\n".format(self.name)
        for i in self.gr_string:
            info += i+"\n"
        info += "Number of global reactions: {}\n".format(self.NGR)
        info += "Number of elementary reactions: {}\n".format(self.NR)
        info += "Number of surface species: {}\n".format(self.NC_sur)
        info += "Number of gas species: {}\n".format(self.NC_gas)
        return info
        
    def set_ODE_params(self, t_final=1000.0, reltol=1e-12, abstol=1e-64):
        """
        Set parameters for numerical integration of ODE solvers.
        Args:
            t_final(float): total integration time [s]
            reltol(float): relative tolerance 
            abstol(float): absolute tolerance
        """
        self.ODE_params["reltol"] = reltol
        self.ODE_params["abstol"] = abstol
        self.ODE_params["tfin"] = t_final
        print("Final integration time = {}s".format(t_final))
        print("Relative tolerance = {}".format(reltol))
        print("Absolute tolerance = {}".format(abstol))
        return "Changed ODE solver parameters."

    def kinetic_coeff(self, overpotential, temperature, area_active_site=1e-19):
        """
        Returns the kinetic coefficient for the direct and reverse reactions, according to 
        the reaction type (adsorption, desorption or surface reaction) and TST.
        Revisited from pymkm for electrocatalysis.                
        Args: 
            overpotential(float): applied overpotential [V].
            temperature(float): absolute temperature [K].
            A_site_0(float): Area of the catalytic ensemble [m2]. Default: 1e-19[m2].
        Returns:
            (list): list with 2 ndarrays for direct and reverse kinetic coefficients.
        """
        Keq = np.zeros(self.NR)  # Equilibrium constant
        kd = np.zeros(self.NR)   # Direct constant
        kr = np.zeros(self.NR)   # Reverse constant
        for reaction in range(self.NR):
            Keq[reaction] = np.exp(-self.dg_reaction[reaction] / (temperature * K_B))
            if self.reaction_type[reaction] == 'ads':
                kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            elif self.reaction_type[reaction] == 'des':
                kd[reaction] = (K_B * temperature / H) * \
                    np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            elif self.reaction_type[reaction] == 'sur':  
                kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            else: # Charge transfer reaction
                f = F / (R * temperature)  # C/J
                index = self.species_tot.index('H(e)')
                if self.v_matrix[index, reaction] < 0: # Reduction (e- in the lhs of the reaction)
                    kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                    kd[reaction] *= np.exp(- self.alfa[reaction] * f * overpotential)
                    Keq[reaction] *= np.exp(-f * overpotential)
                    kr[reaction] = kd[reaction] / Keq[reaction]
                else: # Oxidation (e- in the rhs of the reaction)
                    kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                    kd[reaction] *= np.exp((1 - self.alfa[reaction]) * f * overpotential)
                    Keq[reaction] *= np.exp(f * overpotential)
                    kr[reaction] = kd[reaction] / Keq[reaction]
        return kd, kr

    def differential_pfr(self, time, y, kd, ki):
        """
        Returns the rhs of the ODE system.
        Reactor model: differential PFR (zero conversion)
        """
        # Surface species
        dy = self.v_matrix @ net_rate(y, kd, ki, self.v_f, self.v_b)
        # Gas species and H+
        dy[self.NC_sur:] = 0.0
        return dy

    def jac_diff(self, time, y, kd, ki):
        """
        Returns the analytical Jacobian matrix of the system for
        the differential reactor model.
        """
        J = np.zeros((len(y), len(y)))
        Jg = np.zeros((len(kd), len(y)))
        Jh = np.zeros((len(kd), len(y)))
        v_f = self.v_f.T
        v_b = self.v_b.T
        for r in range(len(kd)):
            for s in range(len(y)):
                if v_f[s, r] == 1:
                    v_f[s, r] -= 1
                    Jg[r, s] = kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                elif v_f[s, r] == 2:
                    v_f[s, r] -= 1
                    Jg[r, s] = 2 * kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                if v_b[s, r] == 1:
                    v_b[s, r] -= 1
                    Jh[r, s] = ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
                elif v_b[s, r] == 2:
                    v_b[s, r] -= 1
                    Jh[r, s] = 2 * ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
        J = self.v_matrix @ (Jg - Jh)
        J[self.NC_sur:, :] = 0
        return J

    def __ode_solver_solve_ivp(self,
                               y_0,
                               dy,
                               temperature, 
                               overpotential, 
                               reltol,
                               abstol,
                               t_final,
                               end_events=None,
                               jacobian_matrix=None):
        """
        Helper function for solve_ivp ODE solver.
        """
        kd, ki = self.kinetic_coeff(overpotential, temperature)
        args_list = [kd, ki]
        r = solve_ivp(dy,
                      (0.0, t_final),
                      y_0,
                      method='BDF', 
                      events=end_events,
                      jac=jacobian_matrix,
                      args=args_list,
                      atol=abstol,
                      rtol=reltol,
                      max_step=t_final)
        return r

    def kinetic_run(self,
                    overpotential: float,
                    pH: float,
                    initial_sur_coverage: list=None,
                    temperature: float=298.0,
                    pressure: float =1e5,
                    gas_composition: np.ndarray=None,
                    verbose: int=0,
                    jac: bool=False):
        """
        Simulates a steady-state electrocatalytic run at the defined operating conditions.        
        Args:
            overpotential(float): applied overpotential [V vs SHE].
            pH(float): pH of the electrolyte solution [-].
            temperature(float): Temperature of the system [K].
            pressure(float): Absolute pressure of the system [Pa].
            initial_conditions(nparray): Initial surface coverage array[-].
            verbose(int): 0=print all output; 1=print nothing.        
        Returns:
            (dict): Report of the electrocatalytic simulation.        
        """
        if verbose == 0:
            print('{}: Microkinetic run'.format(self.name))
            print('Overpotential = {}V vs SHE    pH = {}'.format(overpotential, pH))
            print('Temperature = {}K    Pressure = {:.1f}bar'.format(temperature, pressure/1e5))
        # ODE initial conditions
        y_0 = np.zeros(self.NC_tot)
        # 1) surface coverage
        if initial_sur_coverage == None:  # First surface species is the active site
            y_0[0] = 1.0 
        else:
            sum = np.sum(initial_sur_coverage)
            condition2 = True in [(initial_sur_coverage[i] < 0.0) for i in range(len(initial_sur_coverage))] 
            if sum != 1.0 or condition2:
                raise ValueError('Wrong initial surface coverage: Sum must equal to 1 and values >= 0)')
            y_0[:self.NC_sur] = initial_sur_coverage
        # 2) H+ activity (defined by pH)
        y_0[self.NC_sur] = 10 ** (-pH)
        # 3) gas composition
        if gas_composition is None:
            y_0[self.NC_sur+1:] = pressure  
        else:
            y_0[self.NC_sur+1:] = pressure * gas_composition
        #-----------------------------------------------------------------------------------------------        
        if temperature < 0.0:
            raise ValueError('Wrong temperature (T > 0 K)')
        if pressure < 0.0:
            raise ValueError('Wrong pressure (P > 0 Pa)')
        if pH < 0.0 or pH > 14:
            raise ValueError('Wrong pH definition (0 < pH < 14)')
        #-----------------------------------------------------------------------------------------------
        results_sr = []                      # solver output
        final_sr = []                        # final Value of derivatives
        yfin_sr = np.zeros((self.NC_tot))    # steady-state output [-]
        r_sr = np.zeros((self.NR))           # reaction rate [1/s]
        s_target_sr = np.zeros(1)            # selectivity
        t0 = time.time()
        keys = ['T',
                'P',
                'theta',
                'ddt',
                'r',
                *['r_{}'.format(i) for i in list(self.grl.keys())],
                *['j_{}'.format(i) for i in list(self.grl.keys())],
                'S_{}'.format(self.target_label),
                'MASI',
                'solver']
        r = ['R{}'.format(i+1) for i in range(self.NR)]
        values = [temperature, pressure / 1e5]
        _ = None
        if jac: 
            _ = self.jac_diff
        results_sr = self.__ode_solver_solve_ivp(y_0,
                                                 self.differential_pfr,
                                                 temperature,
                                                 overpotential,
                                                 *list(self.ODE_params.values()),
                                                 end_events=None,
                                                 jacobian_matrix=_)
        final_sr = self.differential_pfr(results_sr.t[-1],
                                         results_sr.y[:, -1],
                                         *self.kinetic_coeff(overpotential,
                                                             temperature))
        yfin_sr = results_sr.y[:self.NC_sur, -1]
        r_sr = net_rate(results_sr.y[:, -1],
                        *self.kinetic_coeff(overpotential,
                                            temperature),
                        self.v_f, 
                        self.v_b)
        j_sr = -r_sr * F / (N_AV * 1.0E-19)
        bp = list(set(self.by_products))
        s_target_sr = r_sr[self.target] / (r_sr[self.target] + r_sr[bp].sum())
        value_masi = max(yfin_sr[:self.NC_sur])
        key_masi = self.species_sur[np.argmax(yfin_sr[:self.NC_sur])]
        masi_sr = {key_masi: value_masi}
        coverage_dict = dict(zip(self.species_sur, yfin_sr))
        ddt_dict = dict(zip(self.species_tot, final_sr))
        r_dict = dict(zip(r, r_sr))
        values += [coverage_dict,
                   ddt_dict,
                   r_dict,
                   *[r_sr[i] for i in list(self.grl.values())],
                   *[j_sr[i] for i in list(self.grl.values())],
                   s_target_sr,
                   masi_sr,
                   results_sr]
        output_dict = dict(zip(keys, values))
        if verbose == 0:
            print('')
            print('{} Current density: {:0.2e} mA cm-2'.format(self.target_label,
                                                               j_sr[self.target]/10))
            print('{} Selectivity: {:.2f}%'.format(self.target_label,
                                                   s_target_sr*100.0))
            print('Most Abundant Surface Intermediate: {} Coverage: {:.2f}% '.format(
                key_masi, value_masi*100.0))
            print('CPU time: {:.2f} s'.format(time.time() - t0))
        return output_dict

    def tafel_plot(self,
                   reaction_label: str,
                   overpotential_vector: np.ndarray,
                   pH: float,
                   initial_sur_coverage: list=None,
                   temperature: float=298.0,
                   pressure: float=1e5,
                   gas_composition: np.ndarray=None,
                   verbose: int=0,
                   jac: bool=True):
        """
        Returns the Tafel plot for the defined potential range.
        Args:
            reaction_label(str): Label of the reaction of interest.
            overpotential_vector(ndarray): applied overpotential vector [V].
            pH(float): pH of the electrolyte solution [-].
            initial_conditions(ndarray): initial surface coverage and gas composition [-]
            temperature(float): Temperature of the system [K].
            pressure(float): Absolute pressure of the system [Pa].
            verbose(bool): 0=; 1=.
            jac(bool): Inclusion of the analytical Jacobian for ODE numerical solution.
        """
        exp = []
        j_vector = np.zeros(len(overpotential_vector))
        if reaction_label not in self.grl.keys():
            raise ValueError("Unexisting reaction label")
        print("{}: Tafel slope experiment for {}".format(self.name, reaction_label))
        print("Temperature: {} K    Pressure: {} bar    pH: {}".format(temperature, int(pressure/1e5), pH))
        print("")
        time0 = time.time()
        for i in range(len(overpotential_vector)):
            exp.append(self.kinetic_run(overpotential_vector[i],
                                           pH,
                                           initial_sur_coverage==initial_sur_coverage,
                                           temperature=temperature,
                                           pressure=pressure,
                                           gas_composition=gas_composition,
                                           verbose=1,
                                           jac=jac))
            j_vector[i] = exp[i]['j_{}'.format(reaction_label)]
            if overpotential_vector[i] < 0:
                print("Overpotential = {} V    {} Current Density = {:.2e} mA cm-2".format(overpotential_vector[i],
                                                                                           reaction_label,
                                                                                           j_vector[i]/10))
            else:
                print("Overpotential = +{} V    {} Current Density = {:.2e} mA cm-2".format(overpotential_vector[i],
                                                                                           reaction_label,
                                                                                           j_vector[i]/10))
        print("------------------------------------------------------------------")
        tafel_slope = calc_tafel_slope(overpotential_vector, j_vector)[0]
        f = F / R / temperature
        alfa = 1 + (tafel_slope / f) # Global charge transfer coefficient
        print("Tafel slope = {:.2f} mV    alfa = {:.2f}".format(tafel_slope * 1000.0, alfa))
        print("CPU time: {:.2f} s".format(time.time() - time0)) 
        fig, ax = plt.subplots(2, figsize=(7,5), dpi=400)
        ax[0].plot(overpotential_vector, j_vector/10, 'ok', linewidth=4)
        ax[0].set(xlabel="Overpotential / V vs SHE", ylabel="j / mA cm-2", title="j vs U")
        ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[0].grid()
        ax[1].plot(overpotential_vector, np.log10(abs(j_vector)), 'ok')
        ax[1].set(title="{}: Tafel plot".format(self.name), xlabel="Overpotential / V vs SHE", ylabel="log10(|j|)")
        ax[1].grid()
        plt.tight_layout()
        plt.show()
        plt.savefig("{}_tafel.png".format(self.name))            
        return tafel_slope     
    
    def degree_of_rate_control(self,
                               global_reaction_label: str,
                               ts_int_label: str,
                               overpotential: float,
                               pH: float,
                               initial_sur_coverage: np.ndarray=None,
                               temperature: float=298.0,
                               pressure: float=1e5,
                               gas_composition: list=None,
                               dg: float=1.0E-6,
                               jac: bool=True,
                               verbose: int=0):
        """
        Calculates the degree of rate control(DRC) for the selected transition state or intermediate species.
        Args:
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            ts_int_label(str): Transition state/surface intermediate for which DRC/DSC are computed
            overpotential_vector(ndarray): applied overpotential vector [V].
            pH(float): pH of the electrolyte solution [-].
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            jac(bool): Inclusion of the analytical Jacobian for ODE numerical solution.
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
            print(gas_string+"\n")          
            
        r = np.zeros(2)
        s = np.zeros(2)
        if switch_ts_int == 0:    # Transition state
            if self.g_ts[index] != 0.0:  # Originally activated reaction
                for i in range(2):
                    mk_object = electroMKM('i',
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    inerts=self.inerts)
                    mk_object.dg_barrier[index] += dg*(-1)**(i)
                    mk_object.dg_barrier_rev[index] += dg*(-1)**(i)
                    run = mk_object.kinetic_run(overpotential,
                                                pH,
                                                initial_sur_coverage==initial_sur_coverage,
                                                temperature=temperature,
                                                pressure=pressure,
                                                gas_composition=gas_composition,
                                                verbose=1,
                                                jac=jac)
                r[i] = list(run['r'].values())[self.grl[global_reaction_label]]
                r_tot = list(run['r'].values())
                r_tot = [r_tot[i] for i in range(self.NR) if i in list(self.grl.values())]
                s[i] = r[i] / sum(r_tot)
                drc = (-K_B * temperature) * (np.log(r[0]) - np.log(r[1])) / (2 * dg)
                dsc = (-K_B * temperature) * (np.log(s[0]) - np.log(s[1])) / (2 * dg)
            else:  # Originally unactivated reaction
                for i in range(2):
                    mk_object = electroMKM('i',
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    inerts=self.inerts)
                    if mk_object.dg_reaction[index] < 0.0:
                        mk_object.dg_barrier[index] = dg * i
                        mk_object.dg_barrier_rev[index] += dg * i
                    else:
                        mk_object.dg_barrier[index] = mk_object.dg_reaction[index] + dg * i
                    run = mk_object.kinetic_run(overpotential,
                                                pH,
                                                initial_sur_coverage=initial_sur_coverage,
                                                temperature=temperature,
                                                pressure=pressure,
                                                gas_composition=gas_composition,
                                                verbose=1,
                                                jac=jac)
                    r[i] = list(run['r'].values())[self.grl[global_reaction_label]]
                    r_tot = list(run['r'].values())
                    r_tot = [r_tot[i] for i in range(self.NR) if i in list(self.grl.values())]
                    s[i] = r[i] / sum(r_tot)
                drc = (-K_B*temperature) * (np.log(r[1])-np.log(r[0])) / dg
                dsc = (-K_B*temperature) * (np.log(s[1])-np.log(s[0])) / dg
        else:  # Surface intermediate
            for i in range(2):
                mk_object = electroMKM('i',
                                       self.input_rm,
                                       self.input_g,
                                       t_ref=self.t_ref,
                                       inerts=self.inerts)
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
                            mk_object.dg_barrier_rev[j] = -mk_object.dg_reaction[j]
                        else:
                            mk_object.dg_barrier[j] = mk_object.dg_reaction[j]
                            mk_object.dg_barrier_rev[j] = 0.0
                run = mk_object.kinetic_run(overpotential,
                                            pH,
                                            initial_sur_coverage=initial_sur_coverage,
                                            temperature=temperature,
                                            pressure=pressure,
                                            gas_composition=gas_composition,
                                            verbose=1,
                                            jac=jac)
                r[i] = list(run['r'].values())[self.grl[global_reaction_label]]
                r_tot = list(run['r'].values())
                r_tot = [r_tot[i] for i in range(self.NR) if i in list(self.grl.values())]
                s[i] = r[i] / sum(r_tot)
            drc = (-K_B*temperature) * (np.log(r[0])-np.log(r[1])) / (2*dg)
            dsc = (-K_B*temperature) * (np.log(s[0])-np.log(s[1])) / (2*dg)
        print('DRC = {:0.2f}'.format(drc))
        return drc, dsc

    def drc_full(self,
                 global_reaction_label,
                 overpotential,
                 pH,
                 initial_sur_coverage=None,
                 temperature=298.0,
                 pressure=1e5,
                 gas_composition=None,
                 dg=1.0E-6,
                 jac=False):
        """
        Wrapper function that calculates the degree of rate control of all
        intermediates and transition states at the desired conditions.        
        Args:
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            overpotential_vector(ndarray): applied overpotential vector [V].
            pH(float): pH of the electrolyte solution [-].
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            jac(bool): Inclusion of the analytical Jacobian for ODE numerical solution.
            verbose(int): 1= Print essential info
                          0= Print additional info

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
            drc_ts[reaction], dsc_ts[reaction] = self.degree_of_rate_control(global_reaction_label,
                                                                            'R{}'.format(reaction+1),
                                                                            overpotential,
                                                                            pH,
                                                                            initial_sur_coverage=initial_sur_coverage,
                                                                            temperature=temperature,
                                                                            pressure=pressure,
                                                                            gas_composition=gas_composition,
                                                                            dg=dg,
                                                                            jac=jac,
                                                                            verbose=1)
        for species in range(self.NC_sur):
            print('')
            print('{}'.format(self.species_sur[species]))
            drc_ts[reaction], dsc_ts[reaction] = self.degree_of_rate_control(global_reaction_label,
                                                                             self.species_sur[species],
                                                                             overpotential,
                                                                             pH,
                                                                             initial_sur_coverage=initial_sur_coverage,
                                                                             temperature=temperature,
                                                                             pressure=pressure,
                                                                             gas_composition=gas_composition,
                                                                             dg=dg,
                                                                             jac=jac,
                                                                             verbose=1)
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
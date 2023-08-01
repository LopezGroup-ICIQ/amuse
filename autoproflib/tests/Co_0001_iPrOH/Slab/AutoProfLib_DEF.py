import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

import ase
from ase.build import bulk
from math import sqrt
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import reference_states, atomic_numbers, chemical_symbols
from ase.utils import plural
from ase.io import vasp
from ase.io import read, write
from ase import thermochemistry
from ase import geometry
from ase import units

import pymatgen
from pymatgen import core
from pymatgen.core import sites
from pymatgen.core.structure import Molecule
import pymatgen.symmetry.analyzer as psa

class PreProcessor:

    def __init__(self, work_folders, geometries, max_pbc, max_freq, spin):
        self.work_folders = work_folders

        if type(self.work_folders) == list:
            self.work_folders = self.work_folders
        else:
            tmp_wf = []
            tmp_wf.append(self.work_folders)
            self.work_folders = tmp_wf

        self.indexes_names = [ [ii for ii in range(len(jj)) if jj[ii]=="/"] for jj in self.work_folders]
        if (len(self.work_folders) == 1 and not self.indexes_names[0]):
            self.outfnames = [self.work_folders[0]+"/"+self.work_folders[0]+".xyz"]
        else:
             self.outfnames = [jj+jj[ii[-2]+1:ii[-1]]+".xyz"
                               if not not ii else jj+"/"+jj+".xyz"
                               for ii,jj in zip(self.indexes_names, self.work_folders)]

        self.geometries = geometries

        if type(self.geometries) == list:
            self.geometries = self.geometries
        else:
            tmp_g = []
            tmp_g.append(self.geometries)
            self.geometries = tmp_g

        self.data_as_coordinates = []
        self.coordinates_and_cell = []
        self.max_pbc = max_pbc
        self.max_freq = max_freq
        self.spin = spin

        try:
            len(self.spin)
        except:
            tmp_s = [self.spin for _ in self.work_folders]
            self.spin = tmp_s
        else:
            self.spin = self.spin



    def _vibrational_energy_contribution(self, temperature, vib_energies,
                                         prefactor=None):
        """Calculates the change in internal energy due to vibrations from
        0K to the specified temperature for a set of vibrations given in
        eV and a temperature given in Kelvin. Returns the energy change
        in eV."""
        kT = units.kB * temperature
        dU = 0.
        try:
            len(prefactor)
        except:
            for energy in vib_energies:
                dU += energy / (np.exp(energy / kT) - 1.)
        else:
            for energy,prefac in zip(vib_energies, prefactor):
                dU += prefac*(energy / (np.exp(energy / kT) - 1.))
        return dU


    def _vibrational_entropy_contribution(self, temperature, vib_energies,
                                          prefactor=None):
        """Calculates the entropy due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the entropy
        in eV/K."""
        kT = units.kB * temperature
        S_v = 0.
        try:
             len(prefactor)
        except:
            for energy in vib_energies:
                x = energy / kT
                S_v += x / (np.exp(x) - 1.) - np.log(1. - np.exp(-x))
        else:
            for energy,prefac in zip(vib_energies, prefactor):
                x = energy / kT
                S_v += prefac*(x / (np.exp(x) - 1.) - np.log(1. - np.exp(-x)))
        S_v *= units.kB

        return S_v

    def _get_zpe_contribution(self, vib_energies, prefactor=None):
        """Returns the zero-point vibrational energy correction in eV."""
        zpe = 0.
        try:
            len(prefactor)
        except:
            for energy in vib_energies:
                zpe += 0.5 * energy
        else:
            for energy,prefac in zip(vib_energies, prefactor):
                zpe += 0.5 * energy*prefac
        return zpe



    def _Grimme_factor(self, vib_energies):
        """Returns the Grimme's factor $w(\omega) = 1 / (1 + (\omega_{0} / \omega)^{4})$,
        where \omega_{0} = 100 cm^{-1} = 12.4 MeV and \omega all the other vibrational energies"""

        # Option proposed by S.Grimme in DOI: 10.1002/chem.201200497
        o0 = 0.0124
        grimme_factor = lambda omega, omega0: 1 / ( 1 + (omega0/omega)**4)

        grimme_factors = np.asarray([grimme_factor(i,o0) for i in vib_energies])

        return grimme_factors

    @staticmethod
    def _methods(self):
        functions = ['get_frequencies',
                     'get_cells_and_part_coords',
                     'apply_pbc',
                     'Gibbs',
                     'Helmholtz',
                     'thermal_and_pressure_analysis']
        for method in functions:
            print(method)

    def get_frequencies(self, fname):
        with open(fname, "r") as inf:
            x = [line.strip().split() for line in inf if "cm-1" in line]
            x_no_im = [i for i in x if i[1]=='f']
            inf.close()

        frequency_list_raw = [float(i[9]) for i in x_no_im]

        return frequency_list_raw

    def get_cells_and_part_coords(self):
        Coordinates = []
        Cells = []
        Labels = []
        for work_folder, outfname in zip(self.work_folders, self.outfnames):
            CONTCAR = work_folder + '/' + 'CONTCAR'
            with open(CONTCAR, "r") as inf:
                #Store in a list each line (as a string) of the file and sepparate the characters
                #according to the default criteria
                x = [line.strip().split() for line in inf]
                #Ignore the 2 first lines

                inter_atomic_distance = float(x[1][0])
                x = x[2:]
                #Keep the director vectors of the cell as a numpy array (necessary if direct coordinates
                #are used)
                Cells.append(inter_atomic_distance*np.asarray(x[:3], dtype=float))
                #Forget the mat part
                new_x = x[3:]
                #Keep the atom names, the amount of each of them and if the file is written in direct or
                #cartesian coordinates
                labels = new_x[0]
                amounts = np.asarray(new_x[1], dtype=int)
                try:
                    float(new_x[3][0])
                except:
                    flag = new_x[3][0]
                    new_x = new_x[4:]
                else:
                    flag = new_x[2][0]
                    new_x = new_x[3:]
                labs = []
                for i,j in zip(labels, amounts):
                    #For each element in the range (0,element of amount=number of atoms labeled as i) do:
                    for z in range(j):
                        #Store the label i in the Labels list
                        labs.append(i)
                renew_x = np.asarray([i[:3] for i in new_x[:sum(amounts)]], dtype=float)
                #Declaration of Labels list
                Coordinates.append(renew_x)
                Labels.append(labs)
        #print(sites.pbc_diff(Coordinates))
        self.coordinates_and_cell = [Cells, Coordinates]
        return Cells, Coordinates,Labels

    def apply_pbc(self, mols, cell, labels):

        #dist = lambda atom_a, atom_b: np.sqrt( (atom_a[0]-atom_b[0])**2 + (atom_a[1]-atom_b[1])**2 +
        #                                  (atom_a[2]-atom_b[2])**2)

        new_mols = []
        for mol in mols:
            tmp_2 = []
            for atom in mol:
                tmp_3 = []
                for coord in atom:
                    if coord > self.max_pbc:
                        coord = coord - 1.0
                    else:
                        coord = coord
                    tmp_3.append(coord)
                tmp_3 = np.asarray(tmp_3)
                tmp_2.append(tmp_3)
            tmp_2 = np.asarray(tmp_2)
            new_mols.append(tmp_2)


        cartesian = [np.matmul(j.T, i.T).T if not not list(i) else [] for i,j in zip(new_mols, cell) ]

        new_dicts = []
        for i,j,fname in zip(cartesian, labels, self.work_folders):
            f = open(fname+"_pbc.xyz", "w")
            f.write(str(len(i))+"\n")
            f.write("Paco\n")

            tmp = []
            for k,l in zip(i, j):
                Dict = {"Name":"", "x":0, "y":0, "z":0}
                Dict["Name"] = l

                Dict["x"] = k[0]
                Dict["y"] = k[1]
                Dict["z"] = k[2]

                tmp.append(Dict)

                f.write(l+"\t"+str(k[0])+"\t"+str(k[1])+"\t"+str(k[2])+"\n")
            new_dicts.append(tmp)


        return new_dicts

    def coordinates(self):
        data_as_coordinates = []
        for work_folder, outfname in zip(self.work_folders, self.outfnames):
            CONTCAR = work_folder + '/' + 'CONTCAR'
            f=open(outfname,'w')
            structure = ase.io.read(CONTCAR)
            atoms=structure.get_chemical_symbols()
            nb_atoms = structure.get_global_number_of_atoms()
            position=structure.get_positions(wrap=True,pbc=True,
                                             pretty_translation=True)

            J = []
            f.write(str(nb_atoms) + "\n")
            f.write("\n")
            for ii,jj in zip(atoms, position):
                tmp = []
                tmp.append(ii)
                f.write(ii+"\t")
                for pos in jj:
                    tmp.append(pos)
                    f.write(str(pos)+"\t")
                J.append(tmp)
                f.write("\n")
            data_as_coordinates.append(J)
            self.data_as_coordinates = data_as_coordinates
        return data_as_coordinates

    def Gibbs(self, T, P):
        Gibbs, Enthalpy, S, elE = [], [], [], []
        for work_folder, outfname, geometry, s in zip(self.work_folders, self.outfnames, self.geometries, self.spin):
            CONTCAR = work_folder + '/CONTCAR'
            OUTCAR = work_folder+'/OUTCAR'
            try:
                open(work_folder+"/FREQ/OUTCAR", "r")
            except:
                OUTCAR_FREQ = work_folder+"/../Freq/OUTCAR"
                try:
                    open(OUTCAR_FREQ, "r")
                except:
                    OUTCAR_FREQ = work_folder + "/../FREQ/OUTCAR"
                    try:
                        open(OUTCAR_FREQ, "r")
                    except:
                        OUTCAR_FREQ = work_folder+"/Freq/OUTCAR"
                    else:
                        OUTCAR_FREQ = OUTCAR_FREQ
                else:
                    OUTCAR_FREQ = OUTCAR_FREQ
            else:
                OUTCAR_FREQ = work_folder+"/FREQ/OUTCAR"

            Out = ase.io.read(OUTCAR)
            Cont = ase.io.read(CONTCAR)

            potentialenergy=Out.get_potential_energy()
            atoms=ase.io.vasp.read_vasp(CONTCAR)
            natoms=Cont.get_global_number_of_atoms()

            if geometry[0] == "l":
                geometry='linear'
            elif 'm' in geometry:
                geometry='monoatomic'
            elif geometry[0] == "n":
                geometry='nonlinear'
            else:
                print("No available geometry in the data-base. Taking linear instead.")
                geometry = "linear"


            frequency_list_raw = self.get_frequencies(OUTCAR_FREQ)

            #The number of frequencies should be equal to 3*number of atoms - 5
            #for linear atoms and 3*number of atoms - 6 for non-linear (?)

            if geometry=='linear':
                max_num_freqs = 3*natoms - 5
            else:
                max_num_freqs = 3*natoms - 6

            #Once it is deffined the maximum number of frequencies, we
            #calculate the difference between the current number of frequencies
            #(len(frequency_list_raw)) and the maximum number of frequencies
            #(diff = len(frequency_list_raw)-max_num_freqs).

            #If the maximum number of frequencies is different to the length
            #of max_num_freqs
            if max_num_freqs != len(frequency_list_raw):
                diff = len(frequency_list_raw) - max_num_freqs
                if diff > 0:
                    #Then, if the difference is positive, we remove the
                    #minimum value of the frequency_list_raw diff times
                    for _ in range(diff):
                        frequency_list_raw.remove(min(frequency_list_raw))
                #Else, we do nothing
                else:
                    frequency_list_raw = frequency_list_raw
            #Else, we do nothing
            else:
                frequency_list_raw = frequency_list_raw

            #Finally we transform the frequency_list_raw in a numpy array
            #to be able to divide it: pass from MeV to eV
            frequency_list_raw = np.asarray(frequency_list_raw)

            if self.max_freq[1] == None:
                frequency_list_1 = frequency_list_raw
                frequency_list_1 = frequency_list_1 / 1000
            else:
                if self.max_freq[0] == "Erase":
                    frequency_list_1 = np.asarray([i for i in frequency_list_raw
                                                   if i >
                                                   self.max_freq[1]*1000])
                elif self.max_freq[0] == "Grimme":

                    print("IMPLEMENTING THE GRIMME METHOD ON"+outfname)
                    frequency_list_1 = frequency_list_raw

                else:
                    frequency_list_1 = np.asarray([i if i
                                                   > self.max_freq[1]*1000
                                                   else self.max_freq[1]*1000
                                                   for i in frequency_list_raw])

                frequency_list_1 = frequency_list_1 / 1000





            def symnumb(mol):
                f=psa.PointGroupAnalyzer(mol, tolerance=0.3, eigen_tolerance=0.01)
                Pgroup=str(f.get_pointgroup())

                Symmetry=0
                if Pgroup in ['C1','Cs','C*v']:
                    Symmetry=1
                elif Pgroup in ['C2', 'C2v', 'D*h']:
                    Symmetry=2
                elif Pgroup=='C3v':
                    Symmetry=3
                elif Pgroup=='D2h':
                    Symmetry=4
                elif Pgroup in ['D3h', 'D3d']:
                    Symmetry=6
                elif Pgroup=='D5h':
                    Symmetry=10
                elif Pgroup=='Td':
                    Symmetry=12
                elif Pgroup=='Oh':
                    Symmetry=24
                return (Symmetry)

            self.coordinates()
            mol=Molecule.from_file(outfname)
            symmetrynumber=(symnumb(mol))

            frequency_list = frequency_list_1
            if work_folder[-1] != "/":
                work_folder = work_folder + "/"
            else:
                work_folder = work_folder

            freq_name = work_folder + "freq.txt"

            print("Frequency list saved on "+freq_name)
            with open(freq_name, "w") as outf2:
                for freq in frequency_list:
                    outf2.write(str(freq)+"\n")



            if self.max_freq[0] == "Grimme":
              #frequency_list_cm = np.asarray([float(i[7]) for i in x_no_im])

                gf = self._Grimme_factor(frequency_list)

                zpe = self._get_zpe_contribution(frequency_list, prefactor=gf)

                dH_v = self._vibrational_energy_contribution(T, frequency_list,prefactor=gf)

                S_r = 0
                freq_hz = frequency_list *241.79991 * 1E12

                if self.max_freq[2] == False:
                    for freq,gfac in zip(freq_hz, gf):
                        log_in = units._k*T*np.pi / (units._hplanck * freq)
                        S_r += (1-gfac)*(units.kB * (0.5+np.log(log_in**0.5)))
                #One of the options from literature XXX
                else:
                    average = sum([units._k*T*np.pi / (units._hplanck * freq)
                                   for freq in freq_hz]) / len(freq_hz)
                    for freq,gfac in zip(freq_hz, gf):
                        log_in_0 = units._k*T*np.pi / (units._hplanck * freq)
                        log_in = log_in_0 * average / (log_in_0 + average)
                        S_r += (1-gfac)*(units.kB * (0.5+np.log(log_in**0.5)))
                S_v = self._vibrational_entropy_contribution(T,frequency_list, prefactor=gf)

            else:
                zpe = self._get_zpe_contribution(frequency_list)
                dH_v = self._vibrational_energy_contribution(T, frequency_list)
                # Rotational entropy (term inside the log is in SI units).
                if geometry == 'monatomic':
                    S_r = 0.0
                elif geometry == 'nonlinear':
                    inertias = (atoms.get_moments_of_inertia() * units._amu / (10.0**10)**2)  # kg m^2
                    S_r = np.sqrt(np.pi * np.product(inertias)) / symmetrynumber
                    S_r *= (8.0 * np.pi**2 * units._k * T / units._hplanck**2)**(3.0 / 2.0)
                    S_r = units.kB * (np.log(S_r) + 3.0 / 2.0)
                elif geometry == 'linear':
                    inertias = (atoms.get_moments_of_inertia() * units._amu / (10.0**10)**2)  # kg m^2
                    inertia = max(inertias)  # should be two identical and one zero
                    S_r = (8 * np.pi**2 * inertia * units._k * T / symmetrynumber / units._hplanck**2)
                    S_r = units.kB * (np.log(S_r) + 1.)

                # Vibrational entropy.
                S_v = self._vibrational_entropy_contribution(T,frequency_list)


            print("Energy summary at T = %.2f K" % T)
            print("=" *31)
            print("Potential energy: "+str(potentialenergy)+" eV")
            print("ZPE correction: "+str(zpe)+" eV")



            Cv_t = 3. / 2. * units.kB  # translational heat capacity (3-d gas)
            if geometry == 'nonlinear':  # rotational heat capacity
                Cv_r = 3. / 2. * units.kB
            elif geometry == 'linear':
                Cv_r = units.kB
            elif geometry == 'monatomic':
                Cv_r = 0.


            Cv = Cv_t*T + Cv_r*T + dH_v
            print("Total Cv: "+str(Cv)+" eV")

            enthalpy = potentialenergy + zpe + Cv_t*T + Cv_r*T + dH_v + units.kB*T

            print("H: "+str(enthalpy)+" eV")


            # Translational entropy (term inside the log is in SI units).
            mass = sum(atoms.get_masses()) * units._amu  # kg/molecule
            S_t = (2 * np.pi * mass * units._k *T / units._hplanck**2)**(3.0 / 2)
            S_t *= units._k * T / 1.013E5
            S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)
            print("The S_t contribution is: "+str(S_t)+"eV/K")

            print("The S_r contribution is: "+str(S_r)+"eV/K")

            # Electronic entropy.
            S_e = units.kB * np.log(2 *s + 1)
            print("The S_e contribution is: "+str(S_e)+"eV/K")

            print("The S_v contribution is: "+str(S_v)+" eV/K")

            # Pressure correction to translational entropy.
            S_p = - units.kB * np.log(P / 1.013E5)
            print("The S_p contribution is: "+str(S_p)+"eV/K")

            entropy = S_e + S_t + S_v + S_p + S_r

            print("S: "+str(entropy)+ " eV/K")
            gibbs = enthalpy - T*entropy
            print("G: "+str(gibbs)+" eV/K")

            print("=" * 31)

            energy_sum_outfname = work_folder+"energy_summary_"+str(self.max_freq[1])+".txt"
            print("Energy summary saved at "+str(energy_sum_outfname))

            with open(energy_sum_outfname, "w") as outf3:
                outf3.write("E"+"\t"+str(potentialenergy)+"\n")
                outf3.write("EZPE"+"\t"+str(zpe)+"\n")
                outf3.write("Cv"+"\t"+str(Cv)+"\n")
                outf3.write("H"+"\t"+str(enthalpy)+"\n")
                outf3.write("S"+"\t"+str(entropy)+"\n")
                outf3.write("G"+"\t"+str(gibbs)+"\n")


            Gibbs.append(gibbs)
            Enthalpy.append(enthalpy)
            S.append(entropy)
            elE.append(potentialenergy)

        return Gibbs,Enthalpy,S,elE

    def Helmholtz(self, T):

        Helmholtz,Internal_energy,Entropy,Ep,elE = [], [], [], [],[]

        for work_folder, outfname in zip(self.work_folders, self.outfnames):
            try:
                open(work_folder+"/FREQ/OUTCAR", "r")
            except:
                OUTCAR_FREQ = work_folder+"/../Freq/OUTCAR"
                try:
                    open(OUTCAR_FREQ, "r")
                except:
                    OUTCAR_FREQ = work_folder + "/../FREQ/OUTCAR"
                    try:
                        open(OUTCAR_FREQ, "r")
                    except:
                        OUTCAR_FREQ = work_folder+"/Freq/OUTCAR"
                    else:
                        OUTCAR_FREQ = OUTCAR_FREQ
                else:
                    OUTCAR_FREQ = OUTCAR_FREQ
            else:
                OUTCAR_FREQ = work_folder+"/FREQ/OUTCAR"



            frequency_list_raw = self.get_frequencies(OUTCAR_FREQ)

            if self.max_freq[1] == None:
                frequency_list_1 = frequency_list_raw
                frequency_list_1 = np.asarray(frequency_list_1) / 1000
            else:
                if self.max_freq[0] == "Erase":
                    frequency_list_1 = np.asarray([i for i in frequency_list_raw
                                                   if i >
                                                   self.max_freq[1]*1000])
                elif self.max_freq[0] == "Grimme":

                    print("IMPLEMENTING THE GRIMME METHOD ON"+outfname)
                    frequency_list_1 = frequency_list_raw

                else:
                    frequency_list_1 = np.asarray([i if i
                                                   > self.max_freq[1]*1000
                                                   else self.max_freq[1]*1000
                                                   for i in frequency_list_raw])

                frequency_list_1 = frequency_list_1 / 1000

            #self.coordinates()
            O=read(OUTCAR_FREQ)
            potentialenergy=O.get_potential_energy()
            ep=O.get_total_energy()

            frequency_list = frequency_list_1

            if work_folder[-1] != "/":
                work_folder = work_folder + "/"
            else:
                work_folder = work_folder

            freq_name = work_folder + "freq.txt"
            print("Frequency list saved on "+freq_name)
            with open(freq_name, "w") as outf2:
                for freq in frequency_list:
                    outf2.write(str(freq)+"\n")



            if self.max_freq[0] == "Grimme":
                # U = Ep + ZPE + Cv

                gf = self._Grimme_factor(frequency_list)


                zpe = self._get_zpe_contribution(frequency_list, prefactor=gf)
                Cv = self._vibrational_energy_contribution(T,frequency_list,prefactor=gf)
                entropy = self._vibrational_entropy_contribution(T,
                                                                 frequency_list,
                                                                 prefactor=gf)

            else:

                zpe = self._get_zpe_contribution(frequency_list)
                Cv = self._vibrational_energy_contribution(T,frequency_list)
                entropy = self._vibrational_entropy_contribution(T,
                                                                 frequency_list)


            print("Energy summary at T = %.2f K" % T)
            print("=" *31)
            print("Potential energy: "+str(potentialenergy)+" eV")

            print("ZPE correction: "+str(zpe)+" eV")

            print("Total Cv: "+str(Cv)+" eV/K")

            internal_energy = potentialenergy + zpe + Cv
            print("U: "+str(internal_energy)+" eV")

            print("S: "+str(entropy)+" eV/K")

            helmholtz = internal_energy - T * entropy
            print("F: "+str(helmholtz)+" eV")
            print("=" *31)
            energy_sum_outfname = work_folder+"energy_summary_"+str(self.max_freq[1])+".txt"
            print("Energy summary saved at "+str(energy_sum_outfname))

            with open(energy_sum_outfname, "w") as outf3:
                outf3.write("E"+"\t"+str(potentialenergy)+"\n")
                outf3.write("EZPE"+"\t"+str(zpe)+"\n")
                outf3.write("Cv"+"\t"+str(Cv)+"\n")
                outf3.write("U"+"\t"+str(internal_energy)+"\n")
                outf3.write("S"+"\t"+str(entropy)+"\n")
                outf3.write("F"+"\t"+str(helmholtz)+"\n")



            elE.append(potentialenergy)
            Ep.append(ep)
            Helmholtz.append(helmholtz)
            Entropy.append(entropy)
            Internal_energy.append(internal_energy)

        return Helmholtz,Internal_energy,Entropy,Ep,elE


    def thermal_and_presure_analysis(self, T, P, G_A = "Gibbs"):

        Df = []
        T_list, P_list = [], []
        try:
            len(T)
        except:
            T_list.append(T)
        else:
            T_list = T
        try:
            len(P)
        except:
            P_list.append(P)
        else:
            P_list = P
        outfnames = [str(i) for i in T]
        if G_A == "Gibbs":
            if len(P_list) == 1:
                G, H, S = [], [], []
                for t in T_list:
                    g, h, s, _ = self.Gibbs(t, P)
                    G.append(g)
                    H.append(h)
                    S.append(s)

                G = np.asarray(G)
                H = np.asarray(H)
                S = np.asarray(S)
                for i in range(len(self.work_folders)):
                    df = pd.DataFrame({"T / K": T_list,"Gibbs / eV":G[:, i],"H / eV":H[:, i],
                                       "S / eV K^-1":S[:, i] })
                    df.to_csv(self.work_folders[i]+outfnames[i], index=False)
                    Df.append(df)
            else:
                for p in P_list:
                    df_tmp = []
                    G, H, S = [], [], []
                    for t in T_list:
                        g, h, s, _ = self.Gibbs(t, p, geometry)
                        G.append(g)
                        H.append(h)
                        S.append(s)
                    for i in range(len(self.work_folders)):
                        df = pd.DataFrame({"T / K": T_list,"Gibbs / eV":G[:, i],"H / eV":H[:, i],
                                           "S / eV K^-1":S[:, i] })
                        df.to_csv(self.work_folders[i]+str(p)+"_"+outfnames[i], index=False)
                        df_tmp.append(df)
                    Df.append(df_tmp)
        else:
            A, H, S, Ep = [], [], [], []
            for t in T_list:
                a, h, s, ep = self.Helmholtz(t)
                A.append(a)
                H.append(h)
                S.append(s)
                Ep.append(ep)

            A = np.asarray(A)
            H = np.asarray(H)
            S = np.asarray(S)
            Ep = np.asarray(Ep)

            for i in range(len(self.work_folders)):
                df = pd.DataFrame({"T / K": T_list,"Helmhotz / eV":A[:, i],"U / eV":H[:, i],
                                   "S / eV K^-1":S[:, i], "Ep / eV":Ep[:, i] })
                df.to_csv(self.work_folders[i]+outfnames[i], index=False)
                Df.append(df)
        Df = pd.concat(Df)
        return Df

class AutoProfLib:
    """The AutoProfLib library is a Python framework used for generating automatically
    reaction pathways parsing a set of .xyz files recovered from IochemBD.
    The AutoProfLib has been tested in the Indium Oxide-Pd dopped systems presented
    in the work of Frei et al. https://www.nature.com/articles/s41467-019-11349-9"""

    #Periodic Table information from https://inventwithpython.com/bigbookpython/project53.html

    def __init__(self, surface_phase, gas_phase, TSs, add_atom, use_pbc, reference,
                 crop_paths_info=[False, [14, 15, 16], [1,2]],
                 metal_off=[False, []],react_hyd=[True, 6], order_options="Consistent", T_P=[298, 1.013e5], colors=["maroon", "orange", "red", "gold"]):
        """Class initialization. Invokes the get_sorted_dicts() function
        (see get_sorted_dicts() fucntion). The AutoProfLib assumes that the names
        of the files to parse (.xyz in this version) are labeled according to an order
        (Ex: 35000.xyz, 35001.xyz...n.xyz).
        Args:
            id_start (int): numerical id. of the initial file.
            id_end (int): numerical id. of the final file.
            crop_paths_info (list): here are defined the files to ignore, and the option
            for the user to divide the mechanism in several paths (2 by the moment).
            It contains:
                 crop_paths_info[0] (bool): a boolean to invoke the crop_paths() method
                 (see crop_paths()). If True, the method is invoked. While False, the
                 method is unactivated.
                 crop_paths_info[1] (list): list of the ids (int) of the first path to
                 remove/ignore.
                 crop_paths_info[2] (list): list of the ids (int) of the second path to
                 remove/ignore.
            metal_off (list): this list provides information concerning if the user
            wants to ignore certain element, like for instance, a metal.
            It contains:
                 metal_off[0] (bool): If True, the element or elements defined in
                 metal_off[1] are ignored. Otherwhise, all elements are taken into
                 account
                 metal_off[1] (list): Contains the name of the element or elements
                 (string) to ignore
        """
        self.NAMES = ["Element", "Valence", "Max_dist"]
        self.PERIODIC_TABLE = pd.read_csv("./periodictable.csv",
                                          names=self.NAMES)


        self.gas_phase = gas_phase
        self.surface_phase = surface_phase
        self.TSs = TSs
        self.T_P = T_P

        self.order_options = order_options
        self.add_atom = add_atom

        self.use_pbc = use_pbc
        self.reference = reference
        if not not self.gas_phase:
            self.gas_preprocessor = PreProcessor(self.gas_phase[0],
                                                self.gas_phase[1],
                                                self.use_pbc[2],
                                                self.gas_phase[2],
                                                 self.gas_phase[3])
        else:
            self.gas_preprocessor = []
        if not not self.surface_phase:
            self.surface_preprocessor = PreProcessor(self.surface_phase[0],
                                                    self.surface_phase[1],
                                                    self.use_pbc[2],
                                                    self.surface_phase[2],
                                                    self.surface_phase[3])
        else:
            self.surface_preprocessor = []
        if not not self.TSs:
            self.TSs_preprocessor = PreProcessor(self.TSs[0], self.TSs[1],
                                                self.use_pbc[2],
                                                self.TSs[2],
                                                self.TSs[3])
        else:
            self.TSs_preprocessor = []
        #Declaration of the list of ccordinates for all the input files, which
        #is a list of
        self.coords_list = np.asarray([])
        self.coords_list_gas = np.asarray([])
        self.coords_list_TSs = np.asarray([])

        self.metal_off = metal_off
        #Invocation of get_sorted_dicts(), which is a list of dicts
        self.unsorted_dicts, self.unsorted_dicts_gas, self.unsorted_dicts_TSs = [], [], []
        self.coord_dicts, self.coord_dicts_gas, self.coord_dicts_TSs = self.get_sorted_dicts()

        #Declaration of the indexes list which will be used to reorder the
        #coordinates (see get_ordered_dicts())
        self.inds = []
        self.inds_gas = []
        self.inds_TSs = []
        #Declaration of the list which contains the indexes of the systems
        #which are the start structure for each path (see is_start())
        self.indexes_is_start = []
        #Declaration of the empty list for the Reactions resorted to compare
        #with the matrix done by hand
        self.Rxs_resorted = []

        self.conn_dicts = []
        self.gas_conn_dicts = []
        self.TSs_conn_dicts = []
        #Declaration of an empty Networkx Grap() object
        self.gr = nx.Graph()
        #List containing the information according if there is one hydrogenated reactant
        #The first element is a bool (True if hydrogenated reactant, false otherwise)
        #The second element is the number of hydrogen present in the hydrogenated reactant
        self.react_hyd = react_hyd

        self.crop_paths_info = crop_paths_info

        self.Assign_TS = []
        self.Assigned_TSs = []
        self.list_TSs = []
        self.int_labels = []
        self.g_labels = []
        self.human_readable_labels = {}

        self.TS_s_exact_labels = []
        self.graph_paths = []

        self.dict_complete = []

        self.colors = colors


    def import_data(self, fname):
        """The import_data(fname) function is a general parser that stores the input
        present in a file on a given path (fname).
        Args:
            fname (string): the path to a given file.
        returns: raw_data (list). The raw_data list contains the content
                 of a given file stored line by line as a string list."""
        #Declaration of the empty raw_data list
        raw_data = []
        #Open the file which path points the fname argument
        with open(fname) as inp:
            #For row in the input file
            for row in inp:
                #Split the row (namelly a string) into a list using whatever
                #separation
                x = row.strip().split()
                #Store the result in the raw_data list
                raw_data.append(x)
        #Security measure: if the user left a blank line in the end,don't store it
        if not raw_data[-1]:
            raw_data = raw_data[:-1]
        else:
            raw_data = raw_data
        return raw_data

    def get_coord_dict(self, data):
        """The get_coord_dict(data) function assumes that the input file is an
        .xyz file, thus first it transforms the last three elements of each
         line in each member of data in floats (numpy arrays, dtype=float).
         Then, it stores each line in a dictionary which entries are:
        'Name': the element symbol for each line
        'x': the x coordinate for each line (previous transformation to float)
        'y': the y coordinate for each line (previous transformation to float)
        'z': the z coordinate for each line (previous transformation to float)
        'Mass': the element mass for each line
        The final output is a list of dictionaries, containing the name of the
        atoms and their coordinates, and a numpy array of numpy arrays containing
        the coordinates only.
        Args:
            data (list): list of strings containing all the data stored on a given
            .xyz file.
        returns (list): contains the coord_dicts, a list which stores a list with
                        all the dictionaries for each line in data, and the coords
                        numpy array, which contains all the coordinates of the lines
                        stored in data as numpy arrays.
        """
        #Reject the 2 first lines in data: it contains the name of the file
        #and the number of atoms
        Data = data[2:]
        #It is assumed that the first element for each line in data is the
        #name of the element
        names = [i[0] for i in Data]
        #Take all the elements for each line in data and transform it to
        #a float numpy array
        coords = np.asarray([np.asarray(i[1:], dtype=float) for i in Data])
        #Declaration of the coord_dict list
        coord_dict=[]
        #For each line in names and coords do:
        for i, j in zip(names, coords):
            #Dict dictionary declaration
            Dict = { "Name": "name", "x": 0.0, "y": 0.0, "z": 0.0}
            #Take the elements of names as "Name" entries
            Dict["Name"] = i
            #Take the elements of coords as the x,y,z coordinates
            Dict["x"] = j[0]
            Dict["y"] = j[1]
            Dict["z"] = j[2]
            #Store each dict in coord_dict list
            coord_dict.append(Dict)
        return [coord_dict, coords]

    def get_unsorted_coord_dicts(self):
        """The get_unsorted_dicts() invokes the generate_fname_list(), which will
        be used as input for the import_data(fname) function. Then,
        the import_data(fname) function will be the input for the
        get_coord_dict(data) function. Additionally, the indexs of the dictionaries are
        ordered according to z (first) and x (second) directions and stored in the inds list (int).
        returns: unsorted_dicts (list): list containing the dictionaries of each system parsed"""
        #Invokes the generate_fname_list() function
        #fnames = self.generate_fname_list()
        #Applies the import_data(fname) function
        if type(self.surface_preprocessor) != list:
            self.surface_preprocessor.coordinates()
            Raw_Data = [self.import_data(i) for i in self.surface_preprocessor.outfnames]
            self.coords_list = np.asarray([self.get_coord_dict(i)[1] for i in Raw_Data])
            unsorted_dicts = [self.get_coord_dict(i)[0] for i in Raw_Data]
        else:
            Raw_Data = []
            self.coords_list = self.coords_list
            unsorted_dicts = []

        if type(self.gas_preprocessor) != list:
            self.gas_preprocessor.coordinates()
            Raw_Data_gas = [self.import_data(i) for i in self.gas_preprocessor.outfnames]
            self.coords_list_gas = np.asarray([self.get_coord_dict(i)[1] for i in Raw_Data_gas])
            unsorted_dicts_gas = [self.get_coord_dict(i)[0] for i in Raw_Data_gas]
        else:
            Raw_Data_gas = []
            self.coords_list_gas = self.coords_list_gas
            unsorted_dicts_gas = []

        if type(self.TSs_preprocessor) != list:
            self.TSs_preprocessor.coordinates()
            Raw_Data_TSs = [self.import_data(i) for i in self.TSs_preprocessor.outfnames]
            self.coords_list_TSs = np.asarray([self.get_coord_dict(i)[1] for i in Raw_Data_TSs])
            unsorted_dicts_TSs = [self.get_coord_dict(i)[0] for i in Raw_Data_TSs]
        else:
            Raw_Data_TSs = []
            self.coords_list_TSs = self.coords_list_TSs
            unsorted_dicts = []



        if (not self.add_atom or type(self.add_atom)!=list):
            unsorted_dicts, unsorted_dicts_gas, unsorted_dicts_TSs = unsorted_dicts, unsorted_dicts_gas, unsorted_dicts_TSs
            self.coords_list, self.coords_list_gas, self.coords_list_TSs = self.coords_list, self.coords_list_gas, self.coords_list_TSs
        else:
            for ii, kk in zip(self.add_atom[0], self.add_atom[1]):
                Dict = { "Name": "name", "x": 0.0, "y": 0.0, "z": 0.0}
                Dict["Name"] = kk[0]
                Dict["x"] = float(kk[1])
                Dict["y"] = float(kk[2])
                Dict["z"] = float(kk[3])
                tmp = np.asarray([kk[1], kk[2], kk[3]], dtype=float)
                if ii[0] == 0:
                    if not not unsorted_dicts:
                        unsorted_dicts[ii[1]].append(Dict)
                        np.append(self.coords_list[ii[1]], tmp)
                    else:
                        print("Not surface input provided: no atom added.")
                elif ii[0] == 1:
                    if not not unsorted_dicts_gas:
                        unsorted_dicts_gas[ii[1]].append(Dict)
                        np.append(self.coords_list_gas[ii[1]], tmp)
                    else:
                        print("Not gas input provided: no atom added.")
                elif ii[0] == 2:
                    if not not unsorted_dicts_TSs:
                        unsorted_dicts_TSs[ii[1]].append(Dict)
                        np.append(self.coords_list_TSs[ii[1]], tmp)
                    else:
                        print("Not TS input provided: no atom added.")
                else:
                    print("Option not available. Not atom added.")
                    unsorted_dicts, unsorted_dicts_gas, unsorted_dicts_TSs = unsorted_dicts, unsorted_dicts_gas, unsorted_dicts_TSs
                    self.coords_list, self.coords_list_gas, self.coords_list_TSs = self.coords_list, self.coords_list_gas, self.coords_list_TSs
        #Fills the inds list with the sorted indexs according to z (first) and x (second)
        if not not list(self.coords_list):
            self.inds = [np.lexsort((i[:, 1], i[:, 0], i[:, 2])) for i in self.coords_list]
        else:
            self.inds = self.inds
        if not not list(self.coords_list_gas):
            self.inds_gas = [np.lexsort((i[:, 1], i[:, 0], i[:, 2])) for i in self.coords_list_gas]
        else:
            self.inds_gas = []
        if not not list(self.coords_list_TSs):
            self.inds_TSs = [np.lexsort((i[:, 1], i[:, 0], i[:, 2])) for i in self.coords_list_TSs]
        else:
            self.inds_TSs = self.inds_TSs

        return unsorted_dicts, unsorted_dicts_gas, unsorted_dicts_TSs

    def get_sorted_dicts(self):
        """The get_sorted_dicts() function orders the dictionaries according to the inds
        list information. Additionaly, applies the argument metal_off, that does not take into
        account the element provided by the user.
        returns coord_dicts (list): list of all the coordinates dictionaries ordered according
        to the inds list."""
        #Invoke the get_unsorted_coord_dicts() function
        self.unsorted_dicts, self.unsorted_dicts_gas, self.unsorted_dicts_TSs = self.get_unsorted_coord_dicts()
        #Declare the coord_dicts list
        def sort_dicts(ud, inds):
            res = []
            for items, indexes in zip(ud, inds):
                tmp = []
                for index in indexes:
                    tmp.append(items[index])
                res.append(tmp)
            return res

        def erase_metal(cd):
            dict_res = [ [ j for j in i if j["Name"] not in self.metal_off[1]]
                          for i in cd]
            new_coord_list = []
            for i in dict_res:
                #Declare a temporary list
                tmp = []
                #For each element in each element of coord_list do:
                for j in i:
                    #Store the coordinates of each element inside each element of coord_dicts
                    tmp.append([j["x"], j["y"], j["z"]])
                #Store the entire temporary list
                new_coord_list.append(np.asarray(tmp))
            return dict_res, np.asarray(new_coord_list)

        if (not self.unsorted_dicts or not self.inds):
            coord_dicts = []
        else:
            coord_dicts = sort_dicts(self.unsorted_dicts, self.inds)
        if (not self.unsorted_dicts_gas or not self.inds_gas):
            coord_dicts_gas = []
        else:
            coord_dicts_gas = sort_dicts(self.unsorted_dicts_gas, self.inds_gas)
        if (not self.unsorted_dicts_TSs or not self.inds_TSs):
            coord_dicts_TSs = []
        else:
            coord_dicts_TSs = sort_dicts(self.unsorted_dicts_TSs, self.inds_TSs)


        if (self.metal_off[0] == True and not not coord_dicts and not not coord_dicts_TSs):
            coord_dicts, self.coords_list = erase_metal(coord_dicts)
            coord_dicts_TSs, self.coords_list_TSs = erase_metal(coord_dicts_TSs)
        elif (self.metal_off[0] == True and not coord_dicts and not not coord_dicts_TSs):
            coord_dicts_TSs, self.coords_list_TSs = erase_metal(coord_dicts_TSs)
        elif (self.metal_off[0] == True and not not  coord_dicts and not coord_dicts_TSs):
            coord_dicts, self.coords_list = erase_metal(coord_dicts)
        #If the user switched off the metal_off option do nothing
        else:
            coord_dicts = coord_dicts
            self.coords_list = self.coords_list
            coord_dicts_TSs = coord_dicts_TSs
            self.coords_list_TSs = self.coords_list_TSs
        return coord_dicts, coord_dicts_gas, coord_dicts_TSs

    def is_start(self, cl):
        """The is_start() function detects which systems are the pristine surface or if a vacancy is
        generated. Then, the program assumes that the subsequent states also contains a vacancy.
        Additionally, stores the index from which the elements are the pristine surface (or pristine
        surface - vacancy). This information will be used to detect the molecules on a given surface.
        return bool_start (list): the bool_start list contains two elements for each one of the parsed
        systems: a bool, which is True if the element is the pristine surface or contains a lower
        amount of atoms (vacancy), and a int, which contains the number of atoms present in each
        system."""
        #Calculate the number of the first system, which is assumed to be the pristine surface
        length_start = len(cl[0])
        #Calculates the number of atoms for each system
        lengths = [len(i) for i in cl[1:]]
        #Declare the bool_start list
        bool_start = []
        #For each element on the number of atoms list do:
        for i in lengths:
            #If the number of atoms is lower or equal to the start system
            if i-length_start <= 0:
                #The first element of bool_start for this system is True
                bool_start.append([True, i])
            else:
                #Else the first element is False. The second element is always the
                #number of atoms of each system
                bool_start.append([False, i])
        return bool_start

    def is_molecule(self, cl):
        """The is_molecule() function detects the molecules for all the systems. To this end,
        the is_start() function is invoked, and the indexes_is_start list is filled with the
        number of atoms for the different pristine (or pristine - vacancy) systems.
        returns list: the first element of the list contains the index up to which the system
        atoms are molecules, while the second element is a flag that indicates the absence of more
        than one pristine surface. If the indexes_is_start list is empty, the flag is None."""

        #Invoke the is_start() function
        starts = self.is_start(cl)
        #Get the number of atoms of the first system
        length_start = len(cl[0])
        #Declare the index_is_mol list
        index_is_mol = []
        #Store the number of atoms of the pristine surface
        index_is_mol.append(length_start)
        #Fill the indexes_is_start list with the number of atoms for all the pristine
        #surfaces (or pristine - vacancy)
        indexes_is_start = [i for i in range(len(starts)) if starts[i][0]==True]
        #If the indexes_is_start list is empty do:
        if not indexes_is_start:
            #Store the number of atoms of the pristine surface for all the systems
            index_is_mol = [length_start for _ in cl]
            #Index_is_start is None
            index_is_start = None
        else:
            #Index_is_start is the number of atoms of the prisitine surface
            index_is_start = indexes_is_start[0]
            #For each position in the starts list do:
            for i in range(len(starts)):
                #If the the number of atoms of the ith system is larger than
                #the pristine surface - 1 do:
                if starts[i][1] >= index_is_start - 1:
                    #Store the pristine surface number of atoms
                    index_is_mol.append(length_start)
                else:
                    #Otherwhise, store the number of atoms of the next pristine surface
                    index_is_mol.append(starts[index_is_start][1])

        return index_is_mol, index_is_start, indexes_is_start

    def get_molecules(self):
        """The get_molecules() invokes the is_molecule() function and selects the molecules
        for each system.
        returns molecules (list): the molecules list contains the atoms considered molecules
        for each system."""
        #Invokes the is_molecule() function
        if self.order_options != "Consistent":
            #TO DO: add periodic boundary conditions here
            system = self.coord_dicts.copy()
            for i in self.coord_dicts_TSs:
                system.append(i)
            indexes, _, indexes_is_start = self.is_molecule(system)
            if not indexes_is_start:
                self.indexes_is_start = indexes_is_start
            else:
                self.indexes_is_start = indexes[:len(system)-len(self.TSs[0])]

            indexes_TSs = indexes[-len(self.TSs[0]):]
            indexes = indexes[:-len(self.TSs[0])]
            #Selects the atoms according to the indexes detected with the is_molecule() function
            surface = self.coord_dicts
            gas_molecules = self.coord_dicts_gas
            molecules = [i[j:] for i, j in zip(surface, indexes)]
            molecules_TSs = [i[j:] for i, j in zip(self.coord_dicts_TSs, indexes_TSs)]
        else:
            if not not self.unsorted_dicts:
                start = len(self.unsorted_dicts[0])
                molecules_cartesian = [i[start:] for i in self.unsorted_dicts]
            else:
                start = 0
                molecules_cartesian = []
            if not not self.unsorted_dicts_TSs:
                molecules_TSs_cartesian = [i[start:] for i in self.unsorted_dicts_TSs]
            else:
                molecules_TSs_cartesian = []
            if not not self.unsorted_dicts_gas:
                gas_molecules =  self.unsorted_dicts_gas
            else:
                gas_molecules = []

            if self.use_pbc[0] == True:
                if type( self.surface_preprocessor) != list:
                    surf_cells, surf_coords, Labels_surf = self.surface_preprocessor.get_cells_and_part_coords()
                    mols_part_coord = [i[start:] for i in surf_coords]
                    Labels_surf_mols = [i[start:] for i in Labels_surf]
                    molecules = self.surface_preprocessor.apply_pbc(mols_part_coord, surf_cells,
                                                                   Labels_surf_mols)

                else:
                    surf_cells, surf_coords, Labels_surf = [],[],[]
                    mols_part_coord = []
                    Labels_surf_mols = []
                    molecules = []



                if type(self.TSs_preprocessor) != list:
                    TSs_cells, TSs_coords, Labels_TSs = self.TSs_preprocessor.get_cells_and_part_coords()
                    mols_part_coord_TSs = [i[start:] for i in TSs_coords]
                    Labels_TSs_mols = [i[start:] for i in Labels_TSs]
                    molecules_TSs = self.TSs_preprocessor.apply_pbc(mols_part_coord_TSs, TSs_cells,
                                                                   Labels_TSs_mols)

                else:
                    TSs_cells, TSs_coords, Labels_TSs = [],[],[]
                    mols_part_coord_TSs = []
                    Labels_TSs_mols = []
                    molecules_TSs = []

               #molecules_TSs = [i[start:] for i in self.unsorted_dicts_TSs]
                if (not self.add_atom or type(self.add_atom)!=list):
                    molecules, molecules_TSs = molecules, molecules_TSs
                else:
                    for ii, kk in zip(self.add_atom[0], self.add_atom[1]):
                        Dict = { "Name": "name", "x": 0.0, "y": 0.0, "z": 0.0}
                        Dict["Name"] = kk[0]
                        Dict["x"] = float(kk[1])
                        Dict["y"] = float(kk[2])
                        Dict["z"] = float(kk[3])
                        tmp = np.asarray([kk[1], kk[2], kk[3]], dtype=float)
                        if ii[0] == 0:
                            if not not molecules:
                                molecules[ii[1]].append(Dict)
                            else:
                                molecules = molecules
                                print("No surface input provided: no atom added.")
                        elif ii[0] == 1:
                            if not not gas_molecules:
                                gas_molecules[ii[1]].append(Dict)
                            else:
                                gas_molecules = gas_molecules
                                print("No gas input provided: no atom added.")
                        #gas = gas
                        elif ii[0] == 2:
                            if not not gas_molecules:
                                molecules_TSs[ii[1]].append(Dict)
                            else:
                                molecules_TSs = molecules_TSs
                                print("No TS input provided: no atom added.")
                        #molecules_TSs = molecules_TSs
                        else:
                            print("Option not available. Not atom added")
                            molecules, molecules_TSs = molecules, molecules_TSs
                    if not self.use_pbc[1]:
                        molecules, molecules_TSs = molecules, molecules_TSs
                    else:
                        for i in self.use_pbc[1]:
                            if i[0] == 0:
                                if not not molecules:
                                    molecules[i[1]] = molecules_cartesian[i[1]]
                                else:
                                    molecules = molecules
                            elif i[0] == 2:
                                if not not molecules_TSs:
                                    molecules_TSs[i[1]] = molecules_TSs_cartesian[i[1]]
                                else:
                                    molecules_TSs = molecules_TSs
                            else:
                                molecules, molecules_TSs = molecules, molecules_TSs

            else:
                molecules, molecules_TSs = molecules_cartesian, molecules_TSs_cartesian
        return molecules, gas_molecules, molecules_TSs

    def is_in_the_sphere(self, points, center):
        """The is_in_the_sphere(points, center) functions deffines the neighbourhood relations
        between atoms according to the distance of those atoms respect to an imaginary sphere.
        Thus, all the atoms inside the sphere, centered in a given atom, are connected to the
        center atom. The sphere radius depends on the center type (atom), which allows different
        number of connections in order to satysfy the chemical rules.
        Args:
            points (list): the points argument is a list of dictionaries (as the object returned
                           by the get_molecules() function), which are the possible neighbours
                           of the center argument.
            center (dict): the center argument is a dictionary (as an element of the
                           get_molecules() returned list), which will define the center of the
                           imaginary sphere.
        returns res (list): the res list has the same number of elements than the points
                            list, and it contains 0 or 1 values depending if the ith element
                            of the points list is attached to the center."""
        #Declaration of the res list
        res = []
        #Declaration and initialization of the sphere radius r
        r = 0.0
        #Declaration and initialization of the max_connectivity value, which depends on the
        #center atom type. Defines the maximum connectivity value for each atom.
        max_connectivity = 0.0
        #Declaration and initialization of the max_radius value, which depends on the
        #center atom type. Defines the maximum sphere radius value for each atom (max. bond length).
        max_radius = 0.0
        #Atom type of the center
        Name = center["Name"]
        #Set the properties of the spehere (maximum connectivity, maximum radius) according to
        #the atom type.

        keys = self.PERIODIC_TABLE.loc[self.PERIODIC_TABLE["Element"]==Name]


        max_connectivity = int(keys["Valence"])
        max_radius = float(keys["Max_dist"])



        #While the radius is smaller than a randomly big value (to ensure that we are taking all
        #possible steps)
        while r<100:
            #Declaration of a temporary list
            tmp = []
            #For each element in the points list do:
            for i in points:
                #Calculate the distance (check) between the center and the point
                check = (i["x"]-center["x"])**2 + (i["y"]-center["y"])**2 + (i["z"]-center["z"])**2
                #If the distance between the point and the center is smaller than the radius of the
                #sphere (that is: the point is inside the sphere) do:
                if (check < r) and (i["x"]!=center["x"]) and (i["y"]!=center["y"]) and (i["z"]!=center["z"]):
                    #The point and the center are attached, thus the value is 1.
                    tmp.append(1)
                else:
                    #Otherwhise the point and the center are not connected, thus append 0
                    tmp.append(0)
            #If all connections of center (sum of tmp) are smaller or equal to the maximum
            #connectivity and the radius of the sphere is smaller than the maximum bond length
            #do:
            if (sum(tmp) < max_connectivity) and (r < max_radius):
                #Increase the radius by 0.5 units
                r = r + 0.5
            else:
                #Otherwhise res = tmp and stop the while loop
                res=tmp
                break
        return res


    def make_adj_mtr(self):
        """The make_adj_mtr() function makes the adjacency matrix of all the systems parsed.
        First, it invokes the get_molecules() function, and then applies the
        is_in_the_sphere(points, center) function to all the elements of the get_molecules() list.
        Finally, it also stores the labels (names) of each one of the elements present in the
        get_molecules() list.
        returns (list): the first element is the adjacency matrix of all the molecules present in
        the system, defined as a list of numpy arrays. The second element is a string list containing
        all the names of the atoms on a given molecule."""

        #Invokes the get_molecules() function
        mols_surf, gas, TSs = self.get_molecules()
        #Generate the adjacency matrix by recursively apply the is_in_the_sphere(points, center)
        #function.
        if not not mols_surf:
            adj_mat = [np.asarray([self.is_in_the_sphere(j, i) for i in j]) for j in mols_surf]
            labels = [[i["Name"] for i in j] for j in mols_surf]
        else:
            adj_mat = []
            laebls = []
        if not not gas:
            adj_mat_gas = [np.asarray([self.is_in_the_sphere(j, i) for i in j]) for j in gas]
            labels_gas = [[i["Name"] for i in j] for j in gas]

        else:
            adj_mat_gas = []
            labels_gas = []

        if not not TSs:
            adj_mat_TSs = [np.asarray([self.is_in_the_sphere(j, i) for i in j]) for j in TSs]
            labels_TSs = [[i["Name"] for i in j] for j in TSs]

        else:
            adj_mat_TSs = []
            labels_TSs = []




        adj_mats = [adj_mat, adj_mat_gas, adj_mat_TSs]
        labelss = [labels, labels_gas, labels_TSs]

        return adj_mats, labelss

    def define_connectivity(self):
        """The define_connectivity() functions condenses all the information retrieved by the
        make_adj_mtr() to compare the diferent adjacency matrixes present in the system in a
        connectivity dictionary (see below).
        returns conn_dicts (list): the conn_dicts list contains the lits of each one of the
        corresponding connectivity dictionaries for all the parsed systems."""

        #Invoke the make_adj_mtr() function
        adj_mats, labels = self.make_adj_mtr()
        #Separe both outputs of the make_adj_mtr() function
        mats_str = adj_mats[0]
        mats_gas = adj_mats[1]
        mats_TSs = adj_mats[2]

        labels_str = labels[0]
        labels_gas = labels[1]
        labels_TSs = labels[2]

        total_labels = []

        if not not labels_gas:
            for i in labels_gas:
                tmp = set(i)
                tmp = list(tmp)
                for j in tmp:
                    if j not in total_labels:
                        total_labels.append(j)
                    else:
                        continue
        else:
            for i in labels_str:
                if not i:
                    continue
                else:
                    tmp = set(i)
                    tmp = list(tmp)
                    for j in tmp:
                        if j not in total_labels:
                            total_labels.append(j)



        def fill_conn_dicts(mats, labels):
            #Declaration of the conn_dicts list
            conn_dicts = []
            def all_item(x, item):
                res = []
                for i in x:
                    if i not in item:
                        res.append(i)
                    else:
                        continue

                if not res:
                    return True
                else:
                    return False

            #For each element on the matrixes and the labels do:
            for i, j in zip(mats, labels):
                #Declaration of an empty con_dict, which contains the following keys:
                #"Empty": 1 if it a pristine surface; else: ''
                #"C": list containing the names of the atoms attached to C
                #"O": list containing the names of the atoms attached to the different oxygens
                    #The [-2] element of "O" list is the number of free (non attached to C) O atoms,
                #while the last element ([-1])  is the total number of oxygen atoms present
                #in the system.
                #"H": list containing the names attached to the different H atoms
                #The [-2] element of "H" list is the number of free H atoms, while
                #the last element ([-1])  is the total number of hydrogen atoms present in the
                #system.
                con_dict = {"Empty": ""}
                for label in total_labels:
                    con_dict[label] = ""

                #If there the labels list is empty (which means that there is no atoms) do:
                if not j:
                    #This a pristine surface: "Empty"=1
                    con_dict["Empty"] = 1
                    conn_dicts.append(con_dict)
                else:
                    indexes = []
                    nums = []


                    for labs in total_labels:
                        if labs in j:
                            indexes.append([w for w in range(len(j))
                                            if j[w]==labs])
                            tmp_num = [1 for n in j if n==labs]
                            nums.append(sum(tmp_num))
                        else:
                            indexes.append(None)
                            nums.append(0)

                    C_indexes = indexes[[i for i,j in
                                         zip(range(len(total_labels)),total_labels)
                                         if j=="C"][0]]
                    O_indexes = indexes[[i for i,j in
                                         zip(range(len(total_labels)),total_labels)
                                         if j=="O"][0]]

                    H_indexes = indexes[[i for i,j in
                                         zip(range(len(total_labels)),total_labels)
                                         if j=="H"][0]]
                    Num_os = nums[[i for i,j in zip(range(len(total_labels)),total_labels) if j=="O"][0]]

                    Num_hs = nums[[i for i,j in
                                         zip(range(len(total_labels)),total_labels)
                                         if j=="H"][0]]
                    Nums_cs = nums[[i for i,j in
                                    zip(range(len(total_labels)),total_labels)
                                    if j=="C"][0]]


                    connections = []
                    disconections = []
                    for ind1 in indexes:
                        tmp = []
                        tmp_2 = []
                        if ind1 == None:
                            tmp = tmp
                        else:
                            for ind2,lab2 in zip(indexes,total_labels):
                                if ind2 == None:
                                    tmp = tmp
                                else:
                                    for ind11 in ind1:
                                        if all_item(i[ind11],[0]):
                                            tmp_2.append(ind11)
                                        else:
                                            for ind22 in ind2:
                                                if i[ind11][ind22] == 1:
                                                    tmp.append(lab2)
                                                else:
                                                    continue
                        tmp.sort()
                        connections.append(tmp)
                        tmp_2 = set(tmp_2)
                        tmp_2 = list(tmp_2)
                        disconections.append(tmp_2)

                    for con,discon in zip(connections, disconections):
                        l = len(con)
                        if not discon:
                            con.append(0)
                            con.append(l)
                        else:
                            con.append(len(discon))
                            con.append(l+len(discon))

                    for labs,cons in zip(total_labels, connections):
                        if labs == "C":
                            c_counts = sum([1 for c in cons if c=="C"])
                            if 2*Nums_cs - 2 == c_counts:
                                cons[-1] = Nums_cs
                                cons[-2] = 0
                            elif 2*Nums_cs == c_counts:
                                cons[-1] = Nums_cs
                                cons[-2] = 0
                            else:
                                cons[-1] = Nums_cs
                                cons[-2] = 2*Nums_cs - 2 - c_counts
                        elif labs == "O":
                            cons[-1] = Num_os
                            o_counts = sum([1 for o in cons if o in ["C"]])
                            cons[-2] = Num_os - o_counts
                        elif labs == "N":
                            #TO DO: Define rules for nitrogen
                            print("Insert rules for Nitrogen")
                        else:
                            cons = cons

                        if cons != [0,0]:
                            con_dict[labs] = cons
                        else:
                            con_dict[labs] = ""
                    conn_dicts.append(con_dict)
            return conn_dicts
        if not not mats_str:
            conn_dicts_surf = fill_conn_dicts(mats_str, labels_str)
        else:
            conn_dicts_surf = []
        if not not mats_gas:
            conn_dicts_gas = fill_conn_dicts(mats_gas, labels_gas)
        else:
            conn_dicts_gas = []
        if not not mats_TSs:
            conn_dicts_TSs = fill_conn_dicts(mats_TSs, labels_TSs)
        else:
            conn_dicts_TSs = []
        return conn_dicts_surf, conn_dicts_gas, conn_dicts_TSs

    def split_paths(self):
        """The split_paths() function split the adjacency matrixes in two groups (or paths)
        in order """
        adj_mats, adj_mats_gas, adj_mat_TSs = self.define_connectivity()
        self.gas_conn_dicts = adj_mats_gas
        self.TSs_conn_dicts = adj_mat_TSs
        if not self.indexes_is_start:
            paths = adj_mats
        else:
            path_indexes = self.indexes_is_start
            paths = []
            paths.append(adj_mats[:path_indexes[0]])
            remainder_paths = adj_mats[path_indexes[0]:]
            paths.append(remainder_paths)
        return paths

    def crop_paths(self):
        paths = self.split_paths()
        paths[0] = [paths[0][i] for i in range(len(paths[0]))
                    if i not in self.crop_paths_info[1]]
        paths[1] = [paths[1][i] for i in range(len(paths[1]))
                    if i not in self.crop_paths_info[2]]


        return paths

    def redo_paths(self):
        if self.crop_paths_info[0] == True:
            croped_paths = self.crop_paths()
            unified_paths = croped_paths[0]
            for i in croped_paths[1]:
                unified_paths.append(i)
        elif not self.indexes_is_start:
            unified_paths = self.split_paths()
        else:
            splited_paths = self.split_paths()
            unified_paths = splited_paths[0]
            for i in splited_paths[1]:
                unified_paths.append(i)

        return unified_paths

    def surface_hydrogenation_test(self, m1, m2):

        Res = 0
        res = False
        test_res = []

        try:
            m1["H"][-1]
        except:
            test_res.append(True)
        else:
            test_res.append(False)
        try:
            m2["H"][-1]
        except:
            test_res.append(True)
        else:
            test_res.append(False)

        if (test_res[0] == True and test_res[1] == True):
            res = res
        elif (test_res[0] == True and test_res[1] == False):
            if (m2["H"][-1] == 2 and m2["H"][-2] == 2):
                res = True
            else:
                res = False
        elif (test_res[1] == True and test_res[0] == False):
            if (m1["H"][-1] == 2 and m1["H"][-2] == 2):
                res = True
            else:
                res = False
        elif (test_res[0] == False and test_res[1] == False):
            if (m1["H"][-1] == m2["H"][-1] + 2 and m1["H"][-2] == m2["H"][-2] + 2):
                res = True
            else:
                res = False


        if (self.react_hyd[0] == True):
            if(test_res[0] == False and test_res[1] == False):
                if (m2["H"][-1] == m1["H"][-1] + self.react_hyd[1] and
                    m2["H"][-2] == m1["H"][-2]):
                    res = True
                    Res = 1
                else:
                    res = res
            else:
                res = res
        try:
            (m1["C"] == m2["C"] and m1["O"][:-2] == m2["O"][:-2])
        except:
            Res = 0
        else:
            if (res == True and m1["C"] == m2["C"] and m1["O"][:-2] == m2["O"][:-2] and Res!=1):
                Res = 1
            else:
                Res = Res

        return Res

    def carbon_hydrogenation_test(self, m1, m2):

        Res = 0

        try:
            (m1["C"] == '' or m2["C"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["C"] == '' or m2["C"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_C_1 = m1["C"].copy()
                tmp_H_1 = m1["H"].copy()
                tmp_C_1.insert(0,"H")
                test = tmp_C_1[:-2]
                test.sort()
                tmp_C_1[:-2] = test
                tmp_H_1.insert(0, "C")
                tmp_H_1[-2] = m1["H"][-2] - 1
                if tmp_H_1[-2] < 0:
                    Res = Res
                elif (tmp_C_1 == m2["C"] and m1["O"] == m2["O"] and tmp_H_1 == m2["H"]):
                    Res = 1
                else:
                    Res = Res

        return Res

    def oxygen_hydrogenation_test(self, m1, m2):

        Res = 0

        try:
            (m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_O = m1["O"].copy()
                tmp_H_1 = m1["H"].copy()
                tmp_O.insert(-2, "H")
                test_o = tmp_O[:-2]
                test_o.sort()
                tmp_O[:-2] = test_o
                tmp_H_1.insert(-2, "O")
                test_H_1 = tmp_H_1[:-2]
                test_H_1.sort()
                tmp_H_1[:-2] = test_H_1
                tmp_H_1[-2] = m1["H"][-2] - 1
                if tmp_H_1[-2] < 0:
                    Res = Res
                elif (m1["C"] == m2["C"] and tmp_O == m2["O"] and tmp_H_1 == m2["H"]):
                    Res = 1
                else:
                    Res = Res

        return Res

    def nitrogen_hydrogenation_test(self, m1, m2):

        Res = 0

        try:
            (m1["N"] == '' or m2["N"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["N"] == '' or m2["N"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_O = m1["N"].copy()
                tmp_H_1 = m1["H"].copy()

                tmp_O.insert(-2, "H")
                test_o = tmp_O[:-2]
                test_o.sort()

                tmp_O[:-2] = test_o
                tmp_H_1.insert(-2, "N")

                test_H_1 = tmp_H_1[:-2]
                test_H_1.sort()
                tmp_H_1[:-2] = test_H_1

                tmp_H_1[-2] = m1["H"][-2] - 1
                if tmp_H_1[-2] < 0:
                    Res = Res
                elif (m1["C"] == m2["C"] and tmp_O == m2["N"] and tmp_H_1 == m2["H"]
                      and m1["O"] == m2["O"]):
                    Res = 1
                else:
                    Res = Res

        return Res


    def adsorption_on_empty_surf(self, m1, m2):

        Res = 0

        gas_species = self.gas_conn_dicts

        if not gas_species:
            Res = Res
        else:
            if (m1["Empty"] != 1):
                Res = Res
            else:
                for i in gas_species:
                    if m2 == i:
                        Res = 1
                        break
                    else:
                        Res = Res
        return Res

    def carbon_carbon_breaking_test(self, m1, m2):
        Res = 0

        try:
            (m1["C"] == '' or m2["C"] == '')
        except:
            Res = Res
        else:
            if (m1["C"] == '' or m2["C"] == ''):
                Res = Res
            elif (not "C" in m1["C"]):
                Res = Res
            else:
                tmp_C_1 = m1["C"].copy()
                tmp_C_1.remove("C")
                tmp_C_1[-2] = tmp_C_1[-2] + 1

                if (tmp_C_1 == m2["C"] and m1["O"] == m2["O"] and
                    m1["H"] == m2["H"]):
                    Res = 1
                else:
                    Res = Res

        return Res

    def nitrogen_nitrogen_breaking_test(self, m1, m2):
        Res = 0

        try:
            (m1["N"] == '' or m2["N"] == '')
        except:
            Res = Res
        else:
            if (m1["N"] == '' or m2["N"] == ''):
                Res = Res
            else:
                tmp_C_1 = m1["N"].copy()
                tmp_C_1.remove("N")
                tmp_C_1[-2] = tmp_C_1[-2] + 1

                if (tmp_C_1 == m2["N"] and m1["O"] == m2["O"] and
                    m1["H"] == m2["H"] and m1["C"] == m2["C"]):
                    Res = 1
                else:
                    Res = Res

        return Res
    def oxygen_carbon_breaking_test(self, m1, m2):

        Res = 0

        try:
            (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_C_1 = m1["C"].copy()
                tmp_O_1 = m1["O"].copy()
                tmp_H_1 = m1["H"].copy()
                tmp_C_1.remove("O")

                tmp_O_1.remove("C")
                tmp_O_1.insert(-2, "H")
                test_o = tmp_O_1[:-2]
                test_o.sort()
                tmp_O_1[:-2] = test_o

                tmp_H_1.insert(-2, "O")
                test_h = tmp_H_1[:-2]
                test_h.sort()
                tmp_H_1[:-2] = test_h

                tmp_H_1[-2] = m1["H"][-2] - 1
                tmp_O_1[-2] = m1["O"][-2] - 1

                if (tmp_H_1[-2] < 0):
                    Res = Res
                elif (tmp_C_1 == m2["C"] and tmp_O_1 == m2["O"] and tmp_H_1 == m2["H"]):
                    Res = 1
                else:
                    Res = Res

        return Res
    def nitrogen_carbon_breaking_test(self, m1, m2):

        Res = 0
        try:
            (m1["C"] == '' or m2["C"] == '' or m1["N"] == '' or m2["N"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["C"] == '' or m2["C"] == '' or m1["N"] == '' or m2["N"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_C_1 = m1["C"].copy()
                tmp_O_1 = m1["N"].copy()
                tmp_H_1 = m1["H"].copy()

                tmp_C_1.remove("N")
                tmp_O_1.remove("C")
                tmp_O_1[-2] = m1["N"][-2] - 1
                #tmp_O_1.insert(-2, "H")
                #tmp_H_1.insert(-2, "O")
                #tmp_H_1[-2] = m1["H"][-2] - 1
                if (tmp_H_1[-2] < 0):
                    Res = Res
                elif (tmp_C_1 == m2["C"] and tmp_O_1 == m2["N"] and
                      tmp_H_1 == m2["H"] and m1["O"]==m2["O"]):
                    Res = 1
                else:
                    Res = Res

        return Res
    def oxygen_carbon_breaking_test_2(self, m1, m2):

        Res = 0

        try:
            (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_C_1 = m1["C"].copy()
                tmp_O_1 = m1["O"].copy()
                tmp_H_1 = m1["H"].copy()

                tmp_C_1.remove("O")
                tmp_O_1.remove("C")
                tmp_O_1[-2] = m1["O"][-2] + 1
                #tmp_O_1.insert(-2, "H")
                #tmp_H_1.insert(-2, "O")
                #tmp_H_1[-2] = m1["H"][-2] - 1
                if (tmp_H_1[-2] < 0):
                    Res = Res
                elif (tmp_C_1 == m2["C"] and tmp_O_1 == m2["O"] and tmp_H_1 == m2["H"]):
                    Res = 1
                else:
                    Res = Res

        return Res

    def oxygen_carbon_breaking_test_3(self, m1, m2):

        flag = False
        Res = 0
        try:
            (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_C_1 = m1["C"].copy()
                tmp_O_1 = m1["O"].copy()
                tmp_H_1 = m1["H"].copy()

                tmp_C_1.remove("O")
                tmp_O_1.remove("C")
                try:
                    tmp_O_1.remove("H")
                except:
                    flag = False
                else:
                    tmp_O_1[-2] = tmp_O_1[-2] + 1

                    try:
                        tmp_H_1.remove("O")
                    except:
                        flag = False
                    else:
                        flag = True
                        tmp_H_1[-2] = tmp_H_1[-2] + 1

                if (tmp_H_1[-2] < 0 or flag == False):
                    Res = Res
                elif (tmp_C_1 == m2["C"] and tmp_O_1 == m2["O"] and tmp_H_1 == m2["H"]):
                    Res = 1
                else:
                    Res = Res


        return Res

    def oxygen_carbon_breaking_test_4(self, m1, m2):

        Res = 0

        try:
            (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == '')
        except:
            Res = Res
        else:
            if (m1["C"] == '' or m2["C"] == '' or m1["O"] == '' or m2["O"] == '' or m1["H"] == '' or m2["H"] == ''):
                Res = Res
            else:
                tmp_C_1 = m1["C"].copy()
                tmp_O_1 = m1["O"].copy()
                tmp_H_1 = m1["H"].copy()

                tmp_C_1.remove("O")
                tmp_O_1.remove("C")

                tmp_O_1.insert(-2, "H")
                test_o = tmp_O_1[:-2]
                test_o.sort()
                tmp_O_1 = test_o
                tmp_O_1[-2] = tmp_O_1[-2] + 1

                tmp_H_1.insert(-2, "O")
                test_h = tmp_H_1[:-2]
                test_h.sort()
                tmp_H_1 = test_h

                tmp_H_1[-2] = tmp_H_1[-2] - 1

                if (tmp_H_1[-2] < 0):
                    Res = Res
                elif (tmp_C_1 == m2["C"] and tmp_O_1 == m2["O"] and tmp_H_1 == m2["H"]):
                    Res = 1
                else:
                    Res = Res

        return Res


    def system_adjacency_matrix(self):

        adj_mat = self.redo_paths()

        self.conn_dicts = adj_mat

        sys_adj_mat = []

        for i, z in zip(adj_mat, range(len(adj_mat))):
            tmp = []
            for j, w in zip(adj_mat, range(len(adj_mat))):
                if z==w:
                    tmp.append(0)
                else:
                    operations = [self.adsorption_on_empty_surf(i, j), self.surface_hydrogenation_test(i, j),
                                 self.carbon_hydrogenation_test(i, j), self.oxygen_hydrogenation_test(i, j),
                                  self.oxygen_carbon_breaking_test_2(i,j), self.carbon_carbon_breaking_test(i,j),
                                  self.nitrogen_hydrogenation_test(i, j), self.nitrogen_carbon_breaking_test(i,j)]
                    if 1 in operations:
                        tmp.append(1)
                    else:
                        tmp.append(0)
            sys_adj_mat.append(tmp)




        return sys_adj_mat


    def make_mol_graphs(self, m, labels):
        gr = nx.Graph()

        edges = [ [labels[i], labels[j]] for i in range(len(labels)) for j in range(len(labels))
                 if (m[i][j]==1 or m[j][i]==1)]
        #empties = [ labels[k] for k in range(len(labels)) if np.asarray(m[k]).any() == False ]
        #if not empties:
        #    edges = edges
        #else:
        #    for l in empties:
        #        edges.append([l, l])
        gr.add_edges_from(edges)

        return gr

    def show_graph_with_labels(self, m, labels, name):
        fig = plt.figure()
        gr = self.make_mol_graphs(m, labels)
        nx.draw_networkx(gr)
        plt.savefig(name)

    def get_labeled_dicts(self, labels):
        Unlabeled_dicts = self.redo_paths()
        Labeled_dicts = []
        for i, j in zip(Unlabeled_dicts, labels):
            Dict = {'Empty': '', 'C': '', 'O': '', 'H': '', 'Label':''}
            Dict["Empty"] = i["Empty"]
            Dict["C"] = i["C"]
            Dict["O"] = i["O"]
            Dict["H"] = i["H"]
            Dict["Label"] = j
            Labeled_dicts.append(Dict)
        return Labeled_dicts

    def equal_dicts(self, d1, d2, ignore_keys):
        d1_filtered = {k:v for k,v in d1.items() if k not in ignore_keys}
        d2_filtered = {k:v for k,v in d2.items() if k not in ignore_keys}
        return d1_filtered == d2_filtered

    def get_stoich_mat(self, labels, gas_Labels, TS_labels, labels_gr_analysis):


        m = self.system_adjacency_matrix()
        Labels = ["i"+str(i) for i in range(len(labels))]
        self.int_labels = labels
        self.g_labels = gas_Labels
        label_dict = {i:j for i,j in zip(Labels,labels)}
        self.human_readable_labels = label_dict
        start = Labels[0]
        self.gr = self.make_mol_graphs(m, Labels)
        Intermediate_num = self.gr.number_of_nodes()
        Rx_num = self.gr.size()

        gr_Labels = self.make_mol_graphs(m,labels)

        labels_with_gas = Labels.copy()
        Labeled_dicts_gas = self.get_labeled_dicts(Labels)
        gas_species = self.gas_conn_dicts.copy()
        Labeled_dicts_TSs = []
        for ii, jj in zip(self.TSs_conn_dicts.copy(), TS_labels):
            ii["Label"] = jj
            Labeled_dicts_TSs.append(ii)

        if not gas_species:
            gas_species = gas_species
            Labeled_dicts_gas = Labeled_dicts_gas
            react_type = "No_gas"
        else:
            for ii,jj in zip(gas_species, gas_Labels):
                ii["Label"] = jj
                Labeled_dicts_gas.append(ii)
                labels_with_gas.append(ii["Label"])
            if gas_species[0]["H"][-1] < gas_species[-1]["H"][-1]:
                react_type = "Hyd"
            elif gas_species[0]["H"][-1] > gas_species[-1]["H"][-1]:
                react_type = "Dehyd"
            else:
                react_type = "No_hyd"

        stoich_mat = np.zeros([Intermediate_num+len(gas_species), Rx_num])
        edges_list_0 = [list(i) for i in list(gr_Labels.edges)]
        empties = [label for label in labels
                   if label not in list(gr_Labels.nodes)]
        if not empties:
            self.graph_paths = [i for i in list(nx.all_simple_paths(gr_Labels,
                                                                    source=labels_gr_analysis[0],
                                                                    target=labels_gr_analysis[-1]))
                            if len(i)>4]

        else:
            labels_2 = labels.copy()
            for empty in empties:
                index_to_remove = labels_2.index(empty)
                Labels.remove(Labels[index_to_remove])
                Labeled_dicts_gas.remove(Labeled_dicts_gas[index_to_remove])
                labels_2.remove(empty)


            self.graph_paths = [i for i in list(nx.all_simple_paths(gr_Labels,
                                                                    source=labels_2[0],
                                                                    target=labels_2[-1]))
                                ]


        Connectivity = [ [ii, list(self.gr.adj[ii])] for ii in Labels]



        cons = [ii[1] for ii in Connectivity]
        Rxs = [list(ii) if float(list(ii)[0][1:]) < float(list(ii)[1][1:])
               else [list(ii)[1], list(ii)[0]] for ii in self.gr.edges()]



        def sort_by_fs(x):
            return x[1]

        Rxs.sort(key=sort_by_fs)

        start_counting = 0

        for i in Rxs:
            if i[0] == start:
                start_counting = start_counting + 1
                if start_counting > 2:
                    self.Rxs_resorted.append([i[1], i[0]])
                else:
                    self.Rxs_resorted.append(i)
            else:
                self.Rxs_resorted.append(i)

        #Re-order to compare with hand-made stoichiometric matrix
        #List_test.insert(20, self.Rxs_resorted[18])
        #List_test.pop(-3)
        #List_test.insert(20, self.Rxs_resorted[16])
        #List_test.pop(-5)
        #List_test.insert(-4, self.Rxs_resorted[-6])
        #List_test.pop(-7)




        len_C_TSs = [len(i["C"]) for i in self.TSs_conn_dicts]

        i_s = [i[0] for i in self.Rxs_resorted]
        f_s = [i[1] for i in self.Rxs_resorted]
        for j in range(len(self.Rxs_resorted)):
            tmp = []
            flag_gas = True
            i_1 = Labels.index(self.Rxs_resorted[j][0])
            i_2 = Labels.index(self.Rxs_resorted[j][1])
            stoich_mat[i_1][j] = -1
            stoich_mat[i_2][j] = 1
            if react_type == "Hyd":
                if ( (Labeled_dicts_gas[i_1]["Empty"] == 1 or
                      self.equal_dicts(Labeled_dicts_gas[i_1],gas_species[1], ["Label"]))
                    and self.equal_dicts(Labeled_dicts_gas[i_2], gas_species[0], ["Label", "H", "C"]) ):
                    stoich_mat[Intermediate_num][j] = -1
                    i_s[j] = i_s[j] + "+" +gas_Labels[0]
                elif ( (Labeled_dicts_gas[i_1]["H"]==''
                        and self.equal_dicts(Labeled_dicts_gas[i_2], gas_species[1], ["Label"]) ) or
                      (Labeled_dicts_gas[i_1]["H"]!='' and Labeled_dicts_gas[i_2]["H"]!=''
                       and Labeled_dicts_gas[i_1]["H"][-1] + 2 == Labeled_dicts_gas[i_2]["H"][-1]
                       and Labeled_dicts_gas[i_1]["H"][-2] + 2 == Labeled_dicts_gas[i_2]["H"][-2] )):
                    stoich_mat[Intermediate_num+1][j] = -1
                    i_s[j] = i_s[j] + "+" +gas_Labels[1]
                elif ( (i_2==0 and self.equal_dicts(Labeled_dicts_gas[i_1], gas_species[2], ["Label"]) ) or
                      (i_1==0 and self.equal_dicts(Labeled_dicts_gas[i_1], gas_species[2], ["Label"]))):
                    stoich_mat[Intermediate_num+2][j] = 1
                    f_s[j] = f_s[j] + "+" +gas_Labels[2]
            elif react_type == "Dehyd":
                for i,k,m in zip(gas_species, gas_Labels, range(len(gas_species))):
                    if (self.equal_dicts(Labeled_dicts_gas[i_1], i, ["Label"]) and i_2 == 0):
                        stoich_mat[Intermediate_num+m][j] = 1
                        i_s[j] = i_s[j] + "+" + k
                        flag_gas = False
                    elif (self.equal_dicts(Labeled_dicts_gas[i_2], i, ["Label"]) and i_1 == 0):
                        stoich_mat[Intermediate_num+m][j] = -1
                        f_s[j] = f_s[j] + "+" + k
                        flag_gas = False
                    else:
                        continue
                if (Labeled_dicts_gas[i_1]["H"]!='' and Labeled_dicts_gas[i_2]["H"]!=''
                    and Labeled_dicts_gas[i_2]["H"][-1] + 2 == Labeled_dicts_gas[i_1]["H"][-1]
                    and Labeled_dicts_gas[i_2]["H"][-2] + 2 == Labeled_dicts_gas[i_1]["H"][-2]):
                    stoich_mat[Intermediate_num+1][j] = 1
                    f_s[j] = f_s[j]+"+"+gas_Labels[1]
                    flag_gas = False
                elif ( Labeled_dicts_gas[i_1]["H"]!='' and Labeled_dicts_gas[i_2]["H"]!='' and
                      ( (Labeled_dicts_gas[i_2]["H"][-2] == Labeled_dicts_gas[i_1]["H"][-2] + 2
                         or Labeled_dicts_gas[i_2]["H"][-1] == Labeled_dicts_gas[i_1]["H"][-1]
                         + self.react_hyd[1]))):
                    stoich_mat[Intermediate_num+2][j] = -1
                    i_s[j] = i_s[j] + "+" +gas_Labels[2]
                    flag_gas = False
            elif react_type == "No_gas":
                for j,lj in zip(self.TSs_conn_dicts, range(len(self.TSs_conn_dicts))):
                    if (self.equal_dicts(Labeled_dicts_gas[i_2], j,["Label"]) and flag_gas):
                        tmp.append(i_2)
                        tmp.append(lj)
                    elif (self.equal_dicts(Labeled_dicts_gas[i_1], j,["Label"]) and flag_gas):
                        tmp.append(i_1)
                        tmp.append(lj)
                    else:
                        continue
            else:
                for i,k,m in zip(gas_species, gas_Labels, range(len(gas_species))):
                    if (self.equal_dicts(Labeled_dicts_gas[i_1], i, ["Label"]) and i_2 == 0):
                        stoich_mat[Intermediate_num+m][j] = 1
                        i_s[j] = i_s[j] + "+" + k
                        flag_gas = False
                    elif (self.equal_dicts(Labeled_dicts_gas[i_2], i, ["Label"]) and i_1 == 0):
                        stoich_mat[Intermediate_num+m][j] = -1
                        f_s[j] = f_s[j] + "+" + k
                        flag_gas = False
                    else:
                        continue




            for j,lj in zip(self.TSs_conn_dicts, range(len(self.TSs_conn_dicts))):
                if (self.equal_dicts(Labeled_dicts_gas[i_2], j,["Label"]) and i_1 != 0 and i_2 != 0
                   and flag_gas):
                    tmp.append(i_2)
                    tmp.append(lj)
                elif (self.equal_dicts(Labeled_dicts_gas[i_1], j,["Label"]) and i_2 != 0 and i_1 != 0
                     and flag_gas):
                    tmp.append(i_1)
                    tmp.append(lj)
                else:
                    continue

            self.Assign_TS.append(tmp)




        R_id = ["R"+str(i) for i in range(1,Rx_num+1)]
        try:
            pd.DataFrame(stoich_mat, columns=["R"+str(i) for i in range(1,Rx_num+1)], index=labels_with_gas)
        except:
            df = pd.DataFrame(stoich_mat, columns=["R"+str(i) for i in range(1,Rx_num+1)], index=labels_with_gas[1:])
        else:
            df = pd.DataFrame(stoich_mat, columns=["R"+str(i) for i in range(1,Rx_num+1)], index=labels_with_gas)



        df.to_csv("./Stoich_mat.csv")
        i_ss = []
        f_ss = []
        for label in R_id:
            i_s = []
            f_s = []
            for i,j in zip(df[label], df[label].index):
                if i == -1:
                    i_s.append(j)
                elif i == 1:
                    f_s.append(j)
                else:
                    continue
            i_ss.append(i_s)
            f_ss.append(f_s)
        with open("./Human_readable_reactions.txt", "w") as outf:
            for i in range(len(i_ss)):
                token_is = ""
                token_fs = ""
                if len(i_ss[i]) == 1:
                    token_is = label_dict[i_ss[i][0]]
                else:
                    token_is = label_dict[i_ss[i][0]] + "+" + i_ss[i][1]
                if len(f_ss[i]) == 1:
                    token_fs = label_dict[f_ss[i][0]]
                else:
                    token_fs = label_dict[f_ss[i][0]] + "+" + f_ss[i][1]



                outf.write(token_is+" "+"->"+" "+token_fs)
                outf.write("\n")

        if react_type == "Hyd":
            first_line = "Add Reaction Label Add Reaction Number: "
            first_line += gas_Labels[0]+" "+"+"+" "+gas_Labels[1]+" "+"->"
            first_line += " "+gas_Labels[-1]
        elif react_type == "Dehyd":
            first_line = "Add Reaction Label: Add Reaction Number "
            first_line += gas_Labels[0]+" "+"->"
            first_line += " "+gas_Labels[-1]+" "+"+"+" "+gas_Labels[1]
        elif react_type == "No_gas":
            first_line = ''
        else:
            first_line = "Add Reaction Label Add Reaction Number: "


        with open("./rm.mkm", "w") as outf:
            outf.write(first_line+"\n")
            for _ in range(3):
                outf.write("\n")
            for i in range(len(i_ss)):
                token_is = ""
                token_fs = ""
                if len(i_ss[i]) == 1:
                    token_is = i_ss[i][0]
                else:
                    token_is = i_ss[i][0] +" " + "+" +" " +i_ss[i][1]
                if len(f_ss[i]) == 1:
                    token_fs = f_ss[i][0]
                else:
                    token_fs = f_ss[i][0] +" " + "+" + " " +f_ss[i][1]



                outf.write(token_is+" "+"->"+" "+token_fs)
                outf.write("\n")

        return df

    def get_energies_and_entropies(self):
        T = self.T_P[0]
        P = self.T_P[1]
        A, U, S_a, Ep, ElE_s = self.surface_preprocessor.Helmholtz(T)
        A_ts, U_ts, S_ts, Ep_ts, ElE_ts = self.TSs_preprocessor.Helmholtz(T)

        A_cp = A.copy()
        U_cp = U.copy()

        A_tscp = A_ts.copy()
        U_tscp = U_ts.copy()
        if type(self.gas_preprocessor) != list:
            G, H, S_g,ElE_g = self.gas_preprocessor.Gibbs(T, P)
            for i in self.add_atom[0]:
                if i[0] == 0:
                    A_cp[i[1]] += G[1]*0.5
                    U_cp[i[1]] += H[1]*0.5
                elif i[0] == 2:
                    A_tscp[i[1]] += G[1]*0.5
                    U_tscp[i[1]] += H[1]*0.5
                else:
                    continue


        else:
            G,H,S_g,ElE_g = [],[],[],[]
            A_cp = A_cp
            U_cp = U_cp
            A_tscp = A_tscp
            U_tscp = U_tscp

        Gas = {"G":G, "H":H, "S_g":S_g, "ElE_g":ElE_g}
        Surface = {"A":A_cp, "U":U_cp, "S_a":S_a, "Ep_s":Ep}
        TS = {"A_ts":A_tscp, "U_ts":U_tscp, "S_ts":S_ts, "Ep_ts":Ep_ts}
        return Gas, Surface, TS

    def export_energies(self, outfname):
        gas, surface, TSs = self.get_energies_and_entropies()


        A_TS = list(TSs["A_ts"])
        U_TS = list(TSs["U_ts"])
        S_TS = list(TSs["S_ts"])

        Gas_G = list(gas["G"])
        Gas_S = list(gas["S_g"])

        surface_A = list(surface["A"])
        surface_S = list(surface["S_a"])

        surface_U = list(surface["U"])
        Gas_H = list(gas["H"])
        TS_U = list(TSs["U_ts"])

        def try_state(state):
            res = False
            try:
                state
            except:
                res = res
                return res
            else:
                return state

        test = []
        aaa = self.Assign_TS
        ran = [i for i in range(len(aaa))]
        count = 0
        for i,j in zip(ran,range(len(ran))):
            if not aaa[i]:
                test.append("")
            elif (len(aaa[i]) > 2 and count <= 0):
                count += 1
                if not not aaa[i+1]:
                    test.append(aaa[i][1])
                    test.append(aaa[i][3])
                else:
                    test.append(aaa[i][1])
                    test.append("")
                try:
                    ran.pop(j+1)
                except:
                    test[-2] = ""
                    test[-1] = ""
                else:
                    test = test
            elif (len(aaa[i]) == 6 and try_state(len(aaa[i+1]))==4):
                count += 1
                if not not aaa[i+1]:
                    test.append(aaa[i][-1])
                    test.append(aaa[i+1][-1])
                else:
                    test.append(aaa[i][-1])
                    test.append("")
                try:
                    ran.pop(j+1)
                except:
                    break
                else:
                    continue
            elif (len(aaa[i]) == 6 and try_state(len(aaa[i+1]))==6):
                count += 1
                if not not aaa[i+1]:
                    test.append(aaa[i][3])
                    test.append(aaa[i+1][5])
                else:
                    test.append(aaa[i][-1])
                    test.append("")
                try:
                    ran.pop(j+1)
                except:
                    break
                else:
                    continue
            elif len(aaa[i])==4:
                count +=1
                if not not aaa[i+1]:
                    test.append(aaa[i][1])
                    test.append(aaa[i][3])
                else:
                    test.append(aaa[i][1])
                    test.append("")
                try:
                    ran.pop(j+1)
                except:
                    test[-2] = ""
                    test[-1] = ""
            elif (len(aaa[i])==6 and try_state(len(aaa[i+1]))==2):
                count += 1
                test.append(aaa[i][-1])
                test.append(aaa[i+1][1])
                try:
                    ran.pop(j+1)
                except:
                    break
                else:
                    continue
            else:
                count +=1
                test.append(aaa[i][1])

        self.Assigned_TSs = test
        self.list_TSs = [U_TS[i] if i!="" else 0 for i in test]
        list_TSs_S = [S_TS[i] if i!="" else 0 for i in test]

        if not not Gas_H:
            New_gas_h = []
            New_gas_h.append(Gas_H[1])
            New_gas_s = []
            New_gas_s.append(Gas_S[1])
            for i,j in zip(Gas_H, Gas_S):
                if i not in New_gas_h:
                    New_gas_h.append(i)
                if j not in New_gas_s:
                    New_gas_s.append(j)
        else:
            New_gas_h = Gas_G
            New_gas_s = Gas_S

        with open(outfname, "w") as inf:
            for i,j in zip(self.list_TSs, list_TSs_S):
                inf.write(str(i)+" "+str(j)+"\n")
            for _ in range(3):
                inf.write("\n")
            for i,j in zip(surface_U, surface_S):
                inf.write(str(i)+" "+str(j)+"\n")
            for _ in range(3):
                inf.write("\n")
            if not not New_gas_h:
                for i,j in zip(New_gas_h, New_gas_s):
                    inf.write(str(i)+" "+str(j)+"\n")
            else:
                inf.close()


        test = self.Rxs_resorted.copy()

        TS_order = []

        for i,j in zip(test, self.list_TSs):
            if j == 0:
                TS_order.append(0)
            else:
                self.TS_s_exact_labels.append("TS"+self.human_readable_labels[i[0]]
                                         +","+ self.human_readable_labels[i[1]])
                TS_order.append(self.TS_s_exact_labels[-1])

        if type(self.gas_preprocessor) != list:
            for i in self.add_atom[0]:
                if i[0] == 0:
                    surface["A"][i[1]] -= gas["G"][1]*0.5
                    surface["U"][i[1]] -= gas["H"][1]*0.5
                elif i[0] == 2:
                    TSs["A_ts"][i[1]] -= gas["G"][1]*0.5
                    TS_U[i[1]] -= gas["H"][1]*0.5
                else:
                    continue

        surf_A = []
        surf_U = []
        surf_S = []
        for i in range(len(surface["A"])):
            surf_A.append(surface["A"][i])
            surf_U.append(surface["U"][i])
            surf_S.append(surface["S_a"][i])

        TS_A = TSs["A_ts"].copy()
        TS_S = TSs["S_ts"].copy()
        Gas_g = gas["G"].copy()
        Gas_u = gas["H"].copy()
        Gas_s = gas["S_g"].copy()

        All_surf_labels = self.int_labels.copy()

        for i,j,k,l in zip(TS_A, self.TS_s_exact_labels, TS_U, TS_S):
            surf_A.append(i)
            surf_U.append(k)
            surf_S.append(l)
            All_surf_labels.append(j)
        if not not Gas_g:
            for i,j,k,l in zip(Gas_g, self.g_labels, Gas_u, Gas_s):
                surf_A.append(i)
                surf_U.append(k)
                surf_S.append(l)
                All_surf_labels.append(j)
        else:
            All_surf_labels = All_surf_labels
            surf_U = surf_U
            surf_A = surf_A

        Energies_dict_A = {i:j for i,j in zip(All_surf_labels, surf_A)}
        Energies_dict_U = {i:j for i,j in zip(All_surf_labels, surf_U)}
        Energies_dict_S = {i:j for i,j in zip(All_surf_labels, surf_S)}

        self.dict_complete = {j:i for i,j in zip(self.conn_dicts, self.int_labels)}
        if not not self.gas_conn_dicts:
            for i,j in zip(self.gas_conn_dicts, self.g_labels):
                self.dict_complete[j] = i
        else:
            self.gas_conn_dicts = self.gas_conn_dicts
        for i,j in zip(self.TSs_conn_dicts, self.TS_s_exact_labels):
            self.dict_complete[j] = i

        paths = self.graph_paths

        for path in paths:
            for i in range(len(path)):
                ts_label = "TS"+path[i] +","+ path[i+1]
                if ts_label in self.TS_s_exact_labels:
                    path.insert(i+1,ts_label)
                else:
                    continue



        energy_dict_free = Energies_dict_A


        if not not self.reference:
            surf = self.reference[0]
            nsurf = self.reference[1]
            reac = self.reference[2]
            prod = self.reference[3]
            hyd_1 = self.reference[4]

            paths_energy = []

            for path in paths:
                energy = {}
                if hyd_1[2] == True:
                    ref = nsurf*energy_dict_free[surf] + energy_dict_free[reac] + energy_dict_free[hyd_1[4]]
                else:
                    ref = nsurf*energy_dict_free[surf] + energy_dict_free[reac]
                for i in path:
                    energy[i] = energy_dict_free[i]
                    if i == surf:
                        energy[i] = 0.0
                    elif (self.equal_dicts(self.dict_complete[i],
                                               self.dict_complete[prod], ["Label"]) and not "TS" in i):
                        energy[i] = energy_dict_free[i] - energy_dict_free[prod]
                        energy[i] = energy[i] - energy_dict_free[surf]
                    elif self.equal_dicts(self.dict_complete[i],
                                          self.dict_complete[reac], ["Label"]):
                        energy[i] = energy_dict_free[i] - energy_dict_free[reac]
                        energy[i] = energy[i] - energy_dict_free[surf]
                    elif hyd_1[0] == True:
                        if self.equal_dicts(self.dict_complete[i],
                                            self.dict_complete[hyd_1[1]], ["Label"]):
                            if hyd_1[2] == True:
                                energy[i] += (hyd_1[3])*energy_dict_free[hyd_1[1]]+energy_dict_free[reac]
                                energy[i] += energy_dict_free[surf]
                                energy[i] -= ref
                            else:
                                energy[i] += (hyd_1[3])*energy_dict_free[hyd_1[1]] + energy_dict_free[reac] - energy_dict_free[hyd_1[4]]
                                energy[i] += energy_dict_free[surf]
                                energy[i] -= ref

                        elif "TS" in i:
                            lab = i[2:i.index(",")]
                            lab_2 = i[i.index(",")+1:]
                            if self.dict_complete[lab]["H"][-2] < self.dict_complete[lab_2]["H"][-2]:
                                lab = lab
                            else:
                                lab = lab_2
                            energy[i] += self.dict_complete[lab]["H"][-2]*energy_dict_free[hyd_1[1]]
                            energy[i] += (nsurf-self.dict_complete[lab]["H"][-2]-1)*energy_dict_free[surf]
                            energy[i] -= ref
                        else:
                            energy[i] = energy[i] + (self.dict_complete[i]["H"][-2])*energy_dict_free[hyd_1[1]] + (nsurf-self.dict_complete[i]["H"][-2]-1)*energy_dict_free[surf]
                            energy[i] -= ref
                    else:
                        energy[i] = energy[i]
                        energy[i] -= ref
                paths_energy.append(energy)
            energies_to_show = []
            referenced_g = {}
            for path in paths_energy:
                test = path
                energy_to_show = [ test[key] for key in list(test.keys())]
                energies_to_show.append(energy_to_show)
                for key in list(test.keys()):
                    try:
                        referenced_g[key]
                    except:
                        referenced_g[key] = test[key]
                    else:
                        continue

            non_referenced_states = []

            for key in list(Energies_dict_A.keys()):
                if key not in list(referenced_g.keys()):
                    non_referenced_states.append(key)
                else:
                    continue

            if not non_referenced_states:
                referenced_g = referenced_g
            else:
                non_referenced_states_energy = []
                if hyd_1[2] == True:
                    ref = nsurf*energy_dict_free[surf] + energy_dict_free[reac] + energy_dict_free[hyd_1[4]]
                else:
                    ref = nsurf*energy_dict_free[surf] + energy_dict_free[reac]
                for i in non_referenced_states:
                    energy = {}
                    energy[i] = energy_dict_free[i]
                    if i == surf:
                        energy[i] = 0.0
                    elif (self.equal_dicts(self.dict_complete[i],
                                               self.dict_complete[prod], ["Label"]) and not "TS" in i):
                        energy[i] = energy_dict_free[i] - energy_dict_free[prod]
                        energy[i] = energy[i] - energy_dict_free[surf]
                    elif self.equal_dicts(self.dict_complete[i],
                                          self.dict_complete[reac], ["Label"]):
                        energy[i] = energy_dict_free[i] - energy_dict_free[reac]
                        energy[i] = energy[i] - energy_dict_free[surf]
                    elif hyd_1[0] == True:
                        if self.equal_dicts(self.dict_complete[i],
                                            self.dict_complete[hyd_1[1]], ["Label"]):
                            if hyd_1[2] == True:
                                energy[i] += (hyd_1[3])*energy_dict_free[hyd_1[1]]+energy_dict_free[reac]
                                energy[i] += energy_dict_free[surf]
                                energy[i] -= ref
                            else:
                                energy[i] += (hyd_1[3])*energy_dict_free[hyd_1[1]] + energy_dict_free[reac] - energy_dict_free[hyd_1[4]]
                                energy[i] += energy_dict_free[surf]
                                energy[i] -= ref

                        elif "TS" in i:
                            lab = i[2:i.index(",")]
                            lab_2 = i[i.index(",")+1:]
                            if self.dict_complete[lab]["H"][-2] < self.dict_complete[lab_2]["H"][-2]:
                                lab = lab
                            else:
                                lab = lab_2
                            energy[i] += self.dict_complete[lab]["H"][-2]*energy_dict_free[hyd_1[1]]
                            energy[i] += (nsurf-self.dict_complete[lab]["H"][-2]-1)*energy_dict_free[surf]
                            energy[i] -= ref
                        else:
                            energy[i] = energy[i] + (self.dict_complete[i]["H"][-2])*energy_dict_free[hyd_1[1]] + (nsurf-self.dict_complete[i]["H"][-2]-1)*energy_dict_free[surf]
                            energy[i] -= ref
                    else:
                        energy[i] = energy[i]
                        energy[i] -= ref
                    non_referenced_states_energy.append(energy)

                for i in non_referenced_states_energy:
                    referenced_g[list(i.keys())[0]] = i[list(i.keys())[0]]

            xs = [ [j for j in range(len(i))] for i in energies_to_show ]

            for i,j,k in zip(xs, energies_to_show, self.colors):
                plt.plot(i,j,color=k)
                plt.savefig("profile.png")

            keys = list(self.human_readable_labels.keys())
            keys.sort()

            new_energy_list = []
            for key in keys:
                new_energy_list.append(referenced_g[self.human_readable_labels[key]])
            new_TS_list = []
            for i in TS_order:
                if i != 0:
                    new_TS_list.append(referenced_g[i])
                else:
                    new_TS_list.append(i)

            new_outfname = outfname[:outfname.index(".")]
            new_outfname += "_ref"
            new_outfname += outfname[outfname.index("."):]
            with open(new_outfname, "w") as inf:
                for i,j in zip(new_TS_list, list_TSs_S):
                    inf.write(str(i)+" "+str(0)+"\n")
                for _ in range(3):
                    inf.write("\n")
                for i,j in zip(new_energy_list, surface_S):
                    inf.write(str(i)+" "+str(0)+"\n")
                for _ in range(3):
                    inf.write("\n")
                if not not New_gas_h:
                    for i,j in zip(New_gas_h, New_gas_s):
                        inf.write(str(0)+" "+str(0)+"\n")
                else:
                    inf.close()


        else:
            paths_energy = []



        return gas,surface,TSs,Energies_dict_A,Energies_dict_U,Energies_dict_S,paths_energy

    def OpenFOAM_mechanism(self, path):
        with open(path, "r") as inf:
            data = [line.strip().split() for line in inf]
        indexes = []
        for rx in data:
            index_i = ""
            index_j = ""
            for i in range(len(rx)):
                if "+" in rx[i]:
                    for j in range(len(rx[i])):
                        if rx[i][j] == "+":
                            index_i = i
                            index_j = j
                        else:
                            continue
                else:
                    continue

                if rx[index_i][index_j]!="+":
                    index_i = ""
                    index_j = ""
                else:
                     index_i = index_i
                     index_j = index_j
            indexes.append([index_i, index_j])
        for test, indexs in zip(data, indexes):
            index_0 = indexs[0]
            index_1 = indexs[1]
            if (index_0 != "" and index_1 != ""):
                if (test[index_0][:index_1] == self.reference[0] and index_0 < len(test)-1):
                    if test[-1][0] == "2":
                        new_test = "2"
                        new_test += test[index_0][:index_1]
                        new_test += test[index_0][index_1:]
                        test[index_0] = new_test
                    else:
                        continue
                else:
                    test = test
            else:
                dct = self.dict_complete[test[0]]["H"][-2]
                dct2 = self.dict_complete[test[-1]]["H"][-2]

                if dct < dct2:
                    test[0] += "+"+self.reference[0]
                    test[-1] += "+"+str(dct2-dct)+"H"
                else:
                    test[-1] += "+"+self.reference[0]
                    test[0] += "+"+str(dct-dct2)+"H"
        with open("PreOpenFoam_readable_reactions.txt", "w") as outf:
            for rx in data:
                for item in rx:
                    outf.write(item)
                    outf.write(" ")
                outf.write("\n")
        return data




    def graph_analysis(self, init_node):
        edges = list( list(i) for i in self.gr.edges)
        for i,j in zip(edges, self.list_TSs):
            self.gr[i[0]][i[1]]["weight"] = j
        output = nx.dijkstra_predecessor_and_distance(self.gr, init_node)
        return output



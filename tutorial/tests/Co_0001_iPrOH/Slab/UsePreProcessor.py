#Imports
import os
import sys
import AutoProfLib_DEF as APL
############################################################################################
#General instructions and input gathering
print("This is an example of the use of the PreProcessor class. This script should be run in the same folder than the AutoProfLib_DEF library.")
print("First, we need to initialize the PreProcessor class, and specify which is free energy that you whish to calculate (Gibbs or Helmholtz).")
flag=str(input("Please, indicate if you want to calculate the Helmholtzi (H) or the Gibbs (G) free energy: "))
print("To do so, we need the following inputs:")
print("1.The directory where the CONTCAR file and the FREQ or ../Freq directory are:")
files=str(input("File: "))

T = float(input("2. Temperature in K: "))
if ("H" in flag or "h" in flag):
    #Calculate Helmholtz free energy
    spin = 0
    geometries = None

    max_pbc = 0.66

    print("3. The frequency list processing")
    print("3.1. Frequency list options. It can be Erase, Substitute or Grimme")
    max_freq_0 = str(input("Set the flag to the frequency process Substitute (S), Erase (E), Grimme (G): "))
    if (max_freq_0[0]=="G" or max_freq_0[0]=="g"):
        max_freq_0 = "Grimme"
    elif (max_freq_0[0]=="E" or max_freq_0[0]=="e"):
        max_freq_0 = "Erase"
    else:
        max_freq_0 = "Substitute"
    if max_freq_0 in ["Substitute", "Erase"]:

        max_freq_2 = False
        try:
            freq = float(input("Treshold (recommended 100 cm^{-1}): "))
        except:
            max_freq_1 = None
        else:
            max_freq_1 = freq
            max_freq_1 = max_freq_1 * 0.0124 / 100
    else:
        max_freq_1 = None
        flag_3 = str(input("Auxiliar Grimme option. Do you want to use the average frequency correction? (Yes/No)"))
        if ("Y" in flag_3 or "y" in flag_3):
            max_freq_2 = True
        else:
            max_freq_2 = False


    #Enclose the answers in a list
    max_freq = [max_freq_0, max_freq_1, max_freq_2]
    flag_pbc = str(input("Would you like to apply PBC to your molecule? (Yes/No): "))
    if (flag_pbc[0] == "Y" or flag_pbc[0] == "y"):
        max_pbc = float(input("Set the maximum treshold (in partitive coordinates) in which the PBC will be applied: "))
        index_mol = int(input("Finally, insert the index from which the atoms in the system are molecules: "))
        test = APL.PreProcessor(files, geometries, max_pbc, max_freq, spin)
        #Application of the get_cells_and_part_coords() function
        Cell, Coordinates, Labels = test.get_cells_and_part_coords()
        #Selection of the molecules
        mols = Coordinates[0][index_mol:]
        labels = Labels[0][index_mol:]
        #Application of the apply_pbc(mols, cells, labels) function
        new_coords = test.apply_pbc([mols], Cell, [labels])[0]
        print("Now the _pbc.xyz file is generated in "+files)
        print("Here are the new coordinates:")
        for nc in new_coords:
            print(nc)
        test.Helmholtz(T)
    else:
        test = APL.PreProcessor(files, geometries, max_pbc, max_freq, spin)
        test.Helmholtz(T)

else:
    P = float(input("3. Pressure in atm: "))
    P=P*101325
    #Calculate the Gibbs free energy
    max_pbc = 0.66

    print("4. The geometry of the molecule, as a str. The accepted keys are linear(l), nonlinear(n), monoatomic(m) or None.")
    geometries=str(input("Geometry: "))
    print("5. The spin")
    spin=int(input("Spin: "))
    print("6. The frequency list processing")
    print("6.1. Frequency list options. It can be Erase, Substitute or Grimme")
    max_freq_0 = str(input("Set the flag to the frequency process Substitute (S), Erase (E), Grimme (G): "))
    if (max_freq_0[0]=="G" or max_freq_0[0]=="g"):
        max_freq_0 = "Grimme"
    elif (max_freq_0[0]=="E" or max_freq_0[0]=="e"):
        max_freq_0 = "Erase"
    else:
        max_freq_0 = "Substitute"
    if max_freq_0 in ["Substitute", "Erase"]:

        max_freq_2 = False
        try:
            freq = float(input("6.2. Treshold (recommended 100 cm^{-1}): "))
        except:
            max_freq_1 = None
        else:
            max_freq_1 = freq
            max_freq_1 = max_freq_1 * 0.0124 / 100
    else:
        max_freq_1 = None
        flag_3 = str(input("6.3. Auxiliar Grimme option. Do you want to use the average frequency correction? (Yes/No)"))
        if ("Y" in flag_3 or "y" in flag_3):
            max_freq_2 = True
        else:
            max_freq_2 = False


    #Enclose the answers in a list
    max_freq = [max_freq_0, max_freq_1, max_freq_2]
    #Initialization of the PreProcessor class
    test = APL.PreProcessor(files, geometries, max_pbc, max_freq, spin)
    test.Gibbs(T,P)

##########################################################################################################

#Finally, the PreProcessor object can be used with a list of structures instead of a single molecule

#To do that, just fill the following lists and floats and uncomment the test_2 example

#files_2 = []

#geometries_2 = []

#max_pbc_2 = 0.0

#max_freq_2 = ["Substitute", 0.0124, False]

#spin_2 = []

#test_2 = APL.PreProcessor(files_2, geometries_2, max_pbc_2, max_freq_2, spin_2)
############################################################################################










To use the Auto_MKM_toOF.py, two files are required:

1) The PreOpenFoam_readable_reactions.txt file, automatically generated by the AutoProfLib.
2) The Barriers.csv file, generated automatically by the PyMKM.

This script assumes that the species containing the _ character are composites,
and actually are defining a state. It also makes a first simplification of the 
mechanism balancing the number of hydrogens for reactants and products.

To run Auto_MKM_toOF.py, open a prompt, type:

python Auto_MKM_toOF.py -h

and follow the instructions that will appear in the terminal. 

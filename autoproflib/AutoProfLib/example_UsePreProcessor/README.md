Instructions:

1. Run the python script UsePreProcessor.py
2. Follow the instructions of the script
3. Only CONTCAR and OUTCAR files can be parsed (in the current realease)
4. When providing the path to the directory containing the frequency folder and the optimization results,
   do not provide the absolute path as ~/path_to_dir, instead provide /home/username/path_to_dir
5. In the current realease, is not mandatory to name the direcory of the frequency calculations as FREQ or ../Freq.
   However, the the PreProcessor class assumes that the frequency calculations are in an independent folder.

EXAMPLE OF PATHS:

For Helmholtz: ../../tests/Co_11-20_iPrOH/CH3CHOHCH3
For Gibbs:     ../../tests/Co_0001_iPrOH/CH3CHOHCH3

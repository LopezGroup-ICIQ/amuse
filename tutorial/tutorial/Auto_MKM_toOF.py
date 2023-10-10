import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='First prototype to generate the interphase between pyMKM, AutoProfLib and OpenFOAM.')
parser.add_argument('--Path_to_PreOpenFoam', metavar='Path_to_PreOpenFoam',
                     type=str, help='Path to the PreOpenFoam_readable_reactions.txt file, which contains the basis to generate the OpenFOAM mechanism')
parser.add_argument('--Path_to_Barriers', metavar='Path_to_Barriers',
                     type=str, help='Path to the Barriers.csv file, which contains the prefactor and the barriers (direct and reverse) for each reaction in the mechanism.')
parser.add_argument('--Path_to_Output', metavar='Path_to_Output',
                     type=str, help='Path to the output file, which contains the OpenFOAM mechanism')

args = parser.parse_args()

fname = args.Path_to_PreOpenFoam
fname_2 = args.Path_to_Barriers
outfname = args.Path_to_Output

with open(fname, "r") as inf:
    data = [line.strip().split() for line in inf]

new_data = []
for i in data:
    tmp = []
    for j in i:
        if "(g)" in j:
            tmp.append(j[:-3])
        elif "+" in j:
            tmp.append(j)
        elif j == "->":
            tmp.append("=>")
        elif j=="":
            continue
        elif "_" in j:
            tmp.append(j[:j.index("_")]+"(s)")
        else:
            tmp.append(j+"(s)")
    new_data.append(tmp)

renew_data = []
pop_index = []
for idx1, d in enumerate(new_data):
    count_p = 0
    count_r = 0
    idx_arr = d.index("=>")
    for idx, item in enumerate(d):
        if "H" in item:
            for idx2, l in enumerate(item):
                if l == "H":
                    try:
                        int(item[idx2+1])
                    except:
                        try:
                            int(item[idx2-1])
                        except:
                            if idx < idx_arr:
                                count_r += 1
                            else:
                                count_p += 1
                        else:
                            if idx < idx_arr:
                                count_r += int(item[idx2-1])
                            else:
                                count_p += int(item[idx2-1])
                    else:
                        if idx < idx_arr:
                             count_r += int(item[idx2+1])
                        else:
                            count_p += int(item[idx2+1])
                else:
                    continue
        else:
            continue
    if count_p == count_r:
        renew_data.append(d)
    else:
        pop_index.append(idx1)
        continue

data = renew_data

barriers = pd.read_csv(fname_2, names=["s","a","P", "o", "B", "RB"])
pref = barriers["P"][1:]
beta = [1 for _ in pref]
bar = np.asarray([float(i)*23000 for i in list(barriers["B"][1:])])
barr = np.asarray([float(i)*23000 for i in list(barriers["RB"][1:])])
if not not pop_index:
    pref = np.delete(np.asarray(pref), pop_index)
    beta = np.delete(np.asarray(beta), pop_index)
    bar = np.delete(np.asarray(bar), pop_index)
    barr = np.delete(np.asarray(barr), pop_index)
else:
    pref = pref
    beta = beta
    bar = bar
    barr = barr


reverse = []
for rx in data:
    if len(rx) == 3:
        reverse.append([rx[-1], rx[1], rx[0]])
    elif len(rx) == 5:
        reverse.append([rx[4], rx[3], rx[2], rx[1], rx[0]])
    elif len(rx) == 7:
        reverse.append([rx[6], rx[5], rx[4], rx[3], rx[2], rx[1], rx[0]])
    else:
        print("This do not seem an elementary reaction")
        exit()



with open(outfname, "w") as outf:
    outf.write("MATERIAL MAT-1")
    outf.write("\n")
    outf.write("\n")
    outf.write("SITE/"+data[0][0]+"/"+"    "+"SDEN/2.49E-9/"+"\n")
    species = []
    for rx in data:
        for i in rx:
            if ("(s)" in i and i not in species):
                try:
                    int(i[0])
                except:
                    outf.write(i+" ")
                    species.append(i)
                else:
                    if i[1:] not in species:
                        outf.write(i[1:]+" ")
                        species.append(i[1:])
                    else:
                        continue
            else:
                continue
    for rx in reverse:
        for i in rx:
            if ("(s)" in rx and rx not in species):
                try:
                    int(i[0])
                except:
                    outf.write(i+" ")
                    species.append(i)
                else:
                    if i[1:] not in species:
                        outf.write(i[1:]+" ")
                        species.append(i[1:])
                    else:
                        continue
            else:
                continue
    outf.write("\n")
    outf.write("END"+"\n")
    outf.write("\n")
    outf.write("REACTIONS")
    outf.write("\n")
    for rx, rrx, p, b, br, d in zip(data, reverse, pref, bar, barr, beta):
        rx.append(str(p))
        rx.append(str(d))
        rx.append(str(b))
        rrx.append(str(p))
        rrx.append(str(d))
        rrx.append(str(br))
        for i in rx:
            if (i == data[0][0] and "H2" in rx):
                outf.write("2"+i+"\t")
            else:
                outf.write(i+"\t")
        outf.write("\n")
        for j in rrx:
            if (j == data[0][0] and "H2" in rrx):
                outf.write("2"+j+"\t")
            else:
                outf.write(j+"\t")
        outf.write("\n")
    outf.write("END")

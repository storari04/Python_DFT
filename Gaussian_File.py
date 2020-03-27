from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import numpy as np

calculation_method = 1 #1 Gaussian, #2 GAMESS, #3 Psi4

test = pd.read_csv('./molecules_with_logS.csv')
smiles = test['SMILES'][:5]
mols = [Chem.MolFromSmiles(m) for m in smiles]

#mols = [m for m in Chem.SDMolSupplier("./hoge.sdf")]

optimize_mols = []
for mol in mols:
    mh = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mh)
    AllChem.MMFFOptimizeMolecule(mh)
    optimize_mols.append(mh)

molBlocks = [Chem.MolToMolBlock(m) for m in optimize_mols]
mol_nums = [m.GetNumAtoms() for m in optimize_mols]
coordinates = []
for i, mol in enumerate(molBlocks):
    mol_num = mol_nums[i]
    lst_molBlock = mol.split("\n")
    coordinates_part = lst_molBlock[4 : 4 + mol_num]
    coordinate = [(atoms[31], atoms[0:30]) for atoms in coordinates_part]
    coordinates.append(coordinate)

print(coordinates)

if calculation_method == 1:
#Gaussian output
    for i in range(len(molBlocks)):
        inp = open(str(i) + '.com', 'w')
        inp.write("""%nprocshared=4
"""
'%chk=' + str(i) + '.chk'
"""
%mem=1024MB
#p opt=(maxcycle=100) b3lyp/6-31G 6d gfinput freq=noraman
empiricaldispersion=gd3 iop(1/8=10) scf=(direct,tight) scfcyc=300

"""
'optimization of ' + str(i) + "\n"
"""
0 1
"""
        )

        for atom, coord in coordinates[i]:
            inp.write(atom + coord + '\n')
        inp.write("""




        """)

        inp.close()

elif calculation_method == 2:
#GAMESS output
    for i in range(len(molBlocks)):
        dic = {"H":"  1", "B":"  5", "C":"  6", "N":"  7", "O":"  8", "F":"  9", "Si":"  14", "P":"  15",
            "S":"  16", "Cl":"  17", "Br":"  35", "I":"  53"}
        inp = open(str(i) + '.inp', 'w')
        inp.write("""
$CONTRL COORD=CART UNITS=ANGS $END

$DATA

C1

""")
        for atom, coord in coordinates[i]:
            inp.write(atom + coord + '\n')
        inp.write("""




$END""")

        inp.close()

elif calculation_method == 3:
#psi4 input
    for i in range(len(molBlocks)):
        inp = open(str(i) + '.dat', 'w')
        inp.write("""molecule inputmol {
0 1

"""
        for atom, coord in coordinates[i]:
            inp.write(atom + coord + '\n')
        inp.write(
"""
}

set basis 6-31G*

set reference  uhf

energy('scf')

"""

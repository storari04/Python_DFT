#from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import numpy as np
import psi4

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

for atom, coord in coordinates[i]:
    a = (atom + coord + '\n')

print(a)

a = psi4.geometry(a)

psi4.energy('scf/6-31G')

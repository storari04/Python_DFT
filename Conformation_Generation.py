from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, rdDistGeom
from rdkit.Chem.Draw import IPythonConsole

import py3Dmol

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

number_of_conformation = 10

print('rdkit version: {}'.format(rdBase.rdkitVersion))
# rdkit version: 2019.09.3

#suppl = Chem.SDMolSupplier('./platinum_dataset_2017_01.sdf', removeHs=False)
df = pd.read_csv('molecules_with_logS.csv')

smiles = df.iloc[: , 1]
mols = []
for smile in smiles:
    mol = Chem.MolFromSmiles(smile)
    mols.append(mol)

mol = mols[1201]

#1 Conformer generation
pm = rdDistGeom.ETKDGv2()
m_h = Chem.AddHs(mol)
cids = rdDistGeom.EmbedMultipleConfs(m_h, number_of_conformation, pm)
print(m_h.GetNumConformers())

#2. MMFF optimization and calculation
energy = []
prop = AllChem.MMFFGetMoleculeProperties(m_h)
for cid in cids:
    mmff = AllChem.MMFFGetMoleculeForceField(m_h, prop, confId=cid)
    mmff.Minimize()
    energy.append(mmff.CalcEnergy())

energy = np.array(energy)

# 3. Calculation for RMS
m = Chem.RemoveHs(m_h)
rms_mat = AllChem.GetConformerRMSMatrix(m)
rms_mat.append(0)
rms = np.zeros((number_of_conformation, number_of_conformation))
idx = 0
for i in range(1, number_of_conformation):
    rms[i][:i+1] = rms_mat[idx:i+idx+1]
    idx += i

## 4. 重原子の座標をnumpy配列に格納
def genConfCoord(cid):
    conf = m.GetConformer(cid)
    coord = []
    for atom in m.GetAtoms():
        atom_idx = atom.GetIdx()
        x,y,z = conf.GetAtomPosition(atom_idx)
        coord.extend([x,y,z])
    return np.array(coord)

AllChem.AlignMolConformers(m)
coord_array = np.zeros((len(cids), 3*m.GetNumAtoms()))
for i, cid in enumerate(cids):
    coord_array[i] = genConfCoord(cid)

### クラスタリング用に標準化
scaler = RobustScaler()
scaler.fit(coord_array)
scaled_coord = scaler.transform(coord_array)

del_index = set()
for i in range(number_of_conformation):
    d = pd.DataFrame({'rms': rms[:,i], 'energy': energy})
    d.energy = d.energy - d.energy[i]
    del_index = del_index | set(d[i:].query('rms < 0.05 and -0.5 < energy and energy < 0.5').index)

d = d.drop(del_index)
scaled_coord = scaled_coord[d.index]
print(len(d)) # 737

num_clusters = []
for e in np.arange(2.0,4.1,0.4):
    for s in range(2,20,2):
        db = DBSCAN(eps=e, min_samples=s)
        classes = db.fit_predict(scaled_coord)
        num_clusters.append([e,s,len(set(classes))])

ddf = pd.DataFrame(num_clusters)
ddf.columns = ['eps', 'min_samples', 'num_clusters']
ddf = pd.pivot_table(data=ddf, index=['eps'], columns=['min_samples'], values=['num_clusters'])

fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(ddf, vmin=3, vmax=80, annot=True, cmap='Reds')
ax.set_xlabel('min_samples')
ax.set_xticklabels(range(2,20,2), rotation=0)
ax.set_yticklabels(['{:.1f}'.format(i) for i in np.arange(2,4.1,0.4)])

dbscan = DBSCAN(eps=3.2, min_samples=6)
clusters = dbscan.fit_predict(scaled_coord)
pd.value_counts(clusters, sort=False)

d['group'] = clusters
db_classes = {}
for i in range(-1,17):
    db_classes[i] = d[ d.group == i ].index

view = py3Dmol.view(width=1000, height=600, viewergrid=(3,6), linked=False)
for i, (j,k) in enumerate((m,n) for m in range(3) for n in range(6)):
    for cid in db_classes[i-1]:
        mb = Chem.MolToMolBlock(m, confId=cid)
        view.addModel(mb, 'sdf', viewer=(j,k))
view.setStyle({'line': {}})
view.setBackgroundColor('0xeeeeee')
view.zoomTo()
view.show()

# generation of gaussian file
molBlocks = []
mol_nums = [m_h.GetNumAtoms() for cid in d.index]

for cid in d.index:
    mb = Chem.MolToMolBlock(m_h, confId=cid)
    molBlocks.append(mb)

coordinates = []
for i, mol in enumerate(molBlocks):
    mol_num = mol_nums[i]
    lst_molBlock = mol.split("\n")
    coordinates_part = lst_molBlock[4 : 4 + mol_num]
    coordinate = [(atoms[31], atoms[0:30]) for atoms in coordinates_part]
    coordinates.append(coordinate)

#Generatio of Gaussian input
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

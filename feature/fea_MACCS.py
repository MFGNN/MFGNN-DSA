import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def get_maccs_fingerprint(DBid, smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    fingerprint_str = fingerprint.ToBitString()
    fingerprint_arr = [int(x) for x in fingerprint_str]
    fingerprint_arr.insert(0,DBid)
    return fingerprint_arr


data1 = pd.read_csv("../data/drug_name.csv")
data2 = pd.read_csv("../data/drug_smiles.csv")
DBid = data1["NAME"]
smiles = data2["SMILES"]

Flen = len(get_maccs_fingerprint(DBid[0], smiles[0]))
header = ['Col-{}'.format(i) for i in range(1, Flen)]
header.insert(0,'NAME')

file_name = '../feature/drug_MACCS.csv'
with open(file_name,'w' ,newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

for i in range(len(smiles)):
    fingerprint = get_maccs_fingerprint(DBid[i], smiles[i])
    print(i+1)
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fingerprint)
file.close()

print("MACCS特征维数：",Flen-1)
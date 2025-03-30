import numpy as np
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd


# Protein dictionary
pro_temp = ['A', 'I', 'L', 'V', 'F', 'W', 'Y', 'N', 'C', 'Q', 'M', 'S', 'T', 'D', 'E', 'R', 'H', 'K', 'G', 'P', 'O',
            'U', 'X', 'B', 'Z']
pro_dic = {w: i + 1 for i, w in enumerate(pro_temp)}

def encode_protein(protein_seq, pro_dict):
    max_pro = 1024
    e_pro = [pro_dict[aa] for aa in protein_seq]
    lp = len(e_pro)
    if lp < max_pro:
        v_t = np.pad(e_pro, (0, max_pro - lp), 'constant', constant_values=0)
    else:
        v_t = e_pro[:max_pro]
    return v_t



class Basic_Encoder(torch.utils.data.Dataset):

    def __init__(self, data_id, all_data):
        """Initialization."""
        self.all_data = all_data
        self.data_id = data_id

    def __len__(self):
        """Get size of input data."""
        return len(self.data_id)

    def __getitem__(self, index):
        """Get items from raw data."""
        index = self.data_id[index]
        pro, smile, label = self.all_data.iloc[index].iloc[1], self.all_data.iloc[index].iloc[0], \
            self.all_data.iloc[index].iloc[2]

        mol = Chem.MolFromSmiles(smile)
        compound = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToList()
        protein = encode_protein(pro, pro_dic)
        
        return np.asarray(compound).astype('float32'), np.asarray(protein), label

import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import json
from sklearn.utils import shuffle
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader
import rdkit
import pandas as pd
from rdkit import Chem
def get_atom_features(atom):
    #possible_atom = ['C', 'N', 'O'] #DU代表其他原子
    #atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    #atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    #atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    #atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_type =atom.GetHybridization()
    atom_feats = [
    int(atom.GetIsAromatic()==True),int(atom.GetIsAromatic()==False),
    int(atom_type==Chem.rdchem.HybridizationType.SP),
    int(atom_type==Chem.rdchem.HybridizationType.SP2),
    int(atom_type==Chem.rdchem.HybridizationType.SP3),
    int(atom_type==Chem.rdchem.HybridizationType.SP3D),
    int(atom_type==Chem.rdchem.HybridizationType.UNSPECIFIED)
    ]
    return np.array(atom_feats) 
def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
    bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),
    bond_type == Chem.rdchem.BondType.SINGLE,bond_type==Chem.rdchem.BondType.DOUBLE,
    bond_type == Chem.rdchem.BondType.TRIPLE,bond_type==Chem.rdchem.BondType.AROMATIC,
    bond.GetIsConjugated(),
    bond.IsInRing()
    ]
    return np.array(bond_feats)

class isoo(InMemoryDataset):
    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):

        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'isoo')
        super(isoo, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.afeat,self.bfeat=[],[]
        self.mfeat=[]
        m_floder=self.folder
        self.data, self.slices = torch.load(self.processed_paths[0])
      

    @property
    def raw_file_names(self):
        return ' '

    @property
    def processed_file_names(self):
        return 'isoo_pyg.pt'

    def download(self):
        return
    def ab_feature(self):
        #mol_num=train_size+valid_size+test_size
        mol=Chem.MolFromSmiles('[CH]([CH](F)Cl)(F)Cl')
        mol=Chem.AddHs(mol)          
        bonds = mol.GetBonds()
        atoms = mol.GetAtoms()
            #every mole
        a_feature,b_feature=[],[]
        for atom in atoms:
            a_feature.append(get_atom_features(atom))
        for bond in bonds:
            b_feature.append(get_bond_features(bond))
        return a_feature,b_feature 

    def process(self):
       
        afe,bfe=self.ab_feature()   
        Z=[6,6,1,1,9,9,17,17]
        target = {}
        data_list = []
        err=0
        for i in tqdm(range(1500)):
            k = str(i)
            ri=[]
            zi=[]
            f=open('dataset/isoo/raw/xyz1/'+k+'.xyz')
            y=np.loadtxt('dataset/isoo/raw/MD-vasp-energy.dat')
            f=f.readlines()
            c= [s.replace('\n','') for s in f[2:10]]
            try:
                for k in c:
                    r=k.split(' ')
                    while '' in r:
                        r.remove('')  
                    r=r[1:4]
                    r= [float(x) for x in r]
                    #print(r)
                    ri.append(r)                              
                R_i = torch.tensor(ri,dtype=torch.float32)
                #print(R_i)
                z_i = torch.tensor(Z,dtype=torch.int64)
                y_i = torch.tensor(y[(i+1)*10-1],dtype=torch.float32)
                #y_i = torch.tensor(y[i]*627.5,dtype=torch.float32)    
                a_fea=torch.tensor(afe)
                b_fea=torch.tensor(bfe)
                data = Data(pos=R_i, z=z_i, y=y_i,afe=a_fea,bfe=b_fea)
                data_list.append(data)
            except:
                err=err+1
                print(err)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        self.data=data
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size,test_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        #ids=range(data_size)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:train_size + valid_size+test_size])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = isoo()
    dataset.ab_feature()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)
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

class QM9(InMemoryDataset):

    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):

        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'qm9')
        print(self.folder )
        self.afeat=[]
        self.bfeat=[]
        super(QM9, self).__init__(self.folder, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        m_floder=self.folder
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'qm9_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)
      
    def download(self):
        download_url(self.url, self.raw_dir)
    def ab_feature(self):
        m_floder=self.folder
        f_name=m_floder+'/raw/'+'err.txt'
        err=np.loadtxt('dataset/qm9/raw/err.txt')
        err=err.astype(int)
        valid_data=list(set(range(130831))-set(err))
        for i in tqdm(valid_data):           
            mol_name=m_floder+'/raw/'+'mol/'+str(i)+'.mol'
            mol=Chem.MolFromMolFile(mol_name)
            smi=Chem.MolToSmiles(mol)
            mol=Chem.AddHs(mol)
            bonds = mol.GetBonds()
            atoms = mol.GetAtoms()
            #every mole
            a_feature,b_feature=[],[]
            #every atom or bond
            k=0
            for atom in atoms:
                a_feature.append(get_atom_features(atom))
            for bond in bonds:
                b_feature.append(get_bond_features(bond))
            self.afeat.append(a_feature)
            self.bfeat.append(b_feature)

    def process(self):
        
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))
        self.ab_feature()
        R = data['R']
        Z = data['Z']
        N= data['N']
        err=np.loadtxt('dataset/qm9/raw/err.txt')
        err=err.astype(int)
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z,split)
        target = {}
        for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']:
            target[name] = np.expand_dims(data[name],axis=-1)
        # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)
        afe=self.afeat
        bfe=self.bfeat
        data_list = []
        j=0
        valid_data=list(set(range(130831))-set(err))
        for i in tqdm(valid_data):
            R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']]
            a_fea=torch.tensor(self.afeat[j])
            b_fea=torch.tensor(self.bfeat[j])
            j=j+1
            data = Data(pos=R_i, z=z_i, y=y_i[0], mu=y_i[0], alpha=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4], r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11],afe=a_fea,bfe=b_fea)
            data_list.append(data)
            print(self.processed_paths[0])
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
    dataset = QM9()
    dataset.ab_feature()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    target = 'mu'
    dataset.data.y = dataset.data[target]
    print(dataset.data.y.shape)
    print(dataset.data.y)
    print(dataset.data.mu)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)
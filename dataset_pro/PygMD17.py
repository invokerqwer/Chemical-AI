import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
import rdkit
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
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
class MD17(InMemoryDataset):
    def __init__(self, root = 'dataset/', name = 'benzene_old', transform = None, pre_transform = None, pre_filter = None):

        self.name = name
        self.folder = osp.join(root, self.name)
        self.url = 'http://quantum-machine.org/gdml/data/npz/' + self.name + '_dft.npz'
        self.afeat,self.bfeat=[],[]

        super(MD17, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        #atoms,bonds feature

    @property
    def raw_file_names(self):
        return self.name + '_dft.npz'
    @property
    def processed_file_names(self):
        return self.name + '_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)
        #辅助函数 
    def ab_feature(self,train_size,valid_size,test_size):
        mol_num=train_size+valid_size+test_size
        if(self.name=='benzene_old'):
            smi='c1ccccc1'
            mol=Chem.MolFromSmiles(smi)
        elif(self.name=='ethanol'):
            smi='CCO'
            mol=Chem.MolFromSmiles(smi)
        elif(self.name=='salicylic'):
            mol = Chem.MolFromMolFile(osp.join(self.raw_dir,'sali.mol'))
        elif(self.name=='toluene'):
            smi='Cc1ccccc1'
            mol=Chem.MolFromSmiles(smi)
        elif(self.name=='uracil'):
            mol = Chem.MolFromMolFile(osp.join(self.raw_dir,'ura.mol'))
        elif(self.name=='malonaldehyde'):
            mol = Chem.MolFromMolFile(osp.join(self.raw_dir,'mal.mol'))
        elif(self.name=='naphthalene'):
            smi='c1cccc2ccccc12'
            mol=Chem.MolFromSmiles(smi)
        else:
            #smi='CC(=O)Oc1ccccc1C(=O)O'
            mol = Chem.MolFromMolFile(osp.join(self.raw_dir,'aspi.mol'))
        #mol=Chem.MolFromSmiles(smi)
        mol=Chem.AddHs(mol)
        #m_floder=self.folder
        for i in range(mol_num):          
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
        return a_feature,b_feature
           
    def process(self):
        train_size=1000
        valid_size=1000
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))
        test_size=len(data['E'])-2000
        afe,bfe=self.ab_feature(train_size,valid_size,test_size)
        E = data['E']
        F = data['F']
        R = data['R']
        z = data['z']
        data_list = []
        for i in tqdm(range(len(E))):
            R_i = torch.tensor(R[i],dtype=torch.float32)
            z_i = torch.tensor(z,dtype=torch.int64)
            E_i = torch.tensor(E[i],dtype=torch.float32)
            F_i = torch.tensor(F[i],dtype=torch.float32)
            a_fea=torch.tensor(afe)
            #print(a_fea[0])
            b_fea=torch.tensor(bfe)
            data = Data(pos=R_i, z=z_i, y=E_i, force=F_i,afe=a_fea,bfe=b_fea)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size,test_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        if seed==False:
            ids=range(data_size) 
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:train_size + valid_size+test_size])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict
 
if __name__ == '__main__':
    dataset = MD17(name='aspirin')
    dataset.ab_feature()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    print(dataset.data.y.shape)
    print(dataset.data.force.shape)
    #print(dataset.feat.shape)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=333,test_size=333,seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)
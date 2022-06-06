import torch
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from dataset_pro import MD17
from method.ps_CFFN import PS_CFFN
from method.run import run
from dig.threedgraph.evaluation import ThreeDEvaluator
import torch_cluster
from torch_geometric.data import DataLoader,Data
import rdkit
import random
import numpy as np
name1='aspirin'
type=3
#num_atom=9
num1=1000
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print(device)
dataset_md17 = MD17(root='dataset/', name=name1)
#dataset_md17.process()
#print(dataset_md17.data)
split_idx_md17 = dataset_md17.get_idx_split(len(dataset_md17.data.y), train_size=num1, valid_size=1000,test_size=1000,seed=42)
train_dataset_md17, valid_dataset_md17, test_dataset_md17 = dataset_md17[split_idx_md17['train']], dataset_md17[split_idx_md17['valid']], dataset_md17[split_idx_md17['test']]
idxs = np.random.randint(42, num1, size=50) 
idxs=split_idx_md17['train'][idxs]
if type==0:
    print(dataset_md17.data.y[idxs])
    dataset_md17.data.y[idxs]=1.1*dataset_md17.data.y[idxs]
    print(dataset_md17.data.y[idxs])
if type==1:
    print(dataset_md17.data.force[idxs*num_atom])
    dataset_md17.data.force[idxs*num_atom]=0*dataset_md17.data.force[idxs*num_atom]
    print(dataset_md17.data.force[idxs*num_atom])
if type==2:
    print(dataset_md17.data.pos[idxs*num_atom])
    dataset_md17.data.pos[idxs*num_atom]=0*dataset_md17.data.pos[idxs*num_atom]
    print(dataset_md17.data.pos[idxs*num_atom])
model_md17 =PS_CFFN(energy_and_force=True, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, 
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
        num_spherical=3, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
        )
loss_func_md17 = torch.nn.L1Loss()
evaluation_md17 = ThreeDEvaluator()
run3d_md17 = run()
run3d_md17.run(device, train_dataset_md17, valid_dataset_md17, test_dataset_md17, model_md17, loss_func_md17, evaluation_md17, epochs=1000, batch_size=1, vt_batch_size=32, lr=0.0006, lr_decay_factor=0.5, lr_decay_step_size=200, energy_and_force=True,p=100,name=name1,num=num1)


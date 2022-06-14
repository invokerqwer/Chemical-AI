import torch
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from dataset_pro import isoo
from method.CFFN import CFFN
from method.run import run
import torch_cluster
from torch_geometric.data import DataLoader,Data
from dig.threedgraph.evaluation import ThreeDEvaluator
import rdkit
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print(device)
dataset=isoo(root='dataset/')
#dataset.process()
print(dataset)
split_idx= dataset.get_idx_split(len(dataset.data.y),train_size=1200, valid_size=150,test_size=150,seed=42)
train_dataset, valid_dataset, test_dataset= dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
model =CFFN(energy_and_force=True, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, 
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
        num_spherical=3, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
        )
loss_func= torch.nn.L1Loss()
evaluation= ThreeDEvaluator()
run3d= run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=1000, batch_size=4, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=150, energy_and_force=False)

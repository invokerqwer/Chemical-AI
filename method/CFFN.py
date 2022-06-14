import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.acts import swish
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph,GCNConv,ChebConv,PDNConv
from torch_scatter import scatter
from math import sqrt
import torch.nn.functional as F
from .geometric import xyz_to_dat
from .features import dist_emb, angle_emb, torsion_emb
import numpy as np
try:
    import sympy as sym
except ImportError:
    sym = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb

class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True):
        super(init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        self.lin_v=Linear(135,128)
        self.conv1=PDNConv(hidden_channels,hidden_channels,8,hidden_channels=8,add_self_loops=False)
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)
        
        else: 
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    #def forward(self, x, emb, i, j,afe,b_edge_index,b_edge_attr):
    def forward(self, x, emb, i, j):
        rbf,_,_ = emb
        if self.use_node_features:
            x = self.emb(x)
            #x=torch.cat([x,afe],1) 
            #x=self.lin_v(x)
            #x=self.conv1(x,b_edge_index,b_edge_attr)
            #x=self.conv1(x,b_edge_index,b_edge_attr)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))
        
        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        #rbf=self.dropout(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf
        #sbf=self.dropout(sbf)
        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t
        #t=self.dropout(t)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))
        #print('bond')
        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2
    
class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init,use_ab=True):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init
        self.use_ab=use_ab
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        #self.lin_up1=nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lin_v=nn.Linear(hidden_channels+7,hidden_channels, bias=True)
        self.conv1=PDNConv(hidden_channels,hidden_channels,8,hidden_channels=8,add_self_loops=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)
        self.bn=nn.BatchNorm1d(hidden_channels)
        #self.dropout = nn.Dropout(p=0.5)
        self.bn_a= nn.BatchNorm1d(7)
        self.bn_b= nn.BatchNorm1d(8)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i,afe,b_edge_index,b_edge_attr):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        #b_edge_attr=self.bn_b(b_edge_attr)
        #afe=self.bn_a(afe)
        if self.use_ab==True:
            #print(v.size())
            #print(afe.size())
            v=torch.cat([v,afe],1)   
            v=self.lin_v(v)  
            #v=self.dropout(v)  
            v=self.conv1(v,b_edge_index,b_edge_attr)
            v=self.conv1(v,b_edge_index,b_edge_attr)
            #v=F.relu(v)
            #print(v.shape)
            #dropout用于降低过拟合情况
            #v=self.bn(v)        
            v=F.relu(v)
            #v=self.dropout(v)  
            #print(v.shape)  
        v = self.lin_up(v)
        #print('a_lay')
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)    
        return v
    
class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()
        self.lin_1d=nn.Linear(20,1)
        #self.act=nn.Softmax(dim=1)
    def forward(self, u, v, batch):
        #拼接u       
        u += scatter(v, batch, dim=0)
        #u=self.act(u)
        return u


class CFFN(torch.nn.Module):
    r"""
         The spherical message passing neural network SphereNet from the `"Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size_dist (int, optional): Embedding size used in the basis transformation of distance. (default: :obj:`8`)
            basis_emb_size_angle (int, optional): Embedding size used in the basis transformation of angle. (default: :obj:`8`)
            basis_emb_size_torsion (int, optional): Embedding size used in the basis transformation of torsion. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default: :obj:`swish`)
            output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)
            
    """
    def __init__(
        self, energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        act=swish, output_init='GlorotOrthogonal', use_node_features=True,use_1d=False):
        super(CFFN, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        #self.use_1d=use_1d
        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init,True)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        self.use_1d=use_1d
        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init,True) for _ in range(int(num_layers-3))])
        for _ in range(int(num_layers-3),int(num_layers)):
            self.update_vs.append(update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init,False))
        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip,act) for _ in range(num_layers)])
        self.lin_1d1=Linear(128,64)
        #self.lin_1d=Linear(20,1)
        self.lin_1d2=Linear(64,1)
        self.lin=Linear(2,1,bias=False)
        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])
        #self.b_net=b_Net(30) 
        self.bn_b= nn.BatchNorm1d(8)
        #input = torch.randn(20, 100)
        #output = m(input)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        if self.use_1d==True:
            self.lin_1d1.reset_parameters()
            self.lin_1d2.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        afe,bfe= batch_data.afe, batch_data.bfe
        b_edge_index=batch_data.bfe[:,0:2]
        b_edge_attr=batch_data.bfe
        b=np.array(b_edge_attr.cpu())
        #mfe=batch_data.mfe
        b_edge_attr=torch.cuda.FloatTensor(b)
        c=b_edge_attr[:,[1,0,2,3,4,5,6,7]]#.shape
        b_edge_attr=torch.cat([b_edge_attr,c])
        b=np.array(b_edge_index.cpu())
        b_edge_index=torch.cuda.LongTensor(b)
        #afe=torch.cuda.FloatTensor(afe)
        #b_edge_attr=self.bn_b(b_edge_attr)
        be=b_edge_index.permute(1,0)[[1,0]].permute(1,0)
        b_edge_index=torch.cat([b_edge_index,be]).transpose(0,1)
        num_node=len(batch_data.z)
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes=z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos,edge_index, num_nodes, use_torsion=True)

        emb = self.emb(dist, angle, torsion, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        #e = self.init_e(z, emb, i, j,afe,b_edge_index,b_edge_attr)
        v = self.init_v(e, i,afe,b_edge_index,b_edge_attr)       
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)
        iniu=u
        inie=e
        iniv=v
        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i,afe,b_edge_index,b_edge_attr)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)
        dict={}
        dict['u']=u
        dict['e']=e
        dict['v']=v
        dict['iniu']=iniu
        dict['inie']=inie
        dict['iniv']=iniv
        return u


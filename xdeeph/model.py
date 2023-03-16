import warnings
import os
from math import pi

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn.nn import Gate, Extract
from e3nn.o3 import Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from .from_nequip.cutoffs import PolynomialCutoff
from .from_nequip.radial_basis import BesselBasis
from .from_nequip.tp_utils import tp_path_exists
from .from_schnetpack.acsf import GaussianBasis
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from .e3modules import SphericalBasis, sort_irreps, e3LayerNorm, e3ElementWise, SkipConnection, SeparateWeightTensorProduct, SelfTp
from .tr_modules import TRIrreps, get_tr_gate_nonlin, get_tr_instr_lin, TRSeparateWeightTensorProduct, tp_tr_path_exists, TRFullyConnectedTensorProduct, TRSkipConnection

epsilon = 1e-8


def get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, 
                    act={1: torch.nn.functional.silu, -1: torch.tanh}, 
                    act_gates={1: torch.sigmoid, -1: torch.tanh}
                    ):
    # get gate nonlinearity after tensor product
    # irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
    # irreps_out is desired irreps after gate nonlin
    # notice that nonlin.irreps_out might not be exactly equal to irreps_out
            
    irreps_scalars = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    irreps_gated = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            ir = "0o"
            warnings.warn('Using odd representations as gates')
        else:
            raise ValueError(
                f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}")
    else:
        ir = None
    irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

    gate_nonlin = Gate(
        irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        irreps_gated  # gated tensors
    )
    
    return gate_nonlin
        

class EquiConv(nn.Module):
    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out, norm='', nonlin=True, 
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(EquiConv, self).__init__()
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps([(mul, ir) for mul, ir in irreps_out if tp_path_exists(irreps_in1, irreps_in2, ir)])
        
        self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)
        
        if nonlin:
            self.cfconv = e3ElementWise(self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = e3ElementWise(irreps_tp_out)
            self.irreps_out = irreps_tp_out
        
        # fully connected net to create tensor product weights
        linear_act = nn.SiLU()
        self.fc = nn.Sequential(nn.Linear(fc_len_in, 64),
                                linear_act,
                                nn.Linear(64, 64),
                                linear_act,
                                nn.Linear(64, self.cfconv.len_weight)
                                )

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.cfconv.irreps_in)
            else:
                raise ValueError(f'unknown norm: {norm}')

    def forward(self, fea_in1, fea_in2, fea_weight, batch_edge):
        z = self.tp(fea_in1, fea_in2)

        if self.nonlin is not None:
            z = self.nonlin(z)

        weight = self.fc(fea_weight)
        z = self.cfconv(z, weight)

        if self.norm is not None:
            z = self.norm(z, batch_edge)

        # TODO self-connection here
        return z


class NodeUpdateBlock(nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_out_node, irreps_in_edge,
                 act, act_gates, use_selftp=False, use_sc=True, concat=True, only_ij=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(NodeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_sh = Irreps(irreps_sh)
        irreps_out_node = Irreps(irreps_out_node)
        irreps_in_edge = Irreps(irreps_in_edge)

        if concat:
            irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
            if if_sort_irreps:
                self.sort = sort_irreps(irreps_in1)
                irreps_in1 = self.sort.irreps_out
        else:
            irreps_in1 = irreps_in_node
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_node, irreps_out=irreps_in_node, biases=True)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_node, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_node
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out
        
        self.sc = None
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_node, f'{num_species}x0e', self.conv.irreps_out)
            
        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_node, self.irreps_out)
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.irreps_in_node = irreps_in_node
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps

    def forward(self, node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch, selfloop_edge, edge_length):
            
        node_fea_old = node_fea
        
        if self.use_sc:
            node_self_connection = self.sc(node_fea, node_one_hot)

        node_fea = self.lin_pre(node_fea)

        index_i = edge_index[0]
        index_j = edge_index[1]
        if self.concat:
            fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
            if self.if_sort_irreps:
                fea_in = self.sort(fea_in)
            edge_update = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        else:
            edge_update = self.conv(node_fea[index_j], edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        # sigma = 3
        # n = 2
        # edge_update = edge_update * torch.exp(- edge_length ** n / sigma ** n / 2).view(-1, 1)
        
        node_fea = scatter(edge_update, index_i, dim=0, dim_size=node_fea.shape[0], reduce='add')
        if self.only_ij:
            node_fea = node_fea + scatter(edge_update[~selfloop_edge], index_j[~selfloop_edge], dim=0, dim_size=node_fea.shape[0], reduce='add')
            
        node_fea = self.lin_post(node_fea)

        if self.use_sc:
            node_fea = node_fea + node_self_connection
            
        if self.nonlin is not None:
            node_fea = self.nonlin(node_fea)
            
        if self.norm is not None:
            node_fea = self.norm(node_fea, batch)
            
        node_fea = self.skip_connect(node_fea_old, node_fea)
        
        if self.self_tp is not None:
            node_fea = self.self_tp(node_fea)
        
        return node_fea


class EdgeUpdateBlock(nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_selftp=False, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, biases=True)
        
        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.conv.irreps_out)

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out) # ! consider init_edge
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)
            
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge

    def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
        
        if not self.init_edge:
            edge_fea_old = edge_fea
            if self.use_sc:
                edge_self_connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = self.lin_pre(edge_fea)
            
        index_i = edge_index[0]
        index_j = edge_index[1]
        fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        if self.if_sort_irreps:
            fea_in = self.sort(fea_in)
        edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection
            
        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])
        
        if not self.init_edge:
            edge_fea = self.skip_connect(edge_fea_old, edge_fea)
        
        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea


class AggrEdgeSh(nn.Module):
    def __init__(self, irreps_sh, irreps_sh_neighbor):
        super(AggrEdgeSh, self).__init__()
        instr_lin = get_tr_instr_lin(irreps_sh, irreps_sh_neighbor)
        self.mix_neighbor = Linear(
            irreps_sh.irreps, irreps_sh_neighbor.irreps, instructions=instr_lin,
            internal_weights=False, shared_weights=False, biases=False
        )
        self.weight_numel = self.mix_neighbor.weight_numel

    def forward(self, edge_sh, weight, edge_index):
        index_i = edge_index[0]
        index_j = edge_index[1]
        len_weight = weight.shape[-1]
        assert len_weight % 2 == 0
        edge_sh_neighbor_i = self.mix_neighbor(edge_sh, weight[:, :len_weight//2])
        edge_sh_neighbor_j = self.mix_neighbor(edge_sh, weight[:, len_weight//2:])
        node_sh_neighbor_i = scatter(edge_sh_neighbor_i, index_i, dim=0, reduce='add')
        node_sh_neighbor_j = scatter(edge_sh_neighbor_j, index_i, dim=0, reduce='add')

        return torch.cat([node_sh_neighbor_i[index_i], node_sh_neighbor_j[index_j]], dim=-1)

class LocalEquiConv(nn.Module):
    def __init__(self, irreps_sh, irreps_sh_neighbor, irreps_in, irreps_out, norm='', nonlin=True,
                 act={1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates={1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(LocalEquiConv, self).__init__()

        assert isinstance(irreps_sh, TRIrreps)
        assert isinstance(irreps_sh_neighbor, TRIrreps)
        assert isinstance(irreps_in, TRIrreps)
        assert isinstance(irreps_out, TRIrreps)

        # aggregation edge_sh of neighbor
        self.aggr_edge_sh = AggrEdgeSh(irreps_sh, irreps_sh_neighbor)

        # get the scalar irreps from the edge_fea
        irreps_scalar = []
        instructions_scalar = []
        for index_ir, (mul_ir, tr) in enumerate(irreps_in):
            if mul_ir.ir.is_scalar() and tr == 1:
                irreps_scalar.append([mul_ir])
                instructions_scalar.append((index_ir,))
        self.extract_scalar = Extract(irreps_in.irreps, irreps_scalar, instructions_scalar)
        scalar_len = sum(mul_ir[0].mul for mul_ir in irreps_scalar)

        # fully connected net to create weights
        linear_act = nn.SiLU()
        self.mlp = nn.Sequential(nn.Linear(scalar_len, 64),
                                 linear_act,
                                 nn.Linear(64, 64),
                                 linear_act,
                                 nn.Linear(64, self.aggr_edge_sh.weight_numel * 2)
                                 )

        self.nonlin = None
        if nonlin:
            self.nonlin, nonlin_irreps_in, nonlin_irreps_out = get_tr_gate_nonlin(
                irreps_sh_neighbor + irreps_sh_neighbor, irreps_in, irreps_out,
                act, act_gates)
            irreps_tp_out = nonlin_irreps_in
        else:
            irreps_tp_out = TRIrreps(
                Irreps([(mul, ir) for (mul, ir), tr in irreps_out if tp_tr_path_exists(irreps_sh_neighbor + irreps_sh_neighbor, irreps_in, ir, tr)]),
                [tr for (mul, ir), tr in irreps_out if tp_tr_path_exists(irreps_sh_neighbor + irreps_sh_neighbor, irreps_in, ir, tr)]
            )

        self.tp = TRSeparateWeightTensorProduct(irreps_sh_neighbor + irreps_sh_neighbor, irreps_in, irreps_tp_out)

        if nonlin:
            self.irreps_out = nonlin_irreps_out
        else:
            self.irreps_out = irreps_tp_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out.irreps, self.irreps_out.tr_list)
            else:
                raise ValueError(f'unknown norm: {norm}')

    def forward(self, edge_sh, edge_fea, edge_index, batch_edge):
        edge_fea_scalar = self.extract_scalar(edge_fea)
        edge_fea_scalar = torch.cat(edge_fea_scalar, dim=-1)
        weight = self.mlp(edge_fea_scalar)
        edge_sh_neighbor = self.aggr_edge_sh(edge_sh, weight, edge_index)

        z = self.tp(edge_sh_neighbor, edge_fea)

        if self.nonlin is not None:
            z = self.nonlin(z)

        if self.norm is not None:
            z = self.norm(z, batch_edge)

        return z


class LocalEdgeUpdateBlock(nn.Module):
    def __init__(self, num_species, irreps_sh, irreps_sh_neighbor, irreps_in, irreps_out,
                 act, act_gates, use_selftp=False, use_sc=True,
                 nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(LocalEdgeUpdateBlock, self).__init__()
        assert isinstance(irreps_in, TRIrreps)
        assert isinstance(irreps_out, TRIrreps)

        if if_sort_irreps:
            raise NotImplementedError
            self.sort = sort_irreps(irreps_in)
            irreps_in = self.sort.irreps_out

        instr_lin = get_tr_instr_lin(irreps_in, irreps_in)
        self.lin_pre = Linear(irreps_in=irreps_in.irreps, irreps_out=irreps_in.irreps,
                              biases=[mul_ir.ir.is_scalar() and tr == 1 for mul_ir, tr in irreps_in],
                              instructions=instr_lin)

        self.nonlin = None
        self.lin_post = None
        if nonlin:
            raise NotImplementedError
            # self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
            # irreps_conv_out = self.nonlin.irreps_in
            # conv_nonlin = False
        else:
            irreps_conv_out = irreps_out
            conv_nonlin = True

        self.conv = LocalEquiConv(irreps_sh=irreps_sh, irreps_sh_neighbor=irreps_sh_neighbor,
                                  irreps_in=irreps_in, irreps_out=irreps_conv_out,
                                  nonlin=conv_nonlin, act=act, act_gates=act_gates)

        instr_lin = get_tr_instr_lin(self.conv.irreps_out, self.conv.irreps_out)
        self.lin_post = Linear(
            irreps_in=self.conv.irreps_out.irreps, irreps_out=self.conv.irreps_out.irreps,
            biases=[mul_ir.ir.is_scalar() and tr == 1 for mul_ir, tr in self.conv.irreps_out],
            instructions=instr_lin
        )

        if use_sc:
            irreps_spec = Irreps(f'{num_species ** 2}x0e')
            irreps_spec = TRIrreps(irreps_spec, [1] * len(irreps_spec))
            self.sc = TRFullyConnectedTensorProduct(irreps_in, irreps_spec, self.conv.irreps_out)

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out.irreps, self.irreps_out.tr_list)
            else:
                raise ValueError(f'unknown norm: {norm}')

        self.skip_connect = TRSkipConnection(irreps_in, self.irreps_out)

        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.use_sc = use_sc
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in = irreps_in

    def forward(self, edge_one_hot, edge_sh, edge_fea, edge_index, batch):

        edge_fea_old = edge_fea
        if self.use_sc:
            edge_self_connection = self.sc(edge_fea, edge_one_hot)
        edge_fea = self.lin_pre(edge_fea)

        if self.if_sort_irreps:
            edge_fea = self.sort(edge_fea)
        edge_fea = self.conv(edge_sh, edge_fea, edge_index, batch[edge_index[0]])

        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection

        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])

        edge_fea = self.skip_connect(edge_fea_old, edge_fea)

        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea

class Net(nn.Module):
    def __init__(self, num_species, irreps_embed_node, irreps_edge_init, irreps_sh, irreps_mid_node, 
                 irreps_post_node, irreps_out_node, irreps_mid_edge, irreps_post_edge, irreps_out_edge, tr_out_edge,
                 num_block, r_max, use_sc=True, no_parity=False, use_sbf=True, selftp=False, edge_upd=True,
                 only_ij=False, num_basis=128,
                 act={1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates={1: torch.sigmoid, -1: torch.tanh},
                 if_sort_irreps=False):
        
        if no_parity:
            for irreps in (irreps_embed_node, irreps_edge_init, irreps_sh, irreps_mid_node, 
                    irreps_post_node, irreps_out_node,irreps_mid_edge, irreps_post_edge, irreps_out_edge,):
                for _, ir in Irreps(irreps):
                    assert ir.p == 1, 'Ignoring parity but requiring representations with odd parity in net'
        
        super(Net, self).__init__()
        self.num_species = num_species
        self.only_ij = only_ij
        
        irreps_embed_node = Irreps(irreps_embed_node)
        irreps_mid_edge = Irreps(irreps_mid_edge)
        irreps_post_edge = Irreps(irreps_post_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        assert irreps_embed_node == Irreps(f'{irreps_embed_node.dim}x0e')
        self.embedding = Linear(irreps_in=f"{num_species}x0e", irreps_out=irreps_embed_node)

        # edge embedding for tensor product weight
        # self.basis = BesselBasis(r_max, num_basis=num_basis, trainable=False)
        # self.cutoff = PolynomialCutoff(r_max, p=6)
        self.basis = GaussianBasis(start=0.0, stop=r_max, n_gaussians=num_basis, trainable=False)
        
        # distance expansion to initialize edge feature
        irreps_edge_init = Irreps(irreps_edge_init)
        assert irreps_edge_init == Irreps(f'{irreps_edge_init.dim}x0e')
        self.distance_expansion = GaussianBasis(
            start=0.0, stop=6.0, n_gaussians=irreps_edge_init.dim, trainable=False
        )

        if use_sbf:
            self.sh = SphericalBasis(irreps_sh, r_max)
        else:
            self.sh = SphericalHarmonics(
                irreps_out=irreps_sh,
                normalize=True,
                normalization='component',
            )
        self.use_sbf = use_sbf

        self.add_mag_initNode_edgeSh = False
        self.loc_mag = True
        self.tr_equiv = True
        mag_sh_lmax = 4
        if self.loc_mag:
            if self.tr_equiv:
                irreps_sh_neighbor = TRIrreps(
                    Irreps([(8, (0, 1)), (8, (1, -1)), (8, (1, 1)), (8, (2, 1)), (8, (3, -1)), (8, (3, 1)), (8, (4, 1))]),
                    # Irreps([(8, (0, 1)), (8, (1, -1)), (8, (1, 1)), (8, (2, 1)), (8, (3, -1)), (8, (3, 1)), (8, (4, 1)), (8, (5, -1)), (8, (5, 1))]),
                    [1, 1, -1, 1, 1, -1, 1]
                    # [1, 1, -1, 1, 1, -1, 1, 1, -1]
                )
            else:
                irreps_sh_neighbor = TRIrreps(
                    Irreps([(8, (0, 1)), (8, (1, 1)), (8, (2, 1)), (8, (3, 1)), (8, (4, 1))]),
                    # Irreps([(8, (0, 1)), (8, (1, 1)), (8, (2, 1)), (8, (3, 1)), (8, (4, 1)), (8, (5, 1))]),
                    [1, 1, 1, 1, 1]
                    # [1, 1, 1, 1, 1, 1]
                )
            self.irreps_sh_neighbor = irreps_sh_neighbor

        if self.tr_equiv:
            assert self.add_mag_initNode_edgeSh == False and self.loc_mag == True
        if self.add_mag_initNode_edgeSh or self.loc_mag:
            num_mag_mod = 16
            irreps_mag_mod = Irreps(f'{num_mag_mod}x0e')
            self.magmom_mod = GaussianBasis(
                start=3.25, stop=3.65, # Cr
                # start=1.3, stop=1.5, # Ni
                # start=4.8, stop=5.0, # Mn
                n_gaussians=num_mag_mod, trainable=True
            )
            irreps_mag_sh = Irreps([(1, (i, 1)) for i in range(mag_sh_lmax + 1)])
            # irreps_mag_sh = Irreps([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))])
            self.magmom_sh =  SphericalHarmonics(
                irreps_out=irreps_mag_sh,
                normalize=True,
                normalization='component',
            )

        if no_parity:
            irreps_sh = Irreps([(mul, (ir.l, 1)) for mul, ir in Irreps(irreps_sh)])
        if self.add_mag_initNode_edgeSh:
            self.irreps_sh = Irreps(irreps_sh) + irreps_mag_sh + irreps_mag_sh # i j 两个节点的磁矩的方位 cat 在 rij 的右边
        else:
            self.irreps_sh = irreps_sh

        # self.edge_update_block_init = EdgeUpdateBlock(num_basis, irreps_sh, self.embedding.irreps_out, None, irreps_mid_edge, act, act_gates, False, init_edge=True)
        if self.add_mag_initNode_edgeSh:
            irreps_node_prev = self.embedding.irreps_out + irreps_mag_mod + irreps_mag_sh
        else:
            irreps_node_prev = self.embedding.irreps_out
        irreps_edge_prev = irreps_edge_init

        self.node_update_blocks = nn.ModuleList([])
        self.edge_update_blocks = nn.ModuleList([])
        for index_block in range(num_block):
            if index_block == num_block - 1:
                node_update_block = NodeUpdateBlock(num_species, num_basis, self.irreps_sh, irreps_node_prev, irreps_post_node, irreps_edge_prev, act, act_gates, use_selftp=selftp, use_sc=use_sc, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                if self.loc_mag:
                    edge_update_block = EdgeUpdateBlock(num_species, num_basis, self.irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_mid_edge, act, act_gates, use_selftp=False, use_sc=use_sc, if_sort_irreps=if_sort_irreps)
                else:
                    edge_update_block = EdgeUpdateBlock(num_species, num_basis, self.irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_post_edge, act, act_gates, use_selftp=selftp, use_sc=use_sc, if_sort_irreps=if_sort_irreps)
            else:
                node_update_block = NodeUpdateBlock(num_species, num_basis, self.irreps_sh, irreps_node_prev, irreps_mid_node, irreps_edge_prev, act, act_gates, use_selftp=False, use_sc=use_sc, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                edge_update_block = None
                if edge_upd:
                    edge_update_block = EdgeUpdateBlock(num_species, num_basis, self.irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_mid_edge, act, act_gates, use_selftp=False, use_sc=use_sc, if_sort_irreps=if_sort_irreps)
            irreps_node_prev = node_update_block.irreps_out
            if edge_update_block is not None:
                irreps_edge_prev = edge_update_block.irreps_out
            self.node_update_blocks.append(node_update_block)
            self.edge_update_blocks.append(edge_update_block)

        if self.loc_mag:
            self.loc_mag_blocks = nn.ModuleList([])
            self.num_loc_block = 3
            irreps_edge_prev = edge_update_block.irreps_out
            self.lin_edge_prev = Linear(
                irreps_in=irreps_mag_mod + irreps_mag_mod + irreps_edge_prev,
                irreps_out=irreps_edge_prev, biases=True
            )
            irreps_edge_prev = TRIrreps(self.lin_edge_prev.irreps_out, [1] * len(self.lin_edge_prev.irreps_out))
            if self.tr_equiv:
                tr_list_sh = [1] * len(Irreps(irreps_sh)) + [(-1) ** i for i in range(mag_sh_lmax + 1)] * 2
            else:
                tr_list_sh = [1] * len(Irreps(irreps_sh)) * 3

            irreps_sh_loc_mag = TRIrreps(Irreps(irreps_sh) + irreps_mag_sh + irreps_mag_sh, tr_list_sh)
            self.irreps_sh_loc_mag = irreps_sh_loc_mag
            for index_block in range(self.num_loc_block):
                if index_block == self.num_loc_block - 1:
                    if self.tr_equiv:
                        # time reversal even + time reversal odd
                        # irreps_out_local_edge = TRIrreps(irreps_post_edge + irreps_post_edge,
                                                         # [1] * len(irreps_post_edge) + [-1] * len(irreps_post_edge))
                        irreps_post_edge_half = []
                        for (mul, ir) in irreps_post_edge:
                            assert mul % 2 == 0
                            irreps_post_edge_half.append([mul // 2, ir])
                        irreps_post_edge_half = Irreps(irreps_post_edge_half)
                        irreps_out_local_edge = TRIrreps(
                            irreps_post_edge_half + irreps_post_edge_half,
                            [1] * len(irreps_post_edge_half) + [-1] * len(irreps_post_edge_half)
                        )
                    else:
                        irreps_out_local_edge = TRIrreps(irreps_post_edge, [1] * len(irreps_post_edge))
                else:
                    if self.tr_equiv:
                        # time reversal even + time reversal odd
                        irreps_out_local_edge = TRIrreps(irreps_mid_edge + irreps_mid_edge,
                                                         [1] * len(irreps_mid_edge) + [-1] * len(irreps_mid_edge))
                    else:
                        irreps_out_local_edge = TRIrreps(irreps_mid_edge, [1] * len(irreps_mid_edge))
                loc_mag_block = LocalEdgeUpdateBlock(
                    num_species=num_species,
                    irreps_sh=irreps_sh_loc_mag,
                    irreps_sh_neighbor=irreps_sh_neighbor,
                    irreps_in=irreps_edge_prev,
                    irreps_out=irreps_out_local_edge,
                    act=act,
                    act_gates=act_gates,
                    use_selftp=selftp,
                    use_sc=use_sc,
                    if_sort_irreps=if_sort_irreps
                )
                irreps_edge_prev = loc_mag_block.irreps_out
                self.loc_mag_blocks.append(loc_mag_block)

        if self.loc_mag:
            if self.tr_equiv:
                irreps_out_edge = TRIrreps(irreps_out_edge, tr_out_edge)
            else:
                irreps_out_edge = TRIrreps(irreps_out_edge, len(irreps_out_edge) * [1])
        else:
            irreps_out_edge = Irreps(irreps_out_edge)
            for _, ir in irreps_out_edge:
                assert ir in irreps_edge_prev, f'required ir {ir} in irreps_out_edge cannot be produced by convolution in the last edge update block ({edge_update_block.irreps_in_edge} -> {edge_update_block.irreps_out})'

        self.irreps_out_node = irreps_out_node
        self.irreps_out_edge = irreps_out_edge
        self.lin_node = Linear(irreps_in=irreps_node_prev, irreps_out=irreps_out_node, biases=True)
        if self.loc_mag:
            instr_lin = get_tr_instr_lin(irreps_edge_prev, irreps_out_edge)
            self.lin_edge = Linear(
                irreps_in=irreps_edge_prev.irreps, irreps_out=irreps_out_edge.irreps,
                biases=[mul_ir.ir.is_scalar() and tr == 1 for mul_ir, tr in irreps_out_edge],
                instructions=instr_lin
            )
        else:
            self.lin_edge = Linear(irreps_in=irreps_edge_prev, irreps_out=irreps_out_edge, biases=True)

    def forward(self, data):
        node_one_hot = F.one_hot(data.x, num_classes=self.num_species).type(torch.get_default_dtype())
        edge_one_hot = F.one_hot(self.num_species * data.x[data.edge_index[0]] + data.x[data.edge_index[1]],
                                 num_classes=self.num_species**2).type(torch.get_default_dtype()) # ! might not be good if dataset has many elements
        
        node_fea = self.embedding(node_one_hot)

        edge_length = data['edge_attr'][:, 0]
        edge_vec = data["edge_attr"][:, [2, 3, 1]] # (y, z, x) order

        if self.use_sbf:
            edge_sh = self.sh(edge_length, edge_vec)
        else:
            edge_sh = self.sh(edge_vec).type(torch.get_default_dtype())
        # edge_length_embedded = (self.basis(data["edge_attr"][:, 0] + epsilon) * self.cutoff(data["edge_attr"][:, 0])[:, None]).type(torch.get_default_dtype())
        edge_length_embedded = self.basis(edge_length)

        if self.add_mag_initNode_edgeSh or self.loc_mag:
            # (y, z, x) order
            magmom_vec = torch.stack([
                torch.sin(data.magmom[:, 2] * pi / 180) * torch.sin(data.magmom[:, 3] * pi / 180),
                torch.cos(data.magmom[:, 2] * pi / 180),
                torch.sin(data.magmom[:, 2] * pi / 180) * torch.cos(data.magmom[:, 3] * pi / 180)
            ], dim=-1)

            node_magmom_sh = self.magmom_sh(magmom_vec) * data.magmom[:, 0:1]
            node_magmom_mod = self.magmom_mod(data.magmom[:, 1]) * data.magmom[:, 0:1]

            index_i = data["edge_index"][0]
            index_j = data["edge_index"][1]

        if self.add_mag_initNode_edgeSh:
            node_magmom = torch.cat([node_magmom_mod, node_magmom_sh], dim=-1)
            node_fea = torch.cat([node_fea, node_magmom], dim=-1)

            # i j 两个节点的磁矩的方位 cat 在 rij 的右边
            edge_sh = torch.cat([edge_sh, node_magmom_sh[index_i], node_magmom_sh[index_j]], dim=-1)
        
        selfloop_edge = None
        if self.only_ij:
            selfloop_edge = torch.abs(data["edge_attr"][:, 0]) < 1e-7

        edge_fea = self.distance_expansion(edge_length).type(torch.get_default_dtype())
        for node_update_block, edge_update_block in zip(self.node_update_blocks, self.edge_update_blocks):
            node_fea = node_update_block(node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, data["edge_index"], data.batch, selfloop_edge, edge_length)
            if edge_update_block is not None:
                edge_fea = edge_update_block(node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, data["edge_index"], data.batch)

        if self.loc_mag:
            if not self.add_mag_initNode_edgeSh:
                edge_sh = torch.cat([edge_sh, node_magmom_sh[index_i], node_magmom_sh[index_j]], dim=-1)
            edge_fea = self.lin_edge_prev(
                torch.cat([node_magmom_mod[index_i], node_magmom_mod[index_j], edge_fea], dim=-1)
            )
            for loc_mag_block in self.loc_mag_blocks:
                edge_fea = loc_mag_block(edge_one_hot, edge_sh, edge_fea, data["edge_index"], data.batch)

        node_fea = self.lin_node(node_fea)
        edge_fea = self.lin_edge(edge_fea)
        return node_fea, edge_fea

    def __repr__(self):
        info = '===== xDeepH model structure: ====='
        if self.use_sbf:
            info += f'\nusing spherical bessel basis: {self.irreps_sh}'
        else:
            info += f'\nusing spherical harmonics: {self.irreps_sh}'
        for index, (nupd, eupd) in enumerate(zip(self.node_update_blocks, self.edge_update_blocks)):
            info += f'\n=== layer {index} ==='
            info += f'\nnode update: ({nupd.irreps_in_node} -> {nupd.irreps_out})'
            if eupd is not None:
                info += f'\nedge update: ({eupd.irreps_in_edge} -> {eupd.irreps_out})'
        if self.loc_mag:
            info += f'\n=== previous local magnetic layer ==='
            info += f'\nedge update: ({self.lin_edge_prev.irreps_in} -> {self.lin_edge_prev.irreps_out})'
            info += f'\nspherical harmonics for position vector and magnetic moment: {self.irreps_sh_loc_mag}'
            info += f'\nneighbor aggregation: {self.irreps_sh_neighbor}'
            for index, loc_mag_block in enumerate(self.loc_mag_blocks):
                info += f'\n=== local magnetic layer {index} ==='
                info += f'\nlocal magnetic: ({loc_mag_block.irreps_in} -> {loc_mag_block.irreps_out})'
        info += '\n=== output ==='
        info += f'\noutput node: ({self.irreps_out_node})'
        info += f'\noutput edge: ({self.irreps_out_edge})'
        
        return info
    
    def analyze_tp(self, path):
        os.makedirs(path, exist_ok=True)
        for index, (nupd, eupd) in enumerate(zip(self.node_update_blocks, self.edge_update_blocks)):
            fig, ax = nupd.conv.tp.visualize()
            fig.savefig(os.path.join(path, f'node_update_{index}.png'))
            fig.clf()
            fig, ax = eupd.conv.tp.visualize()
            fig.savefig(os.path.join(path, f'edge_update_{index}.png'))
            fig.clf()
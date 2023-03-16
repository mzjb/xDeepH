import torch
from torch import nn
from e3nn import o3
from e3nn.nn import Extract, Activation
from e3nn.util.jit import compile_mode


@compile_mode('script')
class _Sortcut(torch.nn.Module):
    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(o3.Irreps(irreps) for irreps in irreps_outs)
        irreps_in = sum(self.irreps_outs, o3.Irreps([]))
        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions += [tuple(range(i, i + len(irreps_out)))]
            i += len(irreps_out)
        assert len(irreps_in) == i, (len(irreps_in), i)

        instructions = [tuple(x) for x in instructions]

        self.cut = Extract(irreps_in, self.irreps_outs, instructions)
        self.irreps_in = irreps_in

    def forward(self, x):
        return self.cut(x)


@compile_mode('script')
class TRGate(torch.nn.Module):
    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated):
        super().__init__()
        assert isinstance(irreps_scalars, TRIrreps)
        assert isinstance(irreps_gates, TRIrreps)
        assert isinstance(irreps_gated, TRIrreps)

        if len(irreps_gates.irreps) > 0 and irreps_gates.irreps.lmax > 0:
            raise ValueError(f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}")
        if len(irreps_scalars.irreps) > 0 and irreps_scalars.irreps.lmax > 0:
            raise ValueError(f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}")
        if irreps_gates.irreps.num_irreps != irreps_gated.irreps.num_irreps:
            raise ValueError(f"There are {irreps_gated.irreps.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.irreps.num_irreps}) of gate scalars in irreps_gates")

        self.sc = _Sortcut(irreps_scalars.irreps, irreps_gates.irreps, irreps_gated.irreps)

        self.irreps_scalars, self.irreps_gates, self.irreps_gated = self.sc.irreps_outs
        self._irreps_in = self.sc.irreps_in

        self.act_scalars = Activation(irreps_scalars.irreps, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates.irreps, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated.irreps, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        with torch.autograd.profiler.record_function('TRGate'):
            scalars, gates, gated = self.sc(features)

            scalars = self.act_scalars(scalars)
            if gates.shape[-1]:
                gates = self.act_gates(gates)
                gated = self.mul(gated, gates)
                features = torch.cat([scalars, gated], dim=-1)
            else:
                features = scalars
            return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out


def tp_tr_path_exists(irreps_in1, irreps_in2, ir_out, tr_out):
    assert isinstance(irreps_in1, TRIrreps)
    assert isinstance(irreps_in2, TRIrreps)
    assert isinstance(ir_out, o3.Irrep)

    for (_, ir1), tr1 in irreps_in1:
        for (_, ir2), tr2 in irreps_in2:
            if ir_out in ir1 * ir2 and tr_out == tr1 * tr2:
                return True
    return False


def get_tr_gate_nonlin(tr_irreps_in1, tr_irreps_in2, tr_irreps_out,
                       act={1: torch.nn.functional.silu, -1: torch.tanh},
                       act_gates={1: torch.sigmoid, -1: torch.tanh}
                       ):
    # get gate nonlinearity after tensor product

    irreps_scalars_list = [
        ((mul, ir), tr)
        for (mul, ir), tr in tr_irreps_out
        if ir.l == 0 and tp_tr_path_exists(tr_irreps_in1, tr_irreps_in2, ir, tr)
    ]
    irreps_scalars = TRIrreps(o3.Irreps([mul_ir for mul_ir, tr in irreps_scalars_list]),
                              [tr for mul_ir, tr in irreps_scalars_list])

    irreps_gated_list = [
        ((mul, ir), tr)
        for (mul, ir), tr in tr_irreps_out
        if ir.l > 0 and tp_tr_path_exists(tr_irreps_in1, tr_irreps_in2, ir, tr)
    ]
    irreps_gated = TRIrreps(o3.Irreps([mul_ir for mul_ir, tr in irreps_gated_list]),
                            [tr for mul_ir, tr in irreps_gated_list])

    ir_gates = o3.Irrep('0e')
    tr_gates = 1
    if irreps_gated.irreps.dim > 0:
        if not tp_tr_path_exists(tr_irreps_in1, tr_irreps_in2, ir_gates, tr_gates):
            raise ValueError(
                f"irreps_in1={tr_irreps_in1} times irreps_in2={tr_irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}")
    else:
        ir_gates = None
    irreps_gates = o3.Irreps([(mul, ir_gates) for (mul, _), _ in irreps_gated]).simplify()
    irreps_gates = TRIrreps(irreps_gates, [tr_gates] * len(irreps_gates))

    act_tr = {(1, 1): torch.nn.functional.silu, (-1, 1): torch.tanh, (1, -1): torch.tanh, (-1, -1): torch.tanh}
    act_gates_tr = {(1, 1): torch.sigmoid}

    gate_nonlin = TRGate(
        irreps_scalars, [act_tr[(ir.p, tr)] for (_, ir), tr in irreps_scalars],  # scalar
        irreps_gates, [act_gates_tr[(ir.p, tr)] for (_, ir), tr in irreps_gates],  # gates (scalars)
        irreps_gated  # gated tensors
    )

    return gate_nonlin, (irreps_scalars + irreps_gates + irreps_gated), (irreps_scalars + irreps_gated)


class TRIrreps:
    def __init__(self, irreps, tr_list):
        self.irreps = o3.Irreps(irreps)
        self.tr_list = tr_list
        assert len(self.tr_list) == len(self.irreps), f"len(tr_list)={len(self.tr_list)} != len(irreps)={len(self.irreps)}"
        for tr in tr_list:
            assert tr in [1, -1]

    def __add__(self, tr_irreps):
        return TRIrreps(self.irreps + tr_irreps.irreps, self.tr_list + tr_irreps.tr_list)

    def __getitem__(self, i):
        return self.irreps[i], self.tr_list[i]

    def __repr__(self):
        tr_symbol_dict = {1: 'E', -1: 'O'}
        return "+".join(f"{mul_ir}{tr_symbol_dict[tr]}" for mul_ir, tr in zip(self.irreps, self.tr_list))


def get_tr_instr_lin(tr_irreps_in, tr_irreps_out, allow_zero=False):
    instructions = [
        (i_in, i_out)
        for i_in, ((mul_in, ir_in), tr_in) in enumerate(tr_irreps_in)
        for i_out, ((mul_out, ir_out), tr_out) in enumerate(tr_irreps_out)
        if ir_in == ir_out and tr_in == tr_out
    ]

    out_set = set([i_out for i_in, i_out in instructions])
    if not allow_zero:
        assert len(out_set) == len(tr_irreps_out.irreps), f'Not all output irreps are used, instructions: {instructions}, tr_irreps_out: {tr_irreps_out}'
    return instructions


@compile_mode('script')
class TRSeparateWeightTensorProduct(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        '''z_i = W'_{ij}x_j W''_{ik}y_k'''
        super().__init__()

        assert not kwargs.pop('internal_weights', False)  # internal weights must be True
        assert kwargs.pop('shared_weights', True)  # shared weights must be false

        assert isinstance(irreps_in1, TRIrreps)
        assert isinstance(irreps_in2, TRIrreps)
        assert isinstance(irreps_out, TRIrreps)

        instr_tp = []
        weights1, weights2 = [], []
        for i1, ((mul1, ir1), tr1) in enumerate(irreps_in1):
            for i2, ((mul2, ir2), tr2) in enumerate(irreps_in2):
                for i_out, ((mul_out, ir3), tr3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2 and tr3 == tr1 * tr2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        instr_tp.append((i1, i2, i_out, 'uvw', True, 1.0))

        out_set = set([i_out for _, _, i_out, _, _, _ in instr_tp])
        assert len(out_set) == len(irreps_out.irreps), f'Not all output irreps are used, instructions: {instr_tp}, tr_irreps_out: {irreps_out}'

        self.tp = o3.TensorProduct(
            irreps_in1.irreps, irreps_in2.irreps, irreps_out.irreps,
            instr_tp, internal_weights=False, shared_weights=True, **kwargs
        )

        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x1, x2):
        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        weights = torch.cat(weights)
        return self.tp(x1, x2, weights)


class TRFullyConnectedTensorProduct(o3.TensorProduct):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        **kwargs
    ):
        assert isinstance(irreps_in1, TRIrreps)
        assert isinstance(irreps_in2, TRIrreps)
        assert isinstance(irreps_out, TRIrreps)

        instr = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, ((_, ir_1), tr_1) in enumerate(irreps_in1)
            for i_2, ((_, ir_2), tr_2) in enumerate(irreps_in2)
            for i_out, ((_, ir_out), tr_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2 and tr_out == tr_1 * tr_2
        ]
        super().__init__(irreps_in1.irreps, irreps_in2.irreps, irreps_out.irreps, instr, **kwargs)


@compile_mode('script')
class TRSkipConnection(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        assert isinstance(irreps_in, TRIrreps)
        assert isinstance(irreps_out, TRIrreps)
        self.sc = None
        if irreps_in == irreps_out:
            self.sc = None
        else:
            instr_lin = get_tr_instr_lin(irreps_in, irreps_out, allow_zero=True)
            self.sc = o3.Linear(irreps_in=irreps_in.irreps, irreps_out=irreps_out.irreps, instructions=instr_lin)

    def forward(self, old, new):
        if self.sc is not None:
            old = self.sc(old)

        return old + new

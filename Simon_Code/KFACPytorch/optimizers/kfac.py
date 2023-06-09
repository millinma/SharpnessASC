import math

import torch

from KFACPytorch.utils.kfac_utils import (ComputeCovA, ComputeCovG)
from KFACPytorch.utils.kfac_utils import update_running_stat
from KFACPytorch.utils.kfac_utils import get_matrix_form_grad

import warnings
warnings.filterwarnings("ignore")


class KFACOptimizer(torch.optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 solver='symeig',
                 print_layers=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)

        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.print_layers = print_layers
        self.model = model
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self._register_modules()

        # utility vars
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.stat_decay = 0  # stat_decay
        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        self.steps = 0

        # one-level KFAC vars
        self.solver = solver
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.Inv_a, self.Inv_g = {}, {}

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            with torch.no_grad():
                aa, _ = self.CovAHandler(input[0], module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.zeros_like(aa)
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg, _ = self.CovGHandler(
                grad_output[0], module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.zeros_like(gg)
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _register_modules(self):
        count = 0
        if self.print_layers:
            print(self.model)
            print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                if self.print_layers:
                    print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition or approximate factorization for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        if self.solver == 'symeig':
            eps = 1e-10  # for numerical stability
            self.d_a[m], self.Q_a[m] = torch.symeig(
                self.m_aa[m], eigenvectors=True)
            self.d_g[m], self.Q_g[m] = torch.symeig(
                self.m_gg[m], eigenvectors=True)

            self.d_a[m].mul_((self.d_a[m] > eps).float())
            self.d_g[m].mul_((self.d_g[m] > eps).float())
        else:
            group = self.param_groups[0]
            damping = group['damping']
            numer = self.m_aa[m].trace() * self.m_gg[m].shape[0]
            denom = self.m_gg[m].trace() * self.m_aa[m].shape[0]
            pi = numer / denom
            assert numer > 0, "trace(A) should be positive"
            assert denom > 0, "trace(G) should be positive"
            # assert pi > 0, "pi should be positive"
            diag_a = self.m_aa[m].new_full(
                (self.m_aa[m].shape[0],), (damping * pi)**0.5)
            diag_g = self.m_gg[m].new_full(
                (self.m_gg[m].shape[0],), (damping / pi)**0.5)
            self.Inv_a[m] = (self.m_aa[m] + torch.diag(diag_a)).inverse()
            self.Inv_g[m] = (self.m_gg[m] + torch.diag(diag_g)).inverse()

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        if self.solver == 'symeig':
            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (self.d_g[m].unsqueeze(1) *
                       self.d_a[m].unsqueeze(0) + damping)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        else:
            v = self.Inv_g[m] @ p_grad_mat @ self.Inv_a[m]

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.size())
            v[1] = v[1].view(m.bias.grad.size())
        else:
            v = [v.view(m.weight.grad.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad * lr ** 2).sum()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad * lr ** 2).sum()
        assert vg_sum != 0, "vg_sum should be non-zero"
        assert vg_sum > 0, "vg_sum should be positive"
        nu = min(1.0, (self.kl_clip / vg_sum)**0.5)
        # update grad
        for m in self.modules:
            v = updates[m]
            m.weight.grad.copy_(v[0])
            m.weight.grad.mul_(nu)
            if m.bias is not None:
                m.bias.grad.copy_(v[1])
                m.bias.grad.mul_(nu)

    @torch.no_grad()
    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:  # and self.steps >= 20 * self.TCov:
                    d_p.add_(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-group['lr'])

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            p_grad_mat = get_matrix_form_grad(m)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
        self.stat_decay = min(1.0 - 1.0 / (self.steps // self.TCov + 1), 0.95)

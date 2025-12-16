import math
import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from spikingjelly.clock_driven import neuron
from surrogate import Arctan


class VO2SingleSpikeLIF(neuron.BaseNode):
    """
    dV/dt = -V/tau + R·(Σw·spike_in)/tau
    V(t) = exp(-Δt/tau)·V(t-1) + (1-exp(-Δt/tau))·R·Σ(w·spike_in)
    """
    def __init__(self, tau, R, v_threshold, v_reset, dt,
                 surrogate_function: Callable = Arctan(mag=math.pi),
                 cv_c2c_vth=0., cv_c2c_iin=0.,
                 cv_d2d_vth=0., cv_d2d_rv=0., cv_d2d_iin=0.,
                 detach_reset: bool = True):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.R = R
        self.dt = dt
        self.factor = torch.exp(torch.tensor(-dt / tau))

        self.v = 0.
        self.register_memory('spike', 0)
        self.register_memory('spiked', False)

        self.cv_c2c_vth = cv_c2c_vth
        self.std_c2c_vth = cv_c2c_vth * 1.0
        self.register_memory('noise_c2c_vth', None)

        self.cv_c2c_iin = cv_c2c_iin
        self.std_c2c_iin = cv_c2c_iin * 1.0
        self.register_memory('noise_c2c_iin', None)

        self.cv_d2d_vth = cv_d2d_vth
        self.std_d2d_vth = cv_d2d_vth * 1.0
        self.cv_d2d_rv = cv_d2d_rv
        self.std_d2d_rv = cv_d2d_rv * 1.0
        self.cv_d2d_iin = cv_d2d_iin
        self.std_d2d_iin = cv_d2d_iin * 1.0
        # d2d, not declared using register_memory so that it is not reset under functional.reset_net (i.e. is fixed)
        self.noise_d2d_vth = None
        self.noise_d2d_rv = None
        self.noise_d2d_iin = None

    def neuronal_charge(self, x: torch.Tensor):
        x = x * (1 + self.noise_d2d_iin)  # d2d cmos noise
        x = x * (1 + self.noise_c2c_iin)  # c2c vdd noise
        R_noisy = self.R * (1 + self.noise_d2d_rv)  # d2d rv noise
        self.v = self.factor * self.v + torch.where(self.spiked, torch.zeros_like(x), (1 - self.factor) * R_noisy * x)

    def neuronal_fire(self):
        v_threshold_noisy = self.v_threshold * (1 + self.noise_d2d_vth)  # d2d vth variations of vo2
        v_threshold_noisy = v_threshold_noisy * (1 + self.noise_c2c_vth)  # c2c vth variations of vo2
        v_threshold_noisy = torch.where(v_threshold_noisy <= 0, 99999., v_threshold_noisy)
        spike = self.surrogate_function((self.v - v_threshold_noisy) / v_threshold_noisy)
        self.spike = spike
        self.spiked = torch.where(spike == 1., torch.ones_like(self.spiked), self.spiked)
        return spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        v_reset_noisy = self.v_reset * (1 + self.noise_c2c_vth)
        v_reset_noisy = torch.where(v_reset_noisy <= 0, 99999., v_reset_noisy)
        self.v = (1 - spike_d) * self.v + spike_d * v_reset_noisy

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.spiked, torch.Tensor):
                self.spiked = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
            if not isinstance(self.noise_c2c_vth, torch.Tensor):
                self.noise_c2c_vth = torch.normal(0., self.std_c2c_vth, size=x.shape).to(device=x.device)
            if not isinstance(self.noise_c2c_iin, torch.Tensor):
                self.noise_c2c_iin = torch.normal(0., self.std_c2c_iin, size=x.shape).to(device=x.device)

            # d2d, different between channels, same across batches, hence size is x.shape[-1], will be broadcasted up
            if not isinstance(self.noise_d2d_vth, torch.Tensor):
                self.noise_d2d_vth = torch.normal(0., self.std_d2d_vth, size=(x.shape[-1],)).to(device=x.device)
            if not isinstance(self.noise_d2d_rv, torch.Tensor):
                self.noise_d2d_rv = torch.normal(0., self.std_d2d_rv, size=(x.shape[-1],)).to(device=x.device)
            if not isinstance(self.noise_d2d_iin, torch.Tensor):
                self.noise_d2d_iin = torch.normal(0., self.std_d2d_iin, size=(x.shape[-1],)).to(device=x.device)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike


class VO2VoltageSingleSpikeLIF(neuron.BaseNode):
    """
    V_in = relu[β·Σ(w·spike_in) + γ], β: TIA amplification, γ: voltage bias due to v_read virtual ground
    C·dV/dt = -V/R + (V_in-V)/R_l
    dV/dt = -V/tau + (R/(R+R_l))·V_in/tau, tau = (R||R_l)·C
    V(t) = exp(-Δt/tau)·V(t-1) + (1-exp(-Δt/tau))·(R/(R+R_l))·V_in
    """
    def __init__(self, tau, R, Rl, v_threshold, v_reset, dt,
                 surrogate_function: Callable = Arctan(mag=math.pi),
                 cv_c2c=0.,
                 detach_reset: bool = True):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.R = R
        self.Rl = Rl
        self.dt = dt
        self.factor = torch.exp(torch.tensor(-dt / tau))
        self.v_div = R / (R + Rl)

        self.v = 0.
        self.register_memory('spike', 0)
        self.register_memory('spiked', False)

        self.cv_c2c = cv_c2c
        self.std_c2c = cv_c2c * 1.0
        self.register_memory('noise_c2c', None)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.factor * self.v + torch.where(self.spiked, torch.zeros_like(x), (1 - self.factor) * self.v_div * x)

    def neuronal_fire(self):
        v_threshold_noisy = self.v_threshold * (1 + self.noise_c2c)
        v_threshold_noisy = torch.where(v_threshold_noisy <= 0, 99999., v_threshold_noisy)
        spike = self.surrogate_function((self.v - v_threshold_noisy) / v_threshold_noisy)
        self.spike = spike
        self.spiked = torch.where(spike == 1., torch.ones_like(self.spiked), self.spiked)
        return spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        v_reset_noisy = self.v_reset * (1 + self.noise_c2c)
        v_reset_noisy = torch.where(v_reset_noisy <= 0, 99999., v_reset_noisy)
        self.v = (1 - spike_d) * self.v + spike_d * v_reset_noisy

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.spiked, torch.Tensor):
                self.spiked = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
            if not isinstance(self.noise_c2c, torch.Tensor):
                self.noise_c2c = torch.normal(0., self.std_c2c, size=x.shape).to(device=x.device)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike


class MultiLayerSingleSpikeNet(nn.Module):
    def __init__(self, num_in, num_hidden, num_out,
                 tau, R, v_threshold, v_reset, cv_c2c, weight_scaling, dt, device,
                 weight_init, weight_bump=None, taus=None, input_limit=torch.inf,
                 vin_mode=False, Rl=None, beta=None, gamma=None,
                 input_width=2e-6, nonideal_input_tau=None,
                 hidden_bias_units=None, hidden_bias_times=None,
                 cv_c2c_vth=0., cv_c2c_iin=0.,
                 cv_d2d_vth=0., cv_d2d_rv=0., cv_d2d_iin=0.,
                 syn_decay=False,
        ):
        super(MultiLayerSingleSpikeNet, self).__init__()

        assert len(num_hidden) == len(weight_init["mean"]) - 1 == len(weight_init["std"]) - 1 == len(weight_bump["max_no_spike"]) - 1
        assert hidden_bias_times is None or len(hidden_bias_times) == len(num_hidden)
        assert hidden_bias_units is None or len(hidden_bias_units) == len(num_hidden)
        if hidden_bias_units is not None:
            for i in hidden_bias_units:
                if i > 1:
                    raise ValueError("hidden_bias_units must be 1 or 0 or None for now, >1 not implemented yet")

        self.num_in = num_in
        self.num_hidden = num_hidden  # list of int(s) (>= 1 hidden layer)
        self.num_out = num_out
        self.tau = tau
        if taus is None:
            print("taus not given, defaulting to taus=tau!")
            self.taus = tau
        else:
            self.taus = taus
        self.R = R
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.weight_init = weight_init  # dict of mean and std, each a list with length 1 more than num_hidden
        self.weight_scaling = weight_scaling
        self.dt = dt
        self.device = device
        self.input_limit = input_limit
        self.vin_mode = vin_mode
        self.Rl = Rl
        self.beta = beta
        self.gamma = gamma
        self.input_width = input_width
        self.nonideal_input_tau = nonideal_input_tau
        self.nonideal_input_factor = math.exp(-dt / nonideal_input_tau) if nonideal_input_tau is not None else 0.
        self.hidden_bias_units = hidden_bias_units
        self.hidden_bias_times = hidden_bias_times
        self.syn_decay = syn_decay
        self.factor_taus = torch.exp(torch.tensor(-self.dt / self.taus, device=self.device))

        self.fc1 = nn.Linear(num_in, num_hidden[0] - hidden_bias_units[0], bias=False)
        self.hidden = []
        self.fc2 = []
        for i in range(len(num_hidden)):
            self.hidden.append(VO2SingleSpikeLIF(
                tau, R, v_threshold, v_reset, dt,
                cv_c2c_vth=cv_c2c_vth, cv_c2c_iin=cv_c2c_iin,
                cv_d2d_vth=cv_d2d_vth, cv_d2d_rv=cv_d2d_rv, cv_d2d_iin=cv_d2d_iin,
            ))
            if i == len(num_hidden) - 1:
                self.fc2.append(nn.Linear(num_hidden[i], num_out, bias=False))
            else:
                self.fc2.append(nn.Linear(num_hidden[i], num_hidden[i+1] - hidden_bias_units[i+1], bias=False))
        self.hidden = nn.ModuleList(self.hidden)
        self.fc2 = nn.ModuleList(self.fc2)
        self.out = VO2SingleSpikeLIF(
            tau, R, v_threshold, v_reset, dt,
            cv_c2c_vth=cv_c2c_vth, cv_c2c_iin=cv_c2c_iin,
            cv_d2d_vth=cv_d2d_vth, cv_d2d_rv=cv_d2d_rv, cv_d2d_iin=cv_d2d_iin,
        )

        self.hidden_biases = []

        nn.init.normal_(self.fc1.weight, weight_init["mean"][0], weight_init["std"][0])
        for i in range(len(num_hidden)):
            nn.init.normal_(self.fc2[i].weight, weight_init["mean"][i + 1], weight_init["std"][i + 1])

        self.fc1.weight.register_hook(lambda grad: print(f'FC1 Grad\n{grad}\nFC1 Weight\n{self.fc1.weight.data}') if torch.isnan(grad).any() else None)
        for i in range(len(num_hidden)):
            self.fc2[i].weight.register_hook(lambda grad: print(f'FC2-{i} Grad\n{grad}\nFC2-{i} Weight\n{self.fc2[i].weight.data}') if torch.isnan(grad).any() else None)

        self.weight_bump = {
            'max_no_spike': [0.3, 0.0],
            'weight_bump_exp': True,
            'initial_bump_value': 0.0005
        } if weight_bump is None else weight_bump  # length should be 1 more than num_hidden
        self.bump_val = 0.
        self.last_bump_layer = -2

        self.v_1 = None
        self.spike_1 = None
        self.neuron_in_1 = None
        self.p_1 = None
        self.spike_for_reg_1 = None
        self.v_for_reg_1 = None
        self.x_for_reg_1 = None
        self.p_for_stat_1 = None

        self.v_2 = None
        self.spike_2 = None
        self.neuron_in_2 = None
        self.p_2 = None
        self.spike_for_reg_2 = None
        self.v_for_reg_2 = None
        self.x_for_reg_2 = None
        self.p_2 = None
        self.p_for_stat_2 = None

    def s_stat(self, end_time=None):
        if end_time is not None:
            assert len(end_time) == len(self.spike_for_reg_1[0]) == len(self.spike_for_reg_2)
            # spike_for_reg_x.shape = [batch, timestep, neuron]
            # end_time is the timestep where classification has ended (firing of first spike in the output layer)
            # do a cumulative sum over timestep,
            # get energy values for all batches and neurons from indices in the timestep dim specified by end_time
            spike_hidden_single_neuron = [torch.cumsum(self.spike_for_reg_1[i], dim=1)[range(len(end_time)), end_time, :] for i in range(len(self.hidden))]
            spike_out_single_neuron = torch.cumsum(self.spike_for_reg_2, dim=1)[range(len(end_time)), end_time, :]
        else:
            spike_hidden_single_neuron = [torch.sum(self.spike_for_reg_1[i], dim=1) for i in range(len(self.hidden))]
            spike_out_single_neuron = torch.sum(self.spike_for_reg_2, dim=1)

        spike_hidden_total = [torch.sum(spike_hidden_single_neuron[i], dim=1) for i in range(len(self.hidden))]  # [layers][batch, neuron]
        spike_out_total = torch.sum(spike_out_single_neuron, dim=1)

        spike_total = torch.stack(spike_hidden_total, dim=0).sum(dim=-1).sum(dim=0) + spike_out_total  # [batch]

        return spike_hidden_total, spike_out_total, spike_total

    def p_stat(self, end_time=None):
        if end_time is not None:
            assert len(end_time) == len(self.p_for_stat_1[0]) == len(self.p_for_stat_2)
            # p_for_stat_x.shape = [batch, timestep, neuron]
            # end_time is the timestep where classification has ended (firing of first spike in the output layer)
            # do a cumulative sum over timestep,
            # get energy values for all batches and neurons from indices in the timestep dim specified by end_time
            energy_hidden_single_neuron = [torch.cumsum(self.p_for_stat_1[i], dim=1)[range(len(end_time)), end_time, :] * self.dt for i in range(len(self.hidden))]
            energy_out_single_neuron = torch.cumsum(self.p_for_stat_2, dim=1)[range(len(end_time)), end_time, :] * self.dt
        else:
            energy_hidden_single_neuron = [torch.sum(self.p_for_stat_1[i], dim=1) * self.dt for i in range(len(self.hidden))]
            energy_out_single_neuron = torch.sum(self.p_for_stat_2, dim=1) * self.dt

        energy_hidden_total = [torch.sum(energy_hidden_single_neuron[i], dim=1) for i in range(len(self.hidden))]  # [layers][batch, neuron]
        energy_out_total = torch.sum(energy_out_single_neuron, dim=1)

        energy_total = torch.stack(energy_hidden_total, dim=0).sum(dim=-1).sum(dim=0) + energy_out_total  # [batch]

        return energy_hidden_total, energy_out_total, energy_total

    def x_stat(self, end_time=None):
        x_nonzero_hidden = [torch.where(self.x_for_reg_1[i] > 0., 1, 0.) for i in range(len(self.hidden))]
        x_nonzero_out = torch.where(self.x_for_reg_2 > 0., 1, 0.)
        if end_time is not None:
            assert len(end_time) == len(self.x_for_reg_1[0]) == len(self.x_for_reg_2)
            # x_for_reg_x.shape = [batch, timestep, neuron]
            # end_time is the timestep where classification has ended (firing of first spike in the output layer)
            # do a cumulative sum over timestep,
            # get energy values for all batches and neurons from indices in the timestep dim specified by end_time
            x_hidden_single_neuron = [torch.cumsum(self.x_for_reg_1[i], dim=1)[range(len(end_time)), end_time, :] * self.dt for i in range(len(self.hidden))]
            x_out_single_neuron = torch.cumsum(self.x_for_reg_2, dim=1)[range(len(end_time)), end_time, :] * self.dt
            x_nonzero_hidden_single_neuron = [torch.cumsum(x_nonzero_hidden[i], dim=1)[range(len(end_time)), end_time, :] for i in range(len(self.hidden))]
            x_nonzero_out_single_neuron = torch.cumsum(x_nonzero_out, dim=1)[range(len(end_time)), end_time, :]
        else:
            x_hidden_single_neuron = [torch.sum(self.x_for_reg_1[i], dim=1) * self.dt for i in range(len(self.hidden))]
            x_out_single_neuron = torch.sum(self.x_for_reg_2, dim=1) * self.dt
            x_nonzero_hidden_single_neuron = [torch.sum(x_nonzero_hidden[i], dim=1) for i in range(len(self.hidden))]
            x_nonzero_out_single_neuron = torch.sum(x_nonzero_out, dim=1)

        x_hidden_total = [torch.sum(x_hidden_single_neuron[i], dim=1) for i in range(len(self.hidden))]  # [layers][batch, neuron]
        x_out_total = torch.sum(x_out_single_neuron, dim=1)
        x_nonzero_hidden_total = [torch.sum(x_nonzero_hidden_single_neuron[i], dim=1) for i in range(len(self.hidden))]
        x_nonzero_out_total = torch.sum(x_nonzero_out_single_neuron, dim=1)

        x_total = torch.stack(x_hidden_total, dim=0).sum(dim=-1).sum(dim=0) + x_out_total  # [batch]
        x_nonzero_total = torch.stack(x_nonzero_hidden_total, dim=0).sum(dim=-1).sum(dim=0) + x_nonzero_out_total

        return x_total, x_nonzero_total

    def bump_weights(self):
        weights_bumped_layer = -2

        for i in range(len(self.hidden)):
            no_spike = self.hidden[i].spiked.logical_not()
            if no_spike.sum() / (no_spike.numel()) > self.weight_bump['max_no_spike'][i]:
                weights_bumped_layer = i
                break

        if weights_bumped_layer == -2:  # if still -2, check output layer
            no_spike = self.out.spiked.logical_not()
            if no_spike.sum() / (no_spike.numel()) > self.weight_bump['max_no_spike'][-1]:
                weights_bumped_layer = -1

        if weights_bumped_layer != -2:
            if self.weight_bump['weight_bump_exp'] and weights_bumped_layer == self.last_bump_layer:
                self.bump_val *= 2
            else:
                self.bump_val = self.weight_bump['initial_bump_value']

            # make bool and then int to have either zero or ones
            should_bump = no_spike.sum(dim=0).bool().int()
            if weights_bumped_layer == -1:
                n_in = self.fc2[-1].weight.data.size()[1]
                bumps = should_bump.repeat(n_in, 1) * self.bump_val
                self.fc2[-1].weight.data += bumps.T
            elif weights_bumped_layer == 0:
                n_in = self.fc1.weight.data.size()[1]
                bumps = should_bump.repeat(n_in, 1) * self.bump_val
                self.fc1.weight.data += bumps.T
            else:
                n_in = self.fc2[weights_bumped_layer - 1].weight.data.size()[1]
                bumps = should_bump.repeat(n_in, 1) * self.bump_val
                self.fc2[weights_bumped_layer - 1].weight.data += bumps.T

        self.last_bump_layer = weights_bumped_layer
        return weights_bumped_layer

    def forward(self, x: torch.Tensor):
        # x.shape = [batch, times, features]

        self.spike_for_reg_1 = [torch.empty([x.shape[0], x.shape[1], self.num_hidden[i]], device=self.device) for i in range(len(self.hidden))]
        self.v_for_reg_1 = [torch.empty([x.shape[0], x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]], device=self.device) for i in range(len(self.hidden))]
        self.x_for_reg_1 = [torch.empty([x.shape[0], x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]], device=self.device) for i in range(len(self.hidden))]
        self.p_for_stat_1 = [torch.empty([x.shape[0], x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]], device=self.device) for i in range(len(self.hidden))]

        self.spike_for_reg_2 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.v_for_reg_2 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.x_for_reg_2 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.p_for_stat_2 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)

        if x.shape[0] == 1:
            self.v_1 = [torch.empty([x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]]) for i in range(len(self.hidden))]
            self.p_1 = [torch.empty([x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]]) for i in range(len(self.hidden))]
            self.spike_1 = [torch.empty([x.shape[1], self.num_hidden[i]]) for i in range(len(self.hidden))]
            self.neuron_in_1 = [torch.empty([x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]]) for i in range(len(self.hidden))]

            self.v_2 = torch.empty([x.shape[1], self.num_out])
            self.p_2 = torch.empty([x.shape[1], self.num_out])
            self.spike_2 = torch.empty([x.shape[1], self.num_out])
            self.neuron_in_2 = torch.empty([x.shape[1], self.num_out])

        y_seq = []

        x = x.permute(1, 0, 2)  # [times, batch, features]

        #  for now, only one biasing neuron per hidden layer, with >= 1 bias spike(s)
        self.hidden_biases = []
        if self.hidden_bias_times is not None:
            for i in range(len(self.hidden_bias_times)):
                t = self.hidden_bias_times[i]
                assert isinstance(t, (list, np.ndarray, int))
                if isinstance(t, list):
                    t = np.array(t)
                t = np.round(t / self.dt).astype(int)
                bias = torch.zeros([x.shape[0], x.shape[1], 1], device=self.device)
                bias[t] = 1.
                self.hidden_biases.append(bias)

        syn_1 = [torch.zeros((x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]), device=self.device) for i in range(len(self.hidden))]
        syn_2 = torch.zeros((x.shape[1], self.num_out), device=self.device)
        buffer_1 = [torch.zeros((int(self.input_width / self.dt), x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]), device=self.device) for i in range(len(self.hidden))]
        buffer_2 = torch.zeros((int(self.input_width / self.dt), x.shape[1], self.num_out), device=self.device)
        neuron_in_1 = [torch.zeros((x.shape[1], self.num_hidden[i] - self.hidden_bias_units[i]), device=self.device) for i in range(len(self.hidden))]
        neuron_in_2 = torch.zeros((x.shape[1], self.num_out), device=self.device)

        no_buffer = True if len(buffer_2) == 0 else False

        for t in range(x.shape[0]):
            y_ = self.fc1(x[t])

            for i in range(len(self.hidden)):
                if self.syn_decay:
                    syn_1[i] = self.factor_taus * syn_1[i] + y_ * self.weight_scaling
                    y = syn_1[i]
                else:
                    if no_buffer:
                        y = neuron_in_1[i] * self.nonideal_input_factor + (1 - self.nonideal_input_factor) * y_ * self.weight_scaling
                    else:
                        buffer_1[i] += y_ * self.weight_scaling
                        y = neuron_in_1[i] * self.nonideal_input_factor + (1 - self.nonideal_input_factor) * buffer_1[i][0]
                        buffer_1[i] = torch.roll(buffer_1[i], -1, 0)
                        buffer_1[i][-1] = 0
                if self.vin_mode:
                    y = self.beta * y + self.gamma
                y = torch.relu(y)
                y = torch.where(y > self.input_limit, self.input_limit, y)
                neuron_in_1[i] = y
                y = self.hidden[i](y)
                if self.hidden_bias_times is not None:
                    y = torch.concatenate([y, self.hidden_biases[i][t]], dim=-1)
                y_ = self.fc2[i](y)
                if self.vin_mode:
                    self.p_for_stat_1[i][:, t] = neuron_in_1[i] * (neuron_in_1[i] - self.hidden[i].v) / self.Rl * (~self.hidden[i].spiked)
                else:
                    self.p_for_stat_1[i][:, t] = self.hidden[i].v * neuron_in_1[i] * (~self.hidden[i].spiked)
                self.spike_for_reg_1[i][:, t] = y

            if self.syn_decay:
                syn_2 = self.factor_taus * syn_2 + y_ * self.weight_scaling
                y = syn_2
            else:
                if no_buffer:
                    y = neuron_in_2 * self.nonideal_input_factor + (1 - self.nonideal_input_factor) * y_ * self.weight_scaling
                else:
                    buffer_2 += y_ * self.weight_scaling
                    y = neuron_in_2 * self.nonideal_input_factor + (1 - self.nonideal_input_factor) * buffer_2[0]
                    buffer_2 = torch.roll(buffer_2, -1, 0)
                    buffer_2[-1] = 0
            if self.vin_mode:
                y = self.beta * y + self.gamma
            y = torch.relu(y)
            y = torch.where(y > self.input_limit, self.input_limit, y)
            neuron_in_2 = y
            y = self.out(y)
            if self.vin_mode:
                self.p_for_stat_2[:, t] = neuron_in_2 * (neuron_in_2 - self.out.v) / self.Rl * (~self.out.spiked)
            else:
                self.p_for_stat_2[:, t] = self.out.v * neuron_in_2 * (~self.out.spiked)
            self.spike_for_reg_2[:, t] = y

            y_seq.append(y.unsqueeze(0))

            if x.shape[1] == 1:
                for i in range(len(self.hidden)):
                    self.v_1[i][t] = self.hidden[i].v
                    self.spike_1[i][t] = torch.concatenate([self.hidden[i].spike, self.hidden_biases[i][t]], dim=-1) if self.hidden_bias_times is not None else self.hidden[i].spike
                    self.neuron_in_1[i][t] = neuron_in_1[i]
                    if self.vin_mode:
                        self.p_1[i][t] = neuron_in_1[i] * (neuron_in_1[i] - self.hidden[i].v) / self.Rl * (~self.hidden[i].spiked)
                    else:
                        self.p_1[i][t] = self.hidden[i].v * neuron_in_1[i] * (~self.hidden[i].spiked)

                self.v_2[t] = self.out.v
                self.spike_2[t] = self.out.spike
                self.neuron_in_2[t] = neuron_in_2
                if self.vin_mode:
                    self.p_2[t] = neuron_in_2 * (neuron_in_2 - self.out.v) / self.Rl * (~self.out.spiked)
                else:
                    self.p_2[t] = self.out.v * neuron_in_2 * (~self.out.spiked)

        return torch.cat(y_seq, 0)


class SingleLayerSingleSpikeNet(nn.Module):
    def __init__(self, num_in, num_out,
                 tau, R, v_threshold, v_reset, cv_c2c, weight_scaling, dt, device,
                 weight_init, weight_bump=None, taus=None, input_limit=torch.inf,
                 vin_mode=False, Rl=None, beta=None, gamma=None,
                 input_width=2e-6, nonideal_input_tau=None,
                 cv_c2c_vth=0., cv_c2c_iin=0.,
                 cv_d2d_vth=0., cv_d2d_rv=0., cv_d2d_iin=0.,
        ):
        super(SingleLayerSingleSpikeNet, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.tau = tau
        if taus is None:
            print("taus not given, defaulting to taus=tau!")
            self.taus = tau
        else:
            self.taus = taus
        self.R = R
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.weight_init = weight_init
        self.weight_scaling = weight_scaling
        self.dt = dt
        self.device = device
        self.input_limit = input_limit
        self.vin_mode = vin_mode
        self.Rl = Rl
        self.beta = beta
        self.gamma = gamma
        self.input_width = input_width
        self.nonideal_input_tau = nonideal_input_tau
        self.nonideal_input_factor = math.exp(-dt / nonideal_input_tau) if nonideal_input_tau is not None else 0.

        self.fc1 = nn.Linear(num_in, num_out, bias=False)
        if vin_mode:
            self.out = VO2VoltageSingleSpikeLIF(tau, R, Rl, v_threshold, v_reset, dt, cv_c2c=cv_c2c)
        else:
            self.out = VO2SingleSpikeLIF(
                tau, R, v_threshold, v_reset, dt,
                cv_c2c_vth=cv_c2c_vth, cv_c2c_iin=cv_c2c_iin,
                cv_d2d_vth=cv_d2d_vth, cv_d2d_rv=cv_d2d_rv, cv_d2d_iin=cv_d2d_iin,
            )

        nn.init.normal_(self.fc1.weight, weight_init["mean"][0], weight_init["std"][0])

        self.fc1.weight.register_hook(lambda grad: print(f'FC1 Grad\n{grad}\nFC1 Weight\n{self.fc1.weight.data}') if torch.isnan(grad).any() else None)

        self.weight_bump = {
            'max_no_spike': [0.0],
            'weight_bump_exp': True,
            'initial_bump_value': 0.0005
        } if weight_bump is None else weight_bump
        self.bump_val = 0.
        self.last_bump_layer = -2

        self.v_1 = None
        self.spike_1 = None
        self.neuron_in_1 = None
        self.p_1 = None
        self.spike_for_reg_1 = None
        self.v_for_reg_1 = None
        self.x_for_reg_1 = None
        self.p_for_stat_1 = None

    def x_stat(self):
        temp_x = self.x_for_reg[self.x_for_reg.nonzero(as_tuple=True)]

        return {'elements': temp_x, 'mean': torch.mean(temp_x), 'std': torch.std(temp_x), 'max': torch.max(temp_x)}

    def s_stat(self):
        spike_total = torch.sum(self.spike_for_reg_1, dim=(1, 2))

        return None, spike_total, spike_total

    def p_stat(self):
        energy_out_single_neuron = torch.sum(self.p_for_stat_1, dim=1) * self.dt
        energy_out_total = torch.sum(energy_out_single_neuron, dim=1)

        energy_total = energy_out_total

        return None, energy_out_total, energy_total

    def bump_weights(self):
        weights_bumped_layer = -2

        no_spike = self.out.spiked.logical_not()
        if no_spike.sum() / (no_spike.numel()) > self.weight_bump['max_no_spike'][-1]:
            weights_bumped_layer = -1
            if self.weight_bump['weight_bump_exp'] and weights_bumped_layer == self.last_bump_layer:
                self.bump_val *= 2
            else:
                self.bump_val = self.weight_bump['initial_bump_value']

            # make bool and then int to have either zero or ones
            should_bump = no_spike.sum(dim=0).bool().int()
            n_in = self.fc1.weight.data.size()[1]
            bumps = should_bump.repeat(n_in, 1) * self.bump_val
            self.fc1.weight.data += bumps.T

        self.last_bump_layer = weights_bumped_layer
        return weights_bumped_layer

    def forward(self, x: torch.Tensor):
        # x.shape = [batch, times, features]
        self.spike_for_reg_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.v_for_reg_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.x_for_reg_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.p_for_stat_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)

        if x.shape[0] == 1:
            self.v_1 = torch.empty([x.shape[1], self.num_out])
            self.p_1 = torch.empty([x.shape[1], self.num_out])
            self.spike_1 = torch.empty([x.shape[1], self.num_out])
            self.neuron_in_1 = torch.empty([x.shape[1], self.num_out])

        y_seq = []

        x = x.permute(1, 0, 2)  # [times, batch, features]

        buffer_1 = torch.zeros((int(self.input_width / self.dt), x.shape[1], self.num_out), device=self.device)
        neuron_in_1 = torch.zeros((x.shape[1], self.num_out), device=self.device)

        for t in range(x.shape[0]):
            y_ = self.fc1(x[t])
            buffer_1 += y_ * self.weight_scaling
            y = neuron_in_1 * self.nonideal_input_factor + (1 - self.nonideal_input_factor) * buffer_1[0]
            buffer_1 = torch.roll(buffer_1, -1, 0)
            buffer_1[-1] = 0
            if self.vin_mode:
                y = self.beta * y + self.gamma
            y = torch.relu(y)
            y = torch.where(y > self.input_limit, self.input_limit, y)
            neuron_in_1 = y
            y = self.out(y)
            if self.vin_mode:
                self.p_for_stat_1[:, t] = neuron_in_1 * (neuron_in_1 - self.out.v) / self.Rl * (~self.out.spiked)
            else:
                self.p_for_stat_1[:, t] = self.out.v * neuron_in_1 * (~self.out.spiked)
            self.spike_for_reg_1[:, t] = y

            y_seq.append(y.unsqueeze(0))

            if x.shape[1] == 1:
                self.v_1[t] = self.out.v
                self.spike_1[t] = self.out.spike
                self.neuron_in_1[t] = neuron_in_1
                self.p_1[t] = self.out.v * neuron_in_1 * (~self.out.spiked)

        return torch.cat(y_seq, 0)

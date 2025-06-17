import numpy as np
import matplotlib.pyplot as plt
import torch


def msst_loss(output: torch.Tensor, target: torch.Tensor, tau: float, no_spike_time: float = 80e-6):
    """
        Mean Square Spike Time
        see Appendix B9 of Eshraghian et al. (2023) Proc. IEEE
    """
    loss = 0.5 * ((torch.where(output == torch.inf, torch.ones_like(output) * no_spike_time, output) - target) / tau) ** 2
    return loss.mean()


def cerst_loss(output: torch.Tensor, label: torch.Tensor, xi: float, tau: float, reg=None):
    """
        Cross Entropy Relative Spike Time
        cross entropy between the true label distribution and the softmax-scaled label spike times
        only depends on the spike time difference, invariant under absolute time shifts
        see Goltz et al. (2021) Nat. Mach. Intell.
        :param reg:
        :param tau:
        :param output:
        :param label:
        :param xi: scaling parameter
        :return:
    """
    output = torch.where(output == torch.inf, torch.ones_like(output) * 100e-6, output)

    label_clone = label.clone().type(torch.long).view(-1, 1)
    baseline = output.gather(1, label_clone)
    scaling_factor = torch.tensor(xi * tau, requires_grad=True)
    numerator = (output - baseline)
    exponent = torch.exp(- numerator / scaling_factor)
    loss = torch.log(torch.sum(exponent, dim=1))
    if reg is not None:
        assert len(reg) == 2, f"Regularizer needs 2 params: alpha=reg[0] & beta=reg[1], received {len(reg)} params!"
        regularizer = reg[0] * (torch.exp(baseline / (reg[1] * tau)) - 1)
        loss = loss + regularizer
    return loss.mean()


class FirstSpikeTime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, dt):
        # s: spike train : [timestep, batch_size, n_out]
        assert torch.max(torch.sum(s, dim=0)) <= 1., f">1 spikes found for some neuron(s):\n{s.shape}\n{s}"
        ts = torch.sum(s.transpose(0, -1) * (torch.arange(s.shape[0], device=s.device).detach() + 1), dim=-1) - 1  # [n_out, batch_size]
        s_alt = s
        ts = torch.where(ts < 0., torch.ones_like(ts) * torch.inf, ts)
        ts *= dt
        ts = ts.transpose(0, -1)
        ctx.save_for_backward(s, s_alt, ts)
        return ts

    @staticmethod
    def backward(ctx, df_dts):
        s, s_alt, ts = ctx.saved_tensors
        # dts_ds = s * -1.#+ (1 - s) * -10  # torch.ones_like(s) * -1.  # s * -1
        dts_ds = s_alt * -1  # torch.ones_like(s) * -1.  # s_alt * -1
        df_ds = df_dts * dts_ds
        return df_ds, None


def get_spike_time(spikes: torch.Tensor, dt: float):
    assert torch.max(torch.sum(spikes, dim=-1)) <= 1., f">1 spikes found for some neuron(s):\n{spikes.shape}\n{spikes}"
    times = torch.sum(
        spikes * (torch.arange(spikes.shape[-1], device=spikes.device, requires_grad=True, dtype=torch.float) + 1),
        dim=-1) - 1
    times = torch.where(times < 0., spikes.shape[-1], times)
    times *= dt
    return times


def get_target_spike_time(labels: torch.Tensor, n_classes: int, true_time: float, false_time: float):
    target = torch.ones([len(labels), n_classes]).to(labels.device) * false_time
    target[torch.arange(len(labels), dtype=torch.long), labels.to(dtype=torch.long)] = true_time
    return target


def get_spike_time_stats(spike_times: torch.Tensor, target_labels: torch.Tensor,
                         spike_times_mean_collate, spike_times_std_collate,
                         spike_times_q25_collate, spike_times_q75_collate):
    spike_times = spike_times.detach().cpu().numpy()
    target_labels = target_labels.detach().cpu().numpy()
    n_sample, n_class = spike_times.shape
    spike_times[np.isinf(spike_times)] = np.nan
    spike_times_sorted = [[] for _ in range(n_class)]
    for c in range(n_class):
        indices = (target_labels == c).nonzero()
        spike_times_sorted[c] = spike_times[indices]
        mask_not_all_nan = np.logical_not(np.isnan(spike_times_sorted[c])).sum(0) > 0
        spike_times_mean = np.ones(n_class) * np.nan
        spike_times_std = np.ones(n_class) * np.nan
        spike_times_q25 = np.ones(n_class) * np.nan
        spike_times_q75 = np.ones(n_class) * np.nan
        spike_times_mean[mask_not_all_nan] = np.nanmean(spike_times_sorted[c][:, mask_not_all_nan], 0)
        spike_times_std[mask_not_all_nan] = np.nanstd(spike_times_sorted[c][:, mask_not_all_nan], 0)
        spike_times_q25[mask_not_all_nan] = np.nanquantile(spike_times_sorted[c][:, mask_not_all_nan], 0.25, 0)
        spike_times_q75[mask_not_all_nan] = np.nanquantile(spike_times_sorted[c][:, mask_not_all_nan], 0.75, 0)
        spike_times_mean_collate[c].append(spike_times_mean)
        spike_times_std_collate[c].append(spike_times_std)
        spike_times_q25_collate[c].append(spike_times_q25)
        spike_times_q75_collate[c].append(spike_times_q75)
    return spike_times_mean_collate, spike_times_std_collate, spike_times_q25_collate, spike_times_q75_collate


def sort_spike_times(spike_times: torch.Tensor, target_labels: torch.Tensor):
    spike_times = spike_times.detach().cpu().numpy()
    target_labels = target_labels.detach().cpu().numpy()
    n_sample, n_class = spike_times.shape

    # sort by earliest spike time
    spike_times_sorted_earliest = [[] for _ in range(n_class)]
    for c in range(n_class):
        indices = (target_labels == c).nonzero()
        spike_times_sorted_earliest[c] = spike_times[indices]
        earliest_spikes = np.min(spike_times_sorted_earliest[c], axis=1)
        indices = np.argsort(earliest_spikes)
        spike_times_sorted_earliest[c] = spike_times_sorted_earliest[c][indices]

    # sort by ground truth spike time
    spike_times_sorted_truth = [[] for _ in range(n_class)]
    for c in range(n_class):
        indices = (target_labels == c).nonzero()
        spike_times_sorted_truth[c] = spike_times[indices]
        ground_truth_spikes = spike_times_sorted_truth[c][:, c]
        indices = np.argsort(ground_truth_spikes)
        spike_times_sorted_truth[c] = spike_times_sorted_truth[c][indices]
    return spike_times_sorted_earliest, spike_times_sorted_truth


def apply_noise(sample, mean, std, scale=20e-6, dt=0.2e-6, T=80e-6, constant_cv=False):
    """
    When constant_cv is True, std = cv
    :param sample:
    :param mean:
    :param std:
    :param scale:
    :param dt:
    :param T:
    :param constant_cv:
    :return:
    """
    input_times, label = sample
    noise = torch.zeros_like(input_times)
    noise.normal_(mean, std)
    if constant_cv:
        input_times = input_times * (1 + noise)
    else:
        input_times = input_times + noise * scale
    negative = input_times < 0.
    input_times[negative] *= -1
    too_large = torch.round(input_times / dt) >= np.round(T / dt)
    input_times[too_large] = T - dt
    return input_times, label


def raster(ts, sp_matrix):
    sp_matrix = np.asarray(sp_matrix)
    if ts is None:
        raise
    ts = np.asarray(ts)

    # get index and time
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    t = ts[elements[0]]
    return t, index


def raster_plot(ts, sp_matrix, ax=None, marker='.', markersize=2, color='k', xlabel='Time (ms)', ylabel='Neuron index',
                xlim=None, ylim=None, title=None, show=False, **kwargs):
    """
        Raster plot to visualize spikes easily
    """

    sp_matrix = np.asarray(sp_matrix)
    if ts is None:
        raise
    ts = np.asarray(ts)

    # get index and time
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    time = ts[elements[0]]

    # plot raster
    if ax is None:
        ax = plt
    ax.plot(time, index, marker + color, markersize=markersize, **kwargs)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if title:
        plt.title(title)

    if show:
        plt.show()


def weight_to_conductance(weight, Rv, gain, Vread, Vth):
    return weight * Vth / Vread / Rv / gain


def conductance_to_weight(conductance, Rv, gain, Vread, Vth):
    return conductance / Vth * Vread * Rv * gain


def to_2T2R_mapping_HRS_LRS(conductance, g_max, g_min, g_upper, g_lower):
    g_pos = np.where(conductance >= 0.,
                     np.where(conductance <= g_lower - g_min,
                              conductance + g_min,
                              np.where(conductance <= g_max - g_upper,
                                       conductance + g_upper,
                                       np.where(g_upper - g_lower <= conductance,
                                                np.where(conductance <= g_max - g_min,
                                                         np.where(conductance <= g_max - g_lower,
                                                                  conductance + g_lower,
                                                                  conductance + g_min
                                                                  ),
                                                         np.ones_like(conductance) * -999.
                                                         ),
                                                np.ones_like(conductance) * -999.
                                                )
                                       )
                              ),
                     np.where(conductance >= g_min - g_lower,
                              np.ones_like(conductance) * g_min,
                              np.where(conductance >= g_upper - g_max,
                                       np.ones_like(conductance) * g_upper,
                                       np.where(g_min - g_max <= conductance,
                                                np.where(conductance <= g_lower - g_upper,
                                                         np.where(- conductance <= g_max - g_lower,
                                                                  np.ones_like(conductance) * g_lower,
                                                                  np.ones_like(conductance) * g_min
                                                                  ),
                                                         np.ones_like(conductance) * -999.
                                                         ),
                                                np.ones_like(conductance) * -999.
                                                )
                                       )
                              )
                     )
    g_neg = np.where(conductance >= 0.,
                     np.where(conductance <= g_lower - g_min,
                              np.ones_like(conductance) * g_min,
                              np.where(conductance <= g_max - g_upper,
                                       np.ones_like(conductance) * g_upper,
                                       np.where(g_upper - g_lower <= conductance,
                                                np.where(conductance <= g_max - g_min,
                                                         np.where(conductance <= g_max - g_lower,
                                                                  np.ones_like(conductance) * g_lower,
                                                                  np.ones_like(conductance) * g_min
                                                                  ),
                                                         np.ones_like(conductance) * -999.
                                                         ),
                                                np.ones_like(conductance) * -999.
                                                )
                                       )
                              ),
                     np.where(conductance >= g_min - g_lower,
                              - conductance + g_min,
                              np.where(conductance >= g_upper - g_max,
                                       - conductance + g_upper,
                                       np.where(g_min - g_max <= conductance,
                                                np.where(conductance <= g_lower - g_upper,
                                                         np.where(- conductance <= g_max - g_lower,
                                                                  - conductance + g_lower,
                                                                  - conductance + g_min
                                                                  ),
                                                         np.ones_like(conductance) * -999.
                                                         ),
                                                np.ones_like(conductance) * -999.
                                                )
                                       )
                              )
                     )
    return g_pos, g_neg


def to_2T2R_mapping_LRS(conductance, g_max, g_min):
    g_pos = np.where(conductance >= 0.,
                     np.where(conductance >= g_max / 2,
                              conductance + g_min,
                              g_max
                              ),
                     np.where(conductance <= - g_max / 2,
                              g_min,
                              g_max + conductance
                              )
                     )
    g_neg = np.where(conductance >= 0.,
                     np.where(conductance >= g_max / 2,
                              g_min,
                              g_max - conductance
                              ),
                     np.where(conductance <= - g_max / 2,
                              - conductance + g_min,
                              g_max
                              )
                     )
    return g_pos, g_neg


def add_write_dG(conductance, dG):
    return conductance + np.random.uniform(-dG, dG, conductance.shape)


def add_relaxation_dG(conductance, std):
    G = conductance + np.random.normal(0.0, std, conductance.shape)
    G = np.where(G < 0, 0, G)
    return G


def add_relaxation_dw(weight, std):
    return weight + np.random.normal(0.0, std, weight.shape)


def add_write_dADC(conductance, a, b, c, Vr, Vref, levels, dADC, bias=0):
    ADC = a / (b + c / conductance) * Vr / Vref * levels + bias
    perturbed = ADC + np.random.uniform(-dADC, dADC, conductance.shape)
    return (a / perturbed * Vr / Vref * levels - b) / c

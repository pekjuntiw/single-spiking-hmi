import json
import math
import os
import shutil
import time

import torch
import torch.utils.data as data
import numpy as np
from spikingjelly.clock_driven import functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import utils
from utils import raster_plot, cerst_loss, msst_loss, get_target_spike_time, FirstSpikeTime, weight_to_conductance
from dataset import TTFSSEMG, loader
from model import SingleLayerSingleSpikeNet
import copy
from functools import partial

"""
ALL PARAMETERS SHOULD BE IN STANDARD S.I. UNITS TO PREVENT CONFUSION!
"""

# torch.autograd.set_detect_anomaly(True)
# torch.set_printoptions(profile="full")
torch.set_printoptions(edgeitems=3)
torch.set_printoptions(precision=2)
torch.set_printoptions(linewidth=300)


def main():
    device = 'cuda'

    FLAGS = {
        "dataset_dir": 'sEMG_wrist_4/generated',
        "torch_seed": np.random.randint(0, 2**31 - 1),
        "numpy_seed": np.random.randint(0, 2**31 - 1),

        "train_epoch": 30,
        "snapshots": [20, 40, 60, 80],
        "batch_size": 64,
        "train_size": 1024,
        "lr": 5e-2,
        "decay_step": 125,
        "decay_rate": 0.95,
        "loss_mode": 'ttfs',
        "weight_bump": True,
        "weight_bump_param": {
            'max_no_spike': [0.05],
            'weight_bump_exp': True,
            'initial_bump_value': 0.005
        },
        "true_time": 1.3e-5,
        "false_time": 2.2e-5,

        "num_in": 6,
        "num_out": 4,
        "weight_init": {"mean": [5], "std": [5]},
        "l2_reg": 0.001,

        "save_model": True,

        "dt": 0.1e-6,

        # VO2 LIF parameters
        "vth": 4.3,
        "vh": 1.,
        "Rh": 15.5e3,
        "Rs": 500.,
        "tau": 1.6e-05,
        "taus": 2e-6,
        "input_scaling": 5e-5,
        "input_limit": 100e-3,
        "weight_scaling": 2.6875e-4,  # scale = vth / R
        "cv_c2c": 0.0,
        "v_mode": False,
        "v_mode_param": {"Rl": 5000., "beta": 1e6, "gamma": 0.3, "weight_scaling_2": 0.040},
        "input_width": 3e-6,
        "nonideal_input_tau": None,
    }

    if 'torch_seed' in FLAGS.keys() and not FLAGS['torch_seed'] is None:
        torch.manual_seed(FLAGS['torch_seed'])
    if 'numpy_seed' in FLAGS.keys() and not FLAGS['numpy_seed'] is None:
        np.random.seed(FLAGS['numpy_seed'])

    cv_c2c = 0.0 if "cv_c2c" not in FLAGS.keys() else FLAGS["cv_c2c"]

    save_model = FLAGS["save_model"]
    snapshots = [] if "snapshots" not in FLAGS.keys() else FLAGS["snapshots"]
    dataset_dir = 'sEMG_wrist/generated' if "dataset_dir" not in FLAGS.keys() else FLAGS["dataset_dir"]
    print(f'Using dataset from {dataset_dir}')
    model_num = 310
    model_dir = f'ttfs_semg/{model_num}'
    model_output_dir = f'{model_dir}/model'
    log_dir = f'{model_dir}/log'
    fig_dir = f'{model_dir}/fig'
    model_output_name = f'{model_output_dir}/ttfs_semg'  # to be completed below

    # set if resume training from a saved model
    resume_training = False
    resume_from_epoch = 35

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # save flags
    if save_model:
        with open(f'{model_dir}/flags.json', 'w') as fp:
            json.dump(FLAGS, fp, sort_keys=True, indent=4)

    batch_size = FLAGS["batch_size"]
    train_size = FLAGS["train_size"]
    train_epoch = FLAGS["train_epoch"]
    lr = FLAGS["lr"]
    decay_step = FLAGS["decay_step"]
    decay_rate = FLAGS["decay_rate"]
    loss_mode = FLAGS["loss_mode"]
    weight_bump = FLAGS["weight_bump"]
    weight_bump_param = FLAGS["weight_bump_param"]
    true_time = FLAGS["true_time"]
    false_time = FLAGS["false_time"]

    num_in = FLAGS["num_in"]
    num_out = FLAGS["num_out"]
    weight_init = FLAGS["weight_init"]
    l2_reg = FLAGS["l2_reg"]

    dt = FLAGS["dt"]

    vth = FLAGS["vth"]
    vh = FLAGS["vh"]
    Rh = FLAGS["Rh"]
    Rs = FLAGS["Rs"]
    tau = FLAGS["tau"]
    taus = FLAGS["taus"]
    input_limit = FLAGS["input_limit"]
    v_mode = FLAGS["v_mode"]
    v_mode_param = FLAGS["v_mode_param"]
    weight_scaling = FLAGS["weight_scaling"] * v_mode_param["weight_scaling_2"]
    input_width = FLAGS["input_width"]
    nonideal_input_tau = FLAGS["nonideal_input_tau"]

    writer = SummaryWriter(log_dir)

    # initialize dataloader
    dataset = TTFSSEMG(path=dataset_dir, dt=dt,
                       transform=partial(utils.apply_noise, mean=0., std=0., scale=tau))
    test_size = len(dataset) - train_size
    test_batch_size = batch_size * 1
    train_batch_per_epoch = int(train_size / batch_size)
    test_batch_per_epoch = math.ceil(test_size / test_batch_size)
    train_dataset, test_dataset = data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(20231214)
    )
    train_data_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False,
    )

    # initialize VO2-based LSNN model
    net = SingleLayerSingleSpikeNet(
        num_in=num_in, num_out=num_out,
        tau=tau, R=(Rh + Rs), v_threshold=vth, v_reset=vh, cv_c2c=cv_c2c,
        weight_init=weight_init, weight_scaling=weight_scaling,
        weight_bump=weight_bump_param,
        taus=taus,
        dt=dt, device=device,
        input_limit=input_limit,
        vin_mode=v_mode, Rl=v_mode_param["Rl"], beta=v_mode_param["beta"], gamma=v_mode_param["gamma"],
        input_width=input_width, nonideal_input_tau=nonideal_input_tau
    )
    net = net.to(device)
    net_initial = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

    # check size of model
    mem_params = sum([param.nelement() * param.element_size() for param in net.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in net.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print(net)
    print(f"Params: {mem_params}bytes      Bufs: {mem_bufs}bytes      Total: {mem}bytes")
    if device == 'cuda':
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # load save model if resume training
    if resume_training:
        print(f'Loading model, scheduler and optimizer from {model_output_name}_ep{resume_from_epoch}.ckpt')
        chkpt = torch.load(f'{model_output_name}_ep{resume_from_epoch}.ckpt')
        net.load_state_dict(chkpt["net"])
        optimizer.load_state_dict(chkpt["optimizer"])
        scheduler.load_state_dict(chkpt["scheduler"])

    test_accs = np.load(f'{model_output_dir}/test_accs_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_accs = np.load(f'{model_output_dir}/train_accs_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_loss = np.load(f'{model_output_dir}/test_loss_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_loss = np.load(f'{model_output_dir}/train_loss_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_accs_step = np.load(f'{model_output_dir}/test_accs_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_accs_step = np.load(f'{model_output_dir}/train_accs_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_loss_step = np.load(f'{model_output_dir}/test_loss_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_loss_step = np.load(f'{model_output_dir}/train_loss_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []

    fc1_grad = np.load(f'{model_output_dir}/grad_fc1_ep{resume_from_epoch}.npy').tolist() if resume_training else []

    val_spike_times_mean = np.load(
        f'{model_output_dir}/val_spike_times_mean_ep{resume_from_epoch}.npy'
    ).tolist() if resume_training else [[] for _ in range(num_out)]
    val_spike_times_std = np.load(
        f'{model_output_dir}/val_spike_times_std_ep{resume_from_epoch}.npy'
    ).tolist() if resume_training else [[] for _ in range(num_out)]
    val_spike_times_q25 = np.load(
        f'{model_output_dir}/val_spike_times_q25_ep{resume_from_epoch}.npy'
    ).tolist() if resume_training else [[] for _ in range(num_out)]
    val_spike_times_q75 = np.load(
        f'{model_output_dir}/val_spike_times_q75_ep{resume_from_epoch}.npy'
    ).tolist() if resume_training else [[] for _ in range(num_out)]

    train_fc1_spike_ratio = np.load(f'{model_output_dir}/ratio_out_ep{resume_from_epoch}.npy').tolist() if resume_training else []

    train_times = len(train_loss_step) if resume_training else 0
    test_times = len(test_loss_step) if resume_training else 0
    max_test_accuracy = max(test_accs) if resume_training else 0
    confusion_matrix = np.zeros([num_out, num_out], dtype=int)

    initial_fc1 = np.load(f'{model_dir}/weights_initial_fc1.npy') if resume_training else net.fc1.weight.data.cpu().numpy()
    np.save(f'{model_dir}/weights_initial_fc1.npy', initial_fc1)
    fc1_stats_step = np.load(f'{model_output_dir}/stats_fc1_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    bumps = np.load(f'{model_output_dir}/bumps_ep{resume_from_epoch}.npy').tolist() if resume_training else []

    NAN_STOP_FLAG = False

    start_epoch = resume_from_epoch + 1 if resume_training else 0

    t_start = time.perf_counter()
    for epoch in range(start_epoch, train_epoch):
        print(f"Epoch {epoch}: lr={scheduler.get_last_lr()}")
        train_correct_sum = 0
        train_sum = 0
        train_loss_batches = []

        # train model
        net.train()

        # set use transform
        dataset.use_transform = True

        # get data in batches
        for i, (fingerprint, label) in enumerate(
                pbar := tqdm(loader(train_data_loader, device), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                             total=train_batch_per_epoch, desc="Training", ascii=" >>>>=")):
            optimizer.zero_grad()

            # forward pass
            output = net(fingerprint.float())
            # output.shape = [timestep, batch_size, n_out]
            # take output neuron that fires first as the classification result
            results = FirstSpikeTime.apply(output, dt)
            # results.shape = [batch_size, n_out]

            # compute loss, backward pass (BPTT)
            if loss_mode == 'ttfs':
                loss = cerst_loss(results, label, 0.2, tau, [0.005, 1.])
            else:
                target = get_target_spike_time(label, num_out, true_time, false_time)
                loss = msst_loss(results, target, tau)

            train_fc1_spike_ratio.append((net.out.spiked.sum() / net.out.spiked.numel()).item())
            weight_bumped_layer = net.bump_weights() if weight_bump else -2
            if weight_bumped_layer != -2:
                # not doing backward
                bumps.append([train_times, weight_bumped_layer])
            else:
                loss.backward()
                optimizer.step()

            w1_f = net.fc1.weight.data.cpu().numpy()
            fc1_stats_step.append([w1_f.mean(), w1_f.std(), np.quantile(w1_f, 0.25), np.quantile(w1_f, 0.5), np.quantile(w1_f, 0.75)])

            scheduler.step()

            # reset SNN state after each forward pass
            functional.reset_net(net)

            # use argmin over output neuron dimension to find the neuron that fired the earliest
            results = results.argmin(1)
            is_correct = (results == label).float()
            train_correct_sum += is_correct.sum().item()
            train_sum += label.numel()
            train_batch_accuracy = is_correct.mean().item()
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            train_accs_step.append(train_batch_accuracy)

            train_loss_batches.append(loss.item())
            train_loss_step.append(loss.item())
            writer.add_scalar('train_batch_loss', loss, train_times)

            fc1_grad_norm = torch.nan if weight_bumped_layer != -2 else torch.linalg.norm(net.fc1.weight.grad).item()
            fc1_grad.append(fc1_grad_norm)

            pbar.set_postfix_str(
                f'    (Step {train_times}: acc={train_batch_accuracy * 100:.4f}, loss={loss.item():.4f})    ({i}a:  {fc1_grad_norm:.6f})'
            )

            train_times += 1

            with torch.no_grad():
                if torch.isnan(net.fc1.weight.data).any():
                    NAN_STOP_FLAG = True
                    print('\nNaN in weights in current batch, stopping current epoch!')
                    break

            if i % 20 == 0:
                fig, ax = plt.subplots(1, 1)
                ax1 = ax.twinx()
                ax1.plot(np.arange(train_times) / train_batch_per_epoch, train_loss_step, color='tab:green', alpha=0.4)
                ax1.plot(np.arange(test_times) / test_batch_per_epoch, test_loss_step, color='tab:red', alpha=0.6)
                if loss_mode == 'ttfs':
                    ax1.set_ylim(0.1, 2)
                else:
                    ax1.set_ylim(0.01, 1.5)
                ax1.set_yscale('log')
                ax1.set_ylabel("train loss")
                ax.plot(np.arange(train_times) / train_batch_per_epoch, train_accs_step, color='tab:blue', alpha=0.4)
                ax.plot(np.arange(test_times) / test_batch_per_epoch, test_accs_step, color='tab:orange', alpha=0.6)
                ax.set_ylim(0.05, 1.05)
                ax.set_ylabel("train accs")
                ax.set_xlabel("epochs")
                fig.savefig('live_accuracy.png')
                print(f"\n===========Saved live accuracy plot @ epoch {epoch}")
                plt.close(fig)

                fig, ax = plt.subplots(2, 1)
                for b in bumps:
                    ax[b[1]].axvline(b[0] / train_batch_per_epoch, color="k", linestyle=":", alpha=0.4)
                l = ax[0].plot(np.arange(train_times) / train_batch_per_epoch, [s[0] for s in fc1_stats_step], '-.')
                ax[0].plot(np.arange(train_times) / train_batch_per_epoch, [s[3] for s in fc1_stats_step], color=l[0].get_color())
                ax[0].fill_between(np.arange(train_times) / train_batch_per_epoch, [s[2] for s in fc1_stats_step], [s[4] for s in fc1_stats_step], alpha=0.4)
                fig.savefig('live_weights.png')
                plt.close(fig)
                fig, ax = plt.subplots(1, 1)
                ax.plot(np.arange(train_times) / train_batch_per_epoch, train_fc1_spike_ratio)
                fig.savefig('live_spike_stats.png')
                plt.close(fig)

        train_accuracy = train_correct_sum / train_sum
        train_loss_avg = np.mean(train_loss_batches)
        train_accs.append(train_accuracy)
        train_loss.append(train_loss_avg)

        # test model
        net.eval()

        # reset use transform
        dataset.use_transform = False

        with torch.no_grad():
            test_correct_sum = 0
            test_sum = 0
            test_loss_batches = []
            val_spike_times_per_batch = []
            val_labels_per_batch = []

            # get data in batches
            for fingerprint, label in tqdm(loader(test_data_loader, device),
                                           bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", total=test_batch_per_epoch,
                                           desc="Testing", ascii=" >>>>="):
                # forward pass
                output = net(fingerprint.float())
                # take output neuron that fires first as the classification result
                results = FirstSpikeTime.apply(output, dt)
                val_spike_times_per_batch.append(results.clone().detach())
                val_labels_per_batch.append(label.clone().detach())

                # compute loss
                if loss_mode == 'ttfs':
                    loss = cerst_loss(results, label, 0.2, tau, [0.005, 1.])
                else:
                    target = get_target_spike_time(label, num_out, true_time, false_time)
                    loss = msst_loss(results, target, tau)

                # reset SNN state after each forward pass
                functional.reset_net(net)

                # use argmin over output neuron dimension to find the neuron that fired the earliest
                results = results.argmin(1)
                is_correct = (results == label).float()
                test_correct_sum += is_correct.sum().item()
                test_sum += label.numel()
                test_accs_step.append(is_correct.mean().item())

                # test_loss_sum += loss.item()
                test_loss_batches.append(loss.item())
                test_loss_step.append(loss.item())
                writer.add_scalar('test_batch_loss', loss, test_times)

                test_times += 1

            test_accuracy = test_correct_sum / test_sum
            test_loss_avg = np.mean(test_loss_batches)
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            test_loss.append(test_loss_avg)

            val_spike_times_per_batch = torch.cat(val_spike_times_per_batch, dim=0)
            val_labels_per_batch = torch.cat(val_labels_per_batch, dim=0)
            val_spike_times_mean, val_spike_times_std, val_spike_times_q25, val_spike_times_q75 = \
                utils.get_spike_time_stats(
                    val_spike_times_per_batch, val_labels_per_batch,
                    val_spike_times_mean, val_spike_times_std, val_spike_times_q25, val_spike_times_q75
                )

        # save model if test accuracy is improved or is final epoch or is snapshot
        if save_model and (test_accuracy >= max_test_accuracy or epoch == train_epoch - 1 or epoch in snapshots):
            if test_accuracy >= max_test_accuracy:
                print(
                    f'Improved. Saving net, scheduler state and optimizer state to {model_output_name}_ep{epoch}.ckpt')
            elif epoch == train_epoch - 1:
                print(
                    f'Final epoch. Saving net, scheduler state and optimizer state to {model_output_name}_ep{epoch}.ckpt')
            else:
                print(
                    f'Snapshot. Saving net, scheduler state and optimizer state to {model_output_name}_ep{epoch}.ckpt')
            chkpt = {
                "net": net.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(chkpt, f'{model_output_name}_ep{epoch}.ckpt')

            print(f'Saving losses and accuracies to {model_output_dir}')
            np.save(f'{model_output_dir}/train_accs_ep{epoch}.npy', np.array(train_accs))
            np.save(f'{model_output_dir}/test_accs_ep{epoch}.npy', np.array(test_accs))
            np.save(f'{model_output_dir}/train_loss_ep{epoch}.npy', np.array(train_loss))
            np.save(f'{model_output_dir}/test_loss_ep{epoch}.npy', np.array(test_loss))
            np.save(f'{model_output_dir}/train_accs_step_ep{epoch}.npy', np.array(train_accs_step))
            np.save(f'{model_output_dir}/test_accs_step_ep{epoch}.npy', np.array(test_accs_step))
            np.save(f'{model_output_dir}/train_loss_step_ep{epoch}.npy', np.array(train_loss_step))
            np.save(f'{model_output_dir}/test_loss_step_ep{epoch}.npy', np.array(test_loss_step))

            print(f'Saving gradients to {model_output_dir}')
            np.save(f'{model_output_dir}/grad_fc1_ep{epoch}.npy', np.array(fc1_grad))

            np.save(f'{model_output_dir}/val_spike_times_mean_ep{epoch}.npy', np.array(val_spike_times_mean))
            np.save(f'{model_output_dir}/val_spike_times_std_ep{epoch}.npy', np.array(val_spike_times_std))
            np.save(f'{model_output_dir}/val_spike_times_q25_ep{epoch}.npy', np.array(val_spike_times_q25))
            np.save(f'{model_output_dir}/val_spike_times_q75_ep{epoch}.npy', np.array(val_spike_times_q75))

            np.save(f'{model_output_dir}/ratio_out_ep{epoch}.npy', np.array(train_fc1_spike_ratio))
            np.save(f'{model_output_dir}/stats_fc1_ep{epoch}.npy', np.array(fc1_stats_step))
            np.save(f'{model_output_dir}/bumps_ep{epoch}.npy', np.array(bumps))

        max_test_accuracy = max(max_test_accuracy, test_accuracy)

        print(
            f"Epoch {epoch}: train_acc = {train_accuracy}, test_acc={test_accuracy}, train_loss_avg={train_loss_avg}, test_loss_avg={test_loss_avg}, max_test_acc={max_test_accuracy}, train_times={train_times}")

        if NAN_STOP_FLAG:
            print('NaN in weights, abort training!')
            break

        print()

    t_end = time.perf_counter()
    duration = t_end - t_start
    if train_epoch - start_epoch > 1:
        print(
            f"\nTotal training time {duration:.3f} s. Average {(duration / (train_times / train_batch_per_epoch - start_epoch)):.3f} s/epoch.\n")

    # run a final validation
    net.eval()
    # reset use transform
    dataset.use_transform = False
    with torch.no_grad():
        val_spike_times_per_batch = []
        val_labels_per_batch = []
        energy_per_sample = []
        spikes_per_sample = []

        # get data in batches
        for fingerprint, label in tqdm(loader(test_data_loader, device), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                                       total=test_batch_per_epoch, desc="Testing"):
            # forward pass
            output = net(fingerprint.float())
            results = FirstSpikeTime.apply(output, dt)
            val_spike_times_per_batch.append(results.clone().detach())
            val_labels_per_batch.append(label.clone().detach())
            energy_per_sample.append(net.p_stat()[2].cpu().numpy())
            spikes_per_sample.append(net.s_stat()[2].cpu().numpy())
            functional.reset_net(net)

            results = results.argmin(1)
            # calculate confusion matrix
            for target_label, predicted_label in zip(label.cpu().numpy(), results.cpu().numpy()):
                confusion_matrix[predicted_label, target_label] += 1

        val_spike_times_per_batch = torch.cat(val_spike_times_per_batch, dim=0)
        val_labels_per_batch = torch.cat(val_labels_per_batch, dim=0)
        output_spikes_sorted_earliest, output_spikes_sorted_truth = utils.sort_spike_times(
            val_spike_times_per_batch, val_labels_per_batch
        )
        energy_per_sample = [element for batch in energy_per_sample for element in batch]
        spikes_per_sample = [element for batch in spikes_per_sample for element in batch]

        print(f'Energy/sample: mean = {np.mean(energy_per_sample):.3e} J, std = {np.std(energy_per_sample):.3e} J, max = {np.max(energy_per_sample):.3e} J, min = {np.min(energy_per_sample):.3e} J')
        print(f'Spikes/sample: mean = {np.mean(spikes_per_sample):.3f}, std = {np.std(spikes_per_sample):.3f}, max = {np.max(spikes_per_sample):.3f}, min = {np.min(spikes_per_sample):.3f}')

    # plot some figures
    net.eval()
    with torch.no_grad():
        fingerprint, label = test_dataset[1]
        fingerprint = torch.unsqueeze(fingerprint, dim=0)
        fingerprint = fingerprint.to(device)

        # forward pass
        net(fingerprint.float())

        # get model dynamical states
        v_evol_1 = net.v_1.cpu().numpy()
        s_evol_1 = net.spike_1.cpu().numpy()
        x_evol_1 = net.neuron_in_1.cpu().numpy()
        p_evol_1 = net.p_1.cpu().numpy()

        try:
            plt.figure()
        except RuntimeError:
            plt.switch_backend('Agg')
        finally:
            plt.close()

        plt.figure()
        plt.plot(np.arange(v_evol_1.shape[0]), v_evol_1)
        plt.title("Vmem Out")
        plt.savefig(f'{fig_dir}/vmem-out.png')
        plt.figure()
        plt.plot(np.arange(x_evol_1.shape[0]), x_evol_1)
        plt.title("Input Out")
        plt.savefig(f'{fig_dir}/input-out.png')
        plt.figure()
        plt.plot(np.arange(p_evol_1.shape[0]), p_evol_1)
        plt.title("Power Out")
        plt.savefig(f'{fig_dir}/power-out.png')
        plt.figure()
        raster_plot(np.arange(s_evol_1.shape[0]), s_evol_1, show=False, xlim=(-10, s_evol_1.shape[0] + 10))
        plt.title("Spike Raster Out")
        plt.savefig(f'{fig_dir}/spike-raster-out.png')
        plt.figure()
        sns.heatmap(net.fc1.weight.data.cpu().numpy(), square=False, center=0, cmap='vlag')
        plt.gca().invert_yaxis()
        plt.title("Weights fc1")
        plt.savefig(f'{fig_dir}/weights.png')
        plt.figure()
        sns.heatmap(
            weight_to_conductance(net.fc1.weight.data.cpu().numpy(), Rh + Rs, 1200, 0.1, vth) * 1000000,
            square=False, center=0, cmap='vlag',
            annot=True, fmt=".3f", cbar_kws={'label': 'Conductance (uS)'}
        )
        plt.gca().invert_yaxis()
        plt.title("Conductance fc1")
        plt.savefig(f'{fig_dir}/conductance.png')
        np.savetxt(f'{model_dir}/conductance_TRAINED.csv', weight_to_conductance(net.fc1.weight.data.cpu().numpy(), Rh + Rs, 1200, 0.1, vth) * 1000000, '%.8f', ',')
        plt.figure()
        sns.heatmap(
            1 / weight_to_conductance(net.fc1.weight.data.cpu().numpy(), Rh + Rs, 1200, 0.1, vth) / 1000,
            square=False, center=0, cmap='vlag',
            annot=True, fmt=".3f", cbar_kws={'label': 'Resistance (kohm)'}
        )
        plt.gca().invert_yaxis()
        plt.title("Resistance fc1")
        plt.figure()
        plt.hist(initial_fc1.flatten(), alpha=0.4, label='initial')
        plt.hist(net.fc1.weight.data.cpu().numpy().flatten(), alpha=0.6, label='trained')
        plt.legend()
        plt.savefig(f'{fig_dir}/weights_hist.png')

    shutil.copy('live_accuracy.png', f'{model_dir}/live_accuracy.png')
    shutil.copy('live_spike_stats.png', f'{model_dir}/live_spike_stats.png')
    shutil.copy('live_weights.png', f'{model_dir}/live_weights.png')

    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    if save_model:
        np.save(f'{model_dir}/train_accs.npy', train_accs)
        np.save(f'{model_dir}/test_accs.npy', test_accs)
        np.save(f'{model_dir}/train_loss.npy', train_loss)
        np.save(f'{model_dir}/test_loss.npy', test_loss)

    plt.figure()
    plt.plot(np.arange(train_accs.shape[0]), train_accs, label="train")
    plt.plot(np.arange(test_accs.shape[0]), test_accs, label="test")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(f'{fig_dir}/accs.png')
    plt.figure()
    plt.plot(np.arange(train_loss.shape[0]), train_loss, label="train")
    plt.plot(np.arange(test_loss.shape[0]), test_loss, label="test")
    plt.legend()
    plt.title("Loss")
    plt.savefig(f'{fig_dir}/loss.png')

    accs = confusion_matrix.trace() / confusion_matrix.sum()
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)
    plt.figure()
    sns.heatmap(
        confusion_matrix,
        annot=True, fmt=("d" if confusion_matrix.dtype == int else ".2%"), xticklabels=dataset.get_classes(False),
        yticklabels=dataset.get_classes(False)
    )
    plt.title(f"Test accuracy = {accs:.2%}")
    plt.xlabel("Target")
    plt.ylabel("Result")
    plt.gca().invert_yaxis()
    plt.yticks(rotation=0)
    plt.savefig(f'{fig_dir}/confusion.png')

    val_spike_times_mean = np.array(val_spike_times_mean)
    val_spike_times_std = np.array(val_spike_times_std)
    fig, axs = plt.subplots(num_out // 2 + 1, 2, figsize=(15, 15), sharex=True)
    for i in range(num_out):
        for j in range(num_out):
            ys = val_spike_times_mean[i, :, j]
            xs = np.arange(len(ys))
            stds = val_spike_times_std[i, :, j]
            l1 = axs[i // 2, i % 2].plot(xs, ys, label=f'neuron {j}')
            axs[i // 2, i % 2].fill_between(xs, ys - stds, ys + stds, alpha=0.4)
        axs[i // 2, i % 2].set_title(f'Class {i}')
        axs[i // 2, i % 2].set_ylabel('Time')
    axs[-1, 0].set_xlabel('Epoch')
    axs[-1, 1].set_xlabel('Epoch')
    plt.savefig(f'{fig_dir}/output_spike_times_stats.png')

    fig, axs = plt.subplots(num_out // 2 + 1, 2, figsize=(15, 15), sharex=True)
    for i in range(num_out):
        for j in range(num_out):
            ys = output_spikes_sorted_earliest[i][:, j]
            xs = np.arange(len(ys))
            axs[i // 2, i % 2].plot(xs, ys, label=f'neuron {j}', alpha=(1. if i == j else 0.5))
        axs[i // 2, i % 2].set_ylim(0.2 * tau, 3. * tau)
        axs[i // 2, i % 2].set_ylabel('Time')
        axs[i // 2, i % 2].set_title(f'Class {i}')
    axs[-1, 0].set_xlabel('Sample number sorted by earliest output spike time')
    axs[-1, 1].set_xlabel('Sample number sorted by earliest output spike time')
    plt.savefig(f'{fig_dir}/sorted_output_spike_times_earliest.png')
    fig, axs = plt.subplots(num_out // 2 + 1, 2, figsize=(15, 15), sharex=True)
    for i in range(num_out):
        for j in range(num_out):
            ys = output_spikes_sorted_truth[i][:, j]
            xs = np.arange(len(ys))
            axs[i // 2, i % 2].plot(xs, ys, label=f'neuron {j}', alpha=(1. if i == j else 0.5))
        axs[i // 2, i % 2].set_ylim([0.2 * tau, 3. * tau])
        axs[i // 2, i % 2].set_ylabel('Time')
        axs[i // 2, i % 2].set_title(f'Class {i}')
    axs[-1, 0].set_xlabel('Sample number sorted by ground truth output spike time')
    axs[-1, 1].set_xlabel('Sample number sorted by ground truth output spike time')
    plt.savefig(f'{fig_dir}/sorted_output_spike_times_truth.png')

    plt.figure()
    plt.plot(np.arange(len(fc1_grad)), fc1_grad, label="FC1 grad")
    plt.yscale('log')
    plt.title('Linear grad')
    plt.legend()
    plt.savefig(f'{fig_dir}/grad.png')

    plt.show()

    print(f'Done! Check model at {model_dir}/')


if __name__ == '__main__':
    main()

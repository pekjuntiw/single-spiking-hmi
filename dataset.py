import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import re


def loader(dataset, device):
    """
        Data generator
    """
    for data, label in dataset:
        data = data.to(device)
        label = label.to(device)
        yield data, label


class TTFSSEMG(data.dataset.Dataset):
    def __init__(self, path,
                 rv=16000., rl=8000., cm=4.9e-9, vth=3.8, min_vin=8., max_vin=13.,
                 dt=0.1e-6, T=50e-6, transform=None, use_transform=False):
        """
            Dataset for TTFS encoded sEMG acquired using commercial sensors

            path
            |  *.csv (1 csv file per class)
        :param path:
        """
        self.fs = [f'{path}/up.csv', f'{path}/down.csv', f'{path}/left.csv', f'{path}/right.csv']
        self.class_names = [re.split(r'[\\/.]', f)[-2] for f in self.fs]
        self.class_numeric = np.arange(len(self.class_names)).astype(int)
        self.cs = []
        self.vals_raw = []
        self.vals = []

        # Voltage-input single spike encoder parameters
        # TTS = tau * ln( (Vin*Rp/Rl) / (Vin*Rp/Rl - Vth) )
        self.rv = rv
        self.rl = rl
        self.rp = rv * rl / (rv + rl)
        self.cm = cm
        self.vth = vth
        self.min_vin = min_vin
        self.max_vin = max_vin

        self.dt = dt
        self.T = T
        self.transform = transform
        self.use_transform = use_transform

        for c, f in zip(self.class_numeric, self.fs):
            d = pd.read_csv(f).drop(columns=['timestamp']).to_numpy()
            self.vals_raw.append(d)
            self.cs.append(np.ones(len(d)) * c)
        self.vals_raw = np.concatenate(self.vals_raw)
        self.cs = np.concatenate(self.cs)

        self.vals = (max_vin - min_vin) * self.vals_raw / 750 + min_vin
        self.vals = self.vals * self.rp / self.rl
        self.vals = self.vals / (self.vals - self.vth)
        self.vals = self.rp * self.cm * np.log(self.vals)

        self.vals = torch.Tensor(self.vals)
        self.cs = torch.Tensor(self.cs).to(dtype=torch.int)

    def __len__(self):
        return len(self.cs)

    def __getitem__(self, index):
        sample = (self.vals[index].clone(), self.cs[index])
        if self.transform is not None and self.use_transform:
            sample = self.transform(sample)
        sample = self.encode(sample)
        return sample

    def get_classes(self, numeric=True):
        return self.class_numeric if numeric else self.class_names

    def encode(self, sample):
        vals, cs = sample
        vals /= self.dt
        vals = torch.round(vals).to(dtype=torch.int)
        timestep = int(self.T / self.dt)
        raster = torch.zeros([timestep, len(vals)])
        for j, t in enumerate(vals):
            raster[t, j] = 1.
        return raster, cs


class TTFSFingerprint(data.dataset.Dataset):
    def __init__(self, path,
                 rv=16000., rl=8000., cm=4.9e-9, vth=3.8, min_vin=6.83, max_vin=18., tts_bias=-7e-6,
                 dt=0.1e-6, T=55e-6, input_bias_units=None, input_bias_times=None, transform=None, use_transform=False,
                 invert=False,
                 min_t=10e-6, max_t=47e-6,
                 linear_mapping=True,
                 cv_d2d_vth=0., cv_d2d_rv=0., cv_d2d_tts=0.):
        """
            Dataset for TTFS encoded flattened fingerprint image

            path
            |  10*_encoded.npy (1 npy file per subject/class)
        :param path:
        """
        self.subjects = np.load(f'{path}/subjects_selected.npy')
        self.classes = np.arange(len(self.subjects)).astype(int)
        self.dataset = np.load(f'{path}/dataset.npy')
        self.cs = []
        self.vals = []
        self.class_names = self.subjects

        self.class_numeric = np.unique(self.cs)
        self.class_names = self.class_numeric

        # Voltage-input single spike encoder parameters
        # TTS = tau * ln( (Vin*Rp/Rl) / (Vin*Rp/Rl - Vth) )
        # with d2d variations
        self.cv_d2d_vth = cv_d2d_vth
        self.cv_d2d_rv = cv_d2d_rv
        self.n_channel = self.dataset.shape[-2] * self.dataset.shape[-1]
        self.vth = np.random.normal(vth, cv_d2d_vth * vth, self.n_channel)
        self.vth = np.clip(self.vth, 0.25 * vth, None)  # prevent vth from being unrealistically low
        self.rv = np.random.normal(rv, cv_d2d_rv * rv, self.n_channel)
        self.rv = np.clip(self.rv, 0.25 * rv, None)  # prevent rv from being unrealistically low
        self.rl = rl
        self.rp = self.rv * rl / (self.rv + rl)
        rp_ideal = rv * rl / (rv + rl)
        self.cm = cm
        self.min_vin = min_vin
        self.max_vin = max_vin
        self.tts_bias = tts_bias
        self.min_t = min_t
        self.max_t = max_t

        self.dt = dt
        self.T = T

        self.input_bias_units = 0 if input_bias_units is None else input_bias_units
        self.input_bias_times = input_bias_times

        self.transform = transform
        self.use_transform = use_transform

        for n_cat, category in enumerate(self.dataset):
            for n_data, img in enumerate(category):
                self.cs.append(n_cat)
                img_flat = img.flatten()
                if invert:
                    img_flat = img_flat * (-1.) + 1.  # make invert white and black

                if linear_mapping:
                    # nonlinear mapping of the input, such that the final tts is linear
                    img_flat = vth / (rp_ideal / rl) / (1 - np.exp(-((max_t - min_t) * img_flat + min_t)/(rp_ideal * cm)))
                else:
                    img_flat = (max_vin - min_vin) * (img_flat - np.min(img_flat)) / (np.max(img_flat) - np.min(img_flat)) + min_vin

                img_flat = img_flat * self.rp / self.rl  # rp will be broadcasted up
                denominator = (img_flat - self.vth)  # vth will be broadcasted up
                img_flat = np.where(denominator <= 0, 9999999, img_flat / denominator)  # in case of invalid values (too large vth)
                img_flat = self.rp * self.cm * np.log(img_flat) + tts_bias
                self.vals.append(img_flat)
        self.vals = np.array(self.vals)

        # d2d variations related to CMOS process
        self.cv_d2d_tts = cv_d2d_tts
        self.variation_tts = np.random.normal(0, cv_d2d_tts, self.n_channel)  # before scaling, cv = sigma at mu = 1
        self.vals = self.vals * (1 + self.variation_tts)  # scale the tts, variation_tts will be broadcasted up

        self.vals = torch.Tensor(self.vals)
        self.cs = torch.Tensor(self.cs).to(dtype=torch.int)

    def __len__(self):
        return len(self.cs)

    def __getitem__(self, index):
        sample = (self.vals[index].clone(), self.cs[index])
        if self.transform is not None and self.use_transform:
            sample = self.transform(sample)
        sample = self.encode(sample)
        return sample

    def get_classes(self, numeric=True):
        return self.class_numeric if numeric else self.class_names

    def encode(self, sample):
        vals, cs = sample
        vals /= self.dt
        vals = torch.round(vals).to(dtype=torch.int)
        timestep = int(self.T / self.dt)
        raster = torch.zeros([timestep, len(vals) + self.input_bias_units])
        for j, t in enumerate(vals):
            if t >= timestep:
                continue
            raster[t, j] = 1.
        if self.input_bias_units > 0:
            if self.input_bias_times is None:
                raise ValueError('input_bias_times must be provided when input_bias_units > 0')
            if len(self.input_bias_times) != self.input_bias_units:
                raise ValueError('input_bias_times must be the same length as input_bias_units')
            for j, t in enumerate(self.input_bias_times):
                if isinstance(t, list):
                    t = np.array(t)
                t = np.round(t / self.dt).astype(int)
                raster[t, j + len(vals)] = 1.
        return raster, cs


class TTFSSEMGCapgmyo(data.dataset.Dataset):
    def __init__(self, which, path,
                 rv=16000., rl=8000., cm=4.9e-9, vth=3.8, min_vin=7., max_vin=18.,
                 dt=0.1e-6, T=80e-6, input_bias_units=None, input_bias_times=None, transform=None, use_transform=False,
                 invert=False,
                 cv_d2d_vth=0., cv_d2d_rv=0., cv_d2d_tts=0.):
        """
            Dataset for TTFS encoded Capgmyo sEMG

            path
            |  train_dataset.csv
            |  train_label.csv
            |  test_dataset.csv
            |  test_label.csv
        :param path:
        """
        assert which in ['train', 'test']
        self.vals_raw = np.loadtxt(f'{path}/{which}_dataset.csv', delimiter=',')
        self.cs = np.loadtxt(f'{path}/{which}_label.csv', dtype=int, delimiter=',')

        self.class_numeric = np.unique(self.cs)
        self.class_names = self.class_numeric

        # Voltage-input single spike encoder parameters
        # TTS = tau * ln( (Vin*Rp/Rl) / (Vin*Rp/Rl - Vth) )
        # with d2d variations
        self.cv_d2d_vth = cv_d2d_vth
        self.cv_d2d_rv = cv_d2d_rv
        self.n_channel = self.vals_raw.shape[1]
        self.vth = np.random.normal(vth, cv_d2d_vth * vth, self.n_channel)
        self.vth = np.clip(self.vth, 0.25 * vth, None)  # prevent vth from being unrealistically low
        self.rv = np.random.normal(rv, cv_d2d_rv * rv, self.n_channel)
        self.rv = np.clip(self.rv, 0.25 * rv, None)  # prevent rv from being unrealistically low
        self.rl = rl
        self.rp = self.rv * rl / (self.rv + rl)
        self.cm = cm
        self.min_vin = min_vin
        self.max_vin = max_vin

        self.dt = dt
        self.T = T

        self.input_bias_units = 0 if input_bias_units is None else input_bias_units
        self.input_bias_times = input_bias_times

        self.transform = transform
        self.use_transform = use_transform

        self.min_vin = max_vin if invert else min_vin
        self.max_vin = min_vin if invert else max_vin
        self.vals = (self.max_vin - self.min_vin) * (self.vals_raw - np.min(self.vals_raw)) / (np.max(self.vals_raw) - np.min(self.vals_raw)) + self.min_vin
        self.vals = self.vals * self.rp / self.rl  # rp will be broadcasted up
        denominator = (self.vals - self.vth)  # vth will be broadcasted up
        self.vals = np.where(denominator <= 0, 9999999, self.vals / denominator)  # in case of invalid values (too large vth)
        self.vals = self.rp * self.cm * np.log(self.vals)

        # d2d variations related to CMOS process
        self.cv_d2d_tts = cv_d2d_tts
        self.variation_tts = np.random.normal(0, cv_d2d_tts, self.n_channel)  # before scaling, cv = sigma at mu = 1
        self.vals = self.vals * (1 + self.variation_tts)  # scale the tts, variation_tts will be broadcasted up

        self.vals = torch.Tensor(self.vals)
        self.cs = torch.Tensor(self.cs).to(dtype=torch.int)

    def __len__(self):
        return len(self.cs)

    def __getitem__(self, index):
        sample = (self.vals[index].clone(), self.cs[index])
        if self.transform is not None and self.use_transform:
            sample = self.transform(sample)
        sample = self.encode(sample)
        return sample

    def get_classes(self, numeric=True):
        return self.class_numeric if numeric else self.class_names

    def encode(self, sample):
        vals, cs = sample
        vals /= self.dt
        vals = torch.round(vals).to(dtype=torch.int)
        timestep = int(self.T / self.dt)
        raster = torch.zeros([timestep, len(vals) + self.input_bias_units])
        for j, t in enumerate(vals):
            if t >= timestep:
                continue
            raster[t, j] = 1.
        if self.input_bias_units > 0:
            if self.input_bias_times is None:
                raise ValueError('input_bias_times must be provided when input_bias_units > 0')
            if len(self.input_bias_times) != self.input_bias_units:
                raise ValueError('input_bias_times must be the same length as input_bias_units')
            for j, t in enumerate(self.input_bias_times):
                if isinstance(t, list):
                    t = np.array(t)
                t = np.round(t / self.dt).astype(int)
                raster[t, j + len(vals)] = 1.
        return raster, cs

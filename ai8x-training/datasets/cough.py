###################################################################################################
# CoughNet data loader
# Habib Ben Abda
# Machine Learning on Microcontrollers
# 2025 - ETH Zurich
###################################################################################################
import errno
import hashlib
import os
import shutil
import tarfile
import time
import urllib
import warnings
from zipfile import ZipFile

import numpy as np
import torch
import torchaudio
from torch.utils.model_zoo import tqdm  # type: ignore # tqdm exists in model_zoo
from torchvision import transforms

import soundfile as sf

import ai8x
from datasets.msnoise import MSnoise
from datasets.signalmixer import SignalMixer


class CoughDataset:
    fs = 16000

    class_dict = {'_silence_': 0, 'cough': 1, 'non_cough': 2, 'deep_breathing': 3, 'laugh': 4, 'throat_clearing': 5,}
    
    dataset_dict = {
        'COUGH': ('cough', 'deep_breathing', 'laugh', 'throat_clearing', '_unknown_'),
    }
    
    # define constants for data types (train, test, validation, benchmark)
    TRAIN = np.uint(0)
    TEST = np.uint(1)
    VALIDATION = np.uint(2)

    def __init__(self, root, classes, d_type, t_type, transform=None, quantization_scheme=None,
                 augmentation=None, gen_dataset=False, save_unquantized=False,):

        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.t_type = t_type
        self.transform = transform
        self.save_unquantized = save_unquantized

        self.__parse_quantization(quantization_scheme)
        self.__parse_augmentation(augmentation)

        if not self.save_unquantized:
            self.data_file = 'dataset.pt'
        else:
            self.data_file = 'unquantized.pt'

        # New
        if gen_dataset:
            if not self.__check_exists():
                self.__makedir_exist_ok(self.raw_folder)
                self.__makedir_exist_ok(self.processed_folder)

                self.__gen_datasets()

        self.data, self.targets, self.data_type, self.shift_limits = \
            torch.load(os.path.join(self.processed_folder, self.data_file))

        print(f'\nProcessing {self.d_type}...')
        self.__filter_dtype()

        if '_silence_' not in self.classes:
            self.__filter_silence()

        self.__filter_classes()

    @property
    def raw_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def raw_test_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw_test')

    @property
    def noise_folder(self):
        """Folder for the different noise data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'noise')

    @property
    def processed_folder(self):
        """Folder for the processed data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def silence_folder(self):
        """Folder for the silence data.
        """
        return os.path.join(self.raw_folder,  '_silence_')
    
    def __parse_quantization(self, quantization_scheme):
        if quantization_scheme:
            self.quantization = quantization_scheme
            if 'bits' not in self.quantization:
                self.quantization['bits'] = 8
            if self.quantization['bits'] == 0:
                self.save_unquantized = True
            if 'compand' not in self.quantization:
                self.quantization['compand'] = False
            if 'mu' not in self.quantization:
                self.quantization['mu'] = 255  # Default, ignored when 'compand' is False
        else:
            print('Undefined quantization schema! ',
                  'Number of bits set to 8.')
            self.quantization = {'bits': 8, 'compand': False, 'mu': 255}

    def __parse_augmentation(self, augmentation):
        self.augmentation = augmentation
        if augmentation:
            if 'aug_num' not in augmentation:
                print('No key `aug_num` in input augmentation dictionary! ',
                      'Using 0.')
                self.augmentation['aug_num'] = 0
            elif self.augmentation['aug_num'] != 0:
                if 'snr' not in augmentation:
                    print('No key `snr` in input augmentation dictionary! ',
                          'Using defaults: [Min: -5.0, Max: 20.0]')
                    self.augmentation['snr'] = {'min': -5.0, 'max': 20.0}
                if 'shift' not in augmentation:
                    print('No key `shift` in input augmentation dictionary! '
                          'Using defaults: [Min:-0.1, Max: 0.1]')
                    self.augmentation['shift'] = {'min': -0.1, 'max': 0.1}

    def __check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.data_file))

    def __makedir_exist_ok(self, dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __filter_dtype(self):
        if self.d_type == 'train':
            idx_to_select = (self.data_type == self.TRAIN)[:, -1]
        elif self.d_type == 'test':
            idx_to_select = (self.data_type == self.TEST)[:, -1]
        else:
            print(f'Unknown data type: {self.d_type}')
            return

        set_size = idx_to_select.sum()
        print(f'{self.d_type} set: {set_size} elements')
        # take a copy of the original data and targets temporarily for validation set
        self.data_original = self.data.clone()
        self.targets_original = self.targets.clone()
        self.data_type_original = self.data_type.clone()
        self.shift_limits_original = self.shift_limits.clone()

        self.data = self.data[idx_to_select, :]
        self.targets = self.targets[idx_to_select, :]
        if self.d_type == 'test':
            self.data_type[idx_to_select, :] = self.TEST

        self.data_type = self.data_type[idx_to_select, :]
        self.shift_limits = self.shift_limits[idx_to_select, :]

        # append validation set to the training set if validation examples are explicitly included
        if self.d_type == 'train':
            idx_to_select = (self.data_type_original == self.VALIDATION)[:, -1]
            if idx_to_select.sum() > 0:  # if validation examples exist
                self.data = torch.cat((self.data, self.data_original[idx_to_select, :]), dim=0)
                self.targets = \
                    torch.cat((self.targets, self.targets_original[idx_to_select, :]), dim=0)
                self.data_type = \
                    torch.cat((self.data_type, self.data_type_original[idx_to_select, :]), dim=0)
                self.shift_limits = \
                    torch.cat((self.shift_limits, self.shift_limits_original[idx_to_select, :]),
                              dim=0)
                # indicate the list of validation indices to be used by distiller's dataloader
                self.valid_indices = range(set_size, set_size + idx_to_select.sum())
                print(f'validation set: {idx_to_select.sum()} elements')

        del self.data_original
        del self.targets_original
        del self.data_type_original
        del self.shift_limits_original

    def __filter_classes(self):
        initial_new_class_label = len(self.class_dict)
        new_class_label = initial_new_class_label
        for c in self.classes:
            if c not in self.class_dict:
                if c == '_unknown_':
                    continue
                raise ValueError(f'Class {c} not found in data')
            num_elems = (self.targets == self.class_dict[c]).cpu().sum()
            print(f'Class {c} (# {self.class_dict[c]}): {num_elems} elements')
            self.targets[(self.targets == self.class_dict[c])] = new_class_label
            new_class_label += 1

        num_elems = (self.targets < initial_new_class_label).cpu().sum()
        print(f'Class _unknown_: {num_elems} elements')
        self.targets[(self.targets < initial_new_class_label)] = new_class_label
        self.targets -= initial_new_class_label

    def __filter_silence(self):
        print('Filtering out _silence_ elements...')
        idx_for_silence = [idx for idx, val in enumerate(self.targets)
                           if val != self.class_dict['_silence_']]

        self.data = torch.index_select(self.data, 0, torch.tensor(idx_for_silence))
        self.targets = torch.index_select(self.targets, 0, torch.tensor(idx_for_silence))
        self.data_type = torch.index_select(self.data_type, 0, torch.tensor(idx_for_silence))
        self.shift_limits = torch.index_select(self.shift_limits, 0, torch.tensor(idx_for_silence))

        if self.d_type == 'train':
            set_size = sum((self.data_type == self.TRAIN)[:, -1])
        elif self.d_type == 'test':
            set_size = sum((self.data_type == self.TEST)[:, -1])
        print(f'Remaining {self.d_type} set: {set_size} elements')

        if self.d_type == 'train':
            train_size = sum((self.data_type == self.TRAIN)[:, -1])
            set_size = sum((self.data_type == self.VALIDATION)[:, -1])
            # indicate the list of validation indices to be used by distiller's dataloader
            self.valid_indices = range(train_size, train_size + set_size)
            print(f'Remaining validation set: {set_size} elements')

#### Audio manipulation #######################################################################
    def __len__(self):
        return len(self.data)

    def __reshape_audio(self, audio, row_len=128):
        # add overlap if necessary later on
        return torch.transpose(audio.reshape((-1, row_len)), 1, 0)

    def shift_and_noise_augment(self, audio, shift_limits):
        """Augments audio by adding random shift and noise.
        """
        random_shift_sample = np.random.randint(shift_limits[0], shift_limits[1])
        aug_audio = self.shift(audio, random_shift_sample)

        if 'snr' in self.augmentation:
            random_snr_coeff = int(np.random.uniform(self.augmentation['snr']['min'],
                                                     self.augmentation['snr']['max']))
            random_snr_coeff = 10 ** (random_snr_coeff / 10)
            if self.quantization['bits'] == 0:
                aug_audio = self.add_white_noise(aug_audio, random_snr_coeff)
            else:
                aug_audio = self.add_quantized_white_noise(aug_audio, random_snr_coeff)

        return aug_audio

    def __getitem__(self, index):
        inp, target = self.data[index], int(self.targets[index])
        data_type, shift_limits = self.data_type[index], self.shift_limits[index]

        # apply dynamic shift and noise augmentation to training examples
        if data_type == self.TRAIN:
            inp = self.shift_and_noise_augment(inp, shift_limits)

        # reshape to 2D
        inp = self.__reshape_audio(inp)
        inp = inp.type(torch.FloatTensor)

        if not self.save_unquantized:
            inp /= 256
        if self.transform is not None:
            inp = self.transform(inp)
        
        #inp = inp.unsqueeze(0)  # from torch.Size([128, 128]) to torch.Size([1, 128, 128]) to represent C; H, W
        return inp, target

    @staticmethod
    def add_white_noise(audio, random_snr_coeff):
        """Adds zero mean Gaussian noise to signal with specified SNR value.
        """
        signal_var = torch.var(audio)
        noise_var_coeff = signal_var / random_snr_coeff
        noise = np.random.normal(0, torch.sqrt(noise_var_coeff), len(audio))
        return audio + torch.Tensor(noise)

    @staticmethod
    def add_quantized_white_noise(audio, random_snr_coeff):
        """Adds zero mean Gaussian noise to signal with specified SNR value.
        """
        signal_var = torch.var(audio.type(torch.float))
        noise_var_coeff = signal_var / random_snr_coeff
        noise = np.random.normal(0, torch.sqrt(noise_var_coeff), len(audio))
        noise = torch.Tensor(noise).type(torch.int16)
        return (audio + noise).clip(0, 255).type(torch.uint8)

    @staticmethod
    def shift(audio, shift_sample):
        """Shifts audio.
        """
        return torch.roll(audio, shift_sample)

    @staticmethod
    def compand(data, mu=255):
        """Compand the signal level to warp from Laplacian distribution to uniform distribution"""
        data = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(1 + mu)
        return data

    @staticmethod
    def expand(data, mu=255):
        """Undo the companding"""
        data = np.sign(data) * (1 / mu) * (np.power((1 + mu), np.abs(data)) - 1)
        return data

    @staticmethod
    def quantize_audio(data, num_bits=8, compand=False, mu=255):
        """Quantize audio
        """
        if compand:
            data = CoughDataset.compand(data, mu)

        step_size = 2.0 / 2 ** (num_bits)
        max_val = 2 ** (num_bits) - 1
        q_data = np.round((data - (-1.0)) / step_size)
        q_data = np.clip(q_data, 0, max_val)

        if compand:
            data_ex = (q_data - 2 ** (num_bits - 1)) / 2 ** (num_bits - 1)
            data_ex = CoughDataset.expand(data_ex)
            q_data = np.round((data_ex - (-1.0)) / step_size)
            q_data = np.clip(q_data, 0, max_val)
        return np.uint8(q_data)

    def get_audio_endpoints(self, audio, fs):
        """Future: May implement a method to detect the beginning & end of voice activity in audio.
        Currently, it returns end points compatible with augmentation['shift'] values
        """
        if self.augmentation:
            return int(-self.augmentation['shift']['min'] * fs), \
                int(len(audio) - self.augmentation['shift']['max'] * fs)

        return (0, int(len(audio)) - 1)

    def speed_augment(self, audio, fs, sample_no=0):
        """Augments audio by randomly changing the speed of the audio.
        The generated coefficient follows 0.9, 1.1, 0.95, 1.05... pattern
        """
        speed_multiplier = 1.0 + 0.2 * (sample_no % 2 - 0.5) / (1 + sample_no // 2)

        sox_effects = [["speed", str(speed_multiplier)], ["rate", str(fs)]]
        aug_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            torch.unsqueeze(torch.from_numpy(audio).float(), dim=0), fs, sox_effects)
        aug_audio = aug_audio.numpy().squeeze()

        return aug_audio, speed_multiplier

    def speed_augment_multiple(self, audio, fs, exp_len, n_augment):
        """Calls `speed_augment` function for n_augment times for given audio data.
        Finally the original audio is added to have (n_augment+1) audio data.
        """
        aug_audio = [None] * (n_augment + 1)
        aug_speed = np.ones((n_augment + 1,))
        shift_limits = np.zeros((n_augment + 1, 2))
        voice_begin_idx, voice_end_idx = self.get_audio_endpoints(audio, fs)
        aug_audio[0] = audio
        for i in range(n_augment):
            aug_audio[i+1], aug_speed[i+1] = self.speed_augment(audio, fs, sample_no=i)
        for i in range(n_augment + 1):
            if len(aug_audio[i]) < exp_len:
                aug_audio[i] = np.pad(aug_audio[i], (0, exp_len - len(aug_audio[i])), 'constant')
            aug_begin_idx = voice_begin_idx * aug_speed[i]
            aug_end_idx = voice_end_idx * aug_speed[i]
            if aug_end_idx - aug_begin_idx <= exp_len:
                # voice activity duration is shorter than the expected length
                segment_begin = max(aug_end_idx, exp_len) - exp_len
                segment_end = max(aug_end_idx, exp_len)
                aug_audio[i] = aug_audio[i][segment_begin:segment_end]
                shift_limits[i, 0] = -aug_begin_idx + (max(aug_end_idx, exp_len) - exp_len)
                shift_limits[i, 1] = max(aug_end_idx, exp_len) - aug_end_idx
            else:
                # voice activity duraction is longer than the expected length
                midpoint = (aug_begin_idx + aug_end_idx) // 2
                aug_audio[i] = aug_audio[i][midpoint - exp_len // 2: midpoint + exp_len // 2]
                shift_limits[i, :] = [0, 0]
        return aug_audio, shift_limits


    def __gen_datasets(self, exp_len=16384):
        """
        Generates datasets from raw audio data samples, including training, validation, 
        and testing sets. This function processes raw audio files, applies augmentations, 
        and saves the processed dataset for later use.

        Args:
            exp_len (int, optional): The expected length of audio samples after processing. 
                                     Defaults to 16384.

        Workflow:
            1. Reads and organizes raw audio files into labels.
            2. Displays the size of the dataset for each keyword label.
            3. Reads testing and validation file lists for dataset partitioning.
            4. Processes each label:
                - Applies speed augmentations for training and validation samples.
                - Quantizes audio data if required.
                - Assigns data type (TRAIN, VALIDATION, TEST, or BENCHMARK).
            5. Concatenates processed data across all labels.
            6. Applies static shift and noise augmentation for validation samples.
            7. Saves the processed dataset as a PyTorch file.

        Attributes:
            - self.raw_folder: Path to the folder containing raw audio data.
            - self.raw_test_folder: Path to the folder containing raw test data.
            - self.silence_folder: Path to the folder containing silence audio data.
            - self.augmentation: Dictionary specifying augmentation parameters.
            - self.save_unquantized: Boolean indicating whether to save unquantized data.
            - self.quantization: Dictionary specifying quantization parameters.
            - self.benchmark_keywords: List of keywords used for benchmarking.
            - self.TRAIN, self.VALIDATION, self.TEST, self.BENCHMARK: Constants for data types.
            - self.processed_folder: Path to save the processed dataset.
            - self.data_file: Name of the file to save the processed dataset.

        Outputs:
            - Saves the processed dataset as a PyTorch file containing:
                - data_in_all: Processed audio data.
                - data_class_all: Class labels for each sample.
                - data_type_all: Data type (TRAIN, VALIDATION, TEST, or BENCHMARK).
                - data_shift_limits_all: Shift limits for each sample.

        Notes:
            - This process may take several minutes depending on the size of the dataset.
            - Augmentations are applied only to training and validation samples.
            - Testing samples are excluded from augmentations.
        """
        print('Generating dataset from raw data samples for the first time.')
        print('This process may take a few minutes.')
        with warnings.catch_warnings():
            warnings.simplefilter('error')

            lst = sorted(os.listdir(self.raw_folder))
            labels = [d for d in lst if os.path.isdir(os.path.join(self.raw_folder, d))
                      and d[0].isalpha() or d == '_silence_']

            # show the size of dataset for each keyword
            print('------------- Label Size ---------------')
            for i, label in enumerate(labels):
                record_list = os.listdir(os.path.join(self.raw_folder, label))
                print(f'{label:8s}:  \t{len(record_list)}')
            print('------------------------------------------')

            # read testing_list.txt & validation_list.txt into sets for fast access
            with open(os.path.join(self.raw_folder, 'testing_list.txt'), encoding="utf-8") as f:
                test_set = set(f.read().splitlines())
            test_silence = [os.path.join('_silence_', rec) for rec in os.listdir(
                os.path.join(self.raw_test_folder, '_silence_'))]
            test_set.update(test_silence)

            with open(os.path.join(self.raw_folder, 'validation_list.txt'), encoding="utf-8") as f:
                validation_set = set(f.read().splitlines())

            # sample 1 out of every 9 generated silence files for the validation set
            silence_files = [os.path.join('_silence_', s) for s in os.listdir(self.silence_folder)
                             if not s[0].isdigit()]  # files starting w/ numbers: used for testing
            validation_set.update(silence_files[::9])

            train_count = 0
            test_count = 0
            valid_count = 0

            for i, label in enumerate(labels):
                print(f'Processing the label: {label}. {i + 1} of {len(labels)}')
                #record_list = sorted(os.listdir(os.path.join(self.raw_folder, label)))
                record_list = sorted([rec for rec in os.listdir(os.path.join(self.raw_folder, label))
                               if not rec.startswith('.')]) # ignore hidden files
                record_len = len(record_list)

                # get the number testing samples for the class
                test_count_class = 0
                for r, record_name in enumerate(record_list):
                    local_filename = os.path.join(label, record_name)
                    if local_filename in test_set:
                        test_count_class += 1

                # no augmentation for test set, subtract them accordingly
                number_of_total_samples = record_len * (self.augmentation['aug_num'] + 1) - \
                    test_count_class * self.augmentation['aug_num']

                if not self.save_unquantized:
                    data_in = np.empty((number_of_total_samples, exp_len), dtype=np.uint8)
                else:
                    data_in = np.empty((number_of_total_samples, exp_len), dtype=np.float32)

                data_type = np.empty((number_of_total_samples, 1), dtype=np.uint8)
                data_shift_limits = np.empty((number_of_total_samples, 2), dtype=np.int16)
                data_class = np.full((number_of_total_samples, 1), i, dtype=np.uint8)

                time_s = time.time()

                sample_index = 0
                for r, record_name in enumerate(record_list):

                    record_name = os.path.join(label, record_name)

                    if r % 1000 == 0:
                        print(f'\t{r + 1} of {record_len}')

                    #raw_test_list = [os.path.join(label, rec) for rec in raw_test_list] # if get error recheck this !!

                    #if record_name in raw_test_list:
                        #d_typ = self.BENCHMARK  # benchmark test
                    #    test_count += 1
                    if record_name in test_set:
                        d_typ = self.TEST
                        test_count += 1
                    elif record_name in validation_set:
                        d_typ = self.VALIDATION
                        valid_count += 1
                    else:
                        d_typ = self.TRAIN
                        train_count += 1

                    record_pth = os.path.join(self.raw_folder, record_name)
                    record, fs = sf.read(record_pth, dtype='float32')

                    # training and validation examples get speed augmentation
                    if d_typ != self.TEST:
                        no_augmentations = self.augmentation['aug_num']
                    else:  # test examples don't get speed augmentation
                        no_augmentations = 0

                    # apply speed augmentations and calculate shift limits
                    audio_seq_list, shift_limits = \
                        self.speed_augment_multiple(record, fs, exp_len, no_augmentations)

                    for local_id, audio_seq in enumerate(audio_seq_list):
                        if not self.save_unquantized:
                            data_in[sample_index] = \
                                CoughDataset.quantize_audio(audio_seq,
                                                   num_bits=self.quantization['bits'],
                                                   compand=self.quantization['compand'],
                                                   mu=self.quantization['mu'])
                        else:
                            data_in[sample_index] = audio_seq
                        data_shift_limits[sample_index] = shift_limits[local_id]
                        data_type[sample_index] = d_typ
                        sample_index += 1

                dur = time.time() - time_s
                print(f'Finished in {dur:.3f} seconds.')
                print(data_in.shape)
                time_s = time.time()
                if i == 0:
                    data_in_all = data_in.copy()
                    data_class_all = data_class.copy()
                    data_type_all = data_type.copy()
                    data_shift_limits_all = data_shift_limits.copy()
                else:
                    data_in_all = np.concatenate((data_in_all, data_in), axis=0)
                    data_class_all = np.concatenate((data_class_all, data_class), axis=0)
                    data_type_all = np.concatenate((data_type_all, data_type), axis=0)
                    data_shift_limits_all = \
                        np.concatenate((data_shift_limits_all, data_shift_limits), axis=0)
                dur = time.time() - time_s
                print(f'Data concatenation finished in {dur:.3f} seconds.')

            data_in_all = torch.from_numpy(data_in_all)
            data_class_all = torch.from_numpy(data_class_all)
            data_type_all = torch.from_numpy(data_type_all)
            data_shift_limits_all = torch.from_numpy(data_shift_limits_all)

            # apply static shift & noise augmentation for validation examples
            for sample_index in range(data_in_all.shape[0]):
                if data_type_all[sample_index] == self.VALIDATION:
                    data_in_all[sample_index] = \
                        self.shift_and_noise_augment(data_in_all[sample_index],
                                                     data_shift_limits_all[sample_index])

            raw_dataset = (data_in_all, data_class_all, data_type_all, data_shift_limits_all)
            torch.save(raw_dataset, os.path.join(self.processed_folder, self.data_file))

        print('Dataset created.')
        print(f'Training: {train_count}, Validation: {valid_count}, Test: {test_count}')

### Dataloader class ends here ###########################################################

### Calling dataloader ############################################################

def cough_get_datasets(data, load_train=True, load_test=True, dataset_name='COUGH', num_classes=4,
                     quantized=True,):
    '''
    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by random stretch/shift and randomly adding noise where
    the stretching coefficient, shift amount and noise SNR level are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, -5 and 20, respectively.
    '''
    (data_dir, args) = data

    if quantized:
        transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])
    else:
        transform = None

    classes = CoughDataset.dataset_dict[dataset_name]

    if num_classes+1 != len(classes):
        raise ValueError(f'num_classes {num_classes}does not match with classes')

    # augmentation
    if quantized:
        augmentation = {'aug_num': 2, 'shift': {'min': -0.1, 'max': 0.1},
                        'snr': {'min': -5.0, 'max': 20.}}
        quantization_scheme = {'compand': False, 'mu': 10}
    else:
        # default: no speed augmentation for unquantized due to memory usage considerations
        augmentation = {'aug_num': 0, 'shift': {'min': -0.1, 'max': 0.1},
                        'snr': {'min': -5.0, 'max': 20.}}
        quantization_scheme = {'bits': 0}

    if load_train:
        train_dataset = CoughDataset(root=data_dir, classes=classes, d_type='train',
                            transform=transform, t_type='keyword',
                            quantization_scheme=quantization_scheme,
                            augmentation=augmentation, gen_dataset=True,)
    else:
        train_dataset = None

    if load_test:
        test_dataset = CoughDataset(root=data_dir, classes=classes, d_type='test',
                           transform=transform, t_type='keyword',
                           quantization_scheme=quantization_scheme,
                           augmentation=augmentation, gen_dataset=True,)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'COUGH',  # cough + non_cough
        'input': (128, 128), # nb channels, dimension (1D or 2D)
        'output': CoughDataset.dataset_dict['COUGH'],
        'weight': (1, 1, 1, 1, 0.33), # class weights used in loss balancing — especially useful when the dataset is imbalanced.
        'loader': cough_get_datasets,
    }
]

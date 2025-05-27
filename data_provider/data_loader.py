import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


class Dataset_GY_hour(Dataset):
    def __init__(self, root_path,
                 flag='train',
                 inp_seq_len=96,
                 label_seq_en=48,
                 pred_seq_len=24):
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.file_path = os.path.join(self.root_path,
                                      f'{flag}_{inp_seq_len}_{label_seq_en}_{pred_seq_len}.pt')

        self.__read_data__()

    def __read_data__(self):
        """
        Read data from file
        self.data is [num_series, [seq_x, seq_x_mark, seq_y, seq_y_mark] ]
        len(data[0]) = 4 is corresponding to seq_x, seq_x_mark, seq_y, seq_y_mark
        """
        self.data = torch.load(self.file_path)

    def __getitem__(self, index):
        """
        seq_x is [inp_seq_len, 1],
        seq_x_mark is [inp_seq_len, 4],
        seq_y is [label_seq_en, 1],
        seq_y_mark is [label_seq_en, 4]
        """
        seq_x, seq_x_mark, seq_y, seq_y_mark = self.data[index]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # data = df_data['LULL'].to_list()
        #
        # # Matplotlib and seaborn for plotting
        # import matplotlib.pyplot as plt
        # import matplotlib
        #
        # matplotlib.rcParams['font.size'] = 18
        # matplotlib.rcParams['figure.dpi'] = 200
        #
        # import seaborn as sns
        #
        # for kernel in ['gau', 'cos', 'biw', 'epa', 'tri', 'triw']:
        #     sns.distplot(data, hist=False, kde=True,
        #                  kde_kws={'kernel': kernel, 'linewidth': 3},
        #                  label=kernel)
        #
        # plt.legend(prop={'size': 16}, title='Kernel')
        # plt.title('Density Plot with Different Kernels');
        # plt.xlabel('Delay (min)')
        # plt.ylabel('Density')
        # plt.show()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # date encoding for time series
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp.loc[:, 'date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TextPrompt(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='text_dataset_in96_out96.csv',
                 target='CO2', scale=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """
        Read data from the text dataset file
        The text dataset has columns: sequence_id, variable, text_description, input_values, output_values
        """
        # Check if we're using the compact version or the full version
        if 'compact' in self.data_path:
            # Compact version has only sequence_id, variable, text_description
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            # We need to load the full version to get input_values and output_values
            full_data_path = self.data_path.replace('_compact', '')
            try:
                df_full = pd.read_csv(os.path.join(self.root_path, full_data_path))
                has_full_data = True
            except FileNotFoundError:
                print(f"Warning: Full dataset file {full_data_path} not found. Using compact version only.")
                has_full_data = False
        else:
            # Full version has all columns
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            df_full = df_raw
            has_full_data = True

        # Filter by target variable if specified
        if self.target != 'all':
            df_raw = df_raw[df_raw['variable'] == self.target]
            if has_full_data:
                df_full = df_full[df_full['variable'] == self.target]

        # Get unique sequence IDs
        sequence_ids = df_raw['sequence_id'].unique()

        # Split into train/val/test sets (60%/20%/20%)
        num_sequences = len(sequence_ids)
        num_train = int(num_sequences * 0.6)
        num_test = int(num_sequences * 0.2)
        num_val = num_sequences - num_train - num_test

        train_ids = sequence_ids[:num_train]
        val_ids = sequence_ids[num_train:num_train+num_val]
        test_ids = sequence_ids[num_train+num_val:]

        # Select sequences based on flag
        if self.set_type == 0:   # train
            selected_ids = train_ids
        elif self.set_type == 1: # val
            selected_ids = val_ids
        else:                    # test
            selected_ids = test_ids

        # Filter data by selected sequence IDs
        df_selected = df_raw[df_raw['sequence_id'].isin(selected_ids)]

        # Store text descriptions
        self.text_descriptions = df_selected['text_description'].values
        self.variables = df_selected['variable'].values
        self.sequence_ids = df_selected['sequence_id'].values

        # If we have the full data, extract input and output values
        if has_full_data:
            df_full_selected = df_full[df_full['sequence_id'].isin(selected_ids)]

            # Convert string representations of lists to actual lists
            input_values = []
            output_values = []

            for _, row in df_full_selected.iterrows():
                # Convert string representation of list to actual list
                if isinstance(row['input_values'], str):
                    input_val = eval(row['input_values'])
                else:
                    input_val = row['input_values']

                if isinstance(row['output_values'], str):
                    output_val = eval(row['output_values'])
                else:
                    output_val = row['output_values']

                input_values.append(input_val)
                output_values.append(output_val)

            # Convert to numpy arrays
            self.data_x = np.array(input_values)
            self.data_y = np.array(output_values)

            # Create dummy time features (same shape as data_x)
            self.data_stamp = np.zeros((len(self.data_x), self.seq_len, 4))
        else:
            # If we don't have the full data, create dummy data
            # This is just for compatibility with the interface
            num_samples = len(df_selected)
            self.data_x = np.zeros((num_samples, self.seq_len, 1))
            self.data_y = np.zeros((num_samples, self.seq_len + self.pred_len, 1))
            self.data_stamp = np.zeros((num_samples, self.seq_len, 4))

    def __getitem__(self, index):
        """
        Return a sample from the dataset

        Returns:
            seq_x: Input sequence [seq_len, feature_dim]
            seq_y: Target sequence [seq_len+pred_len, feature_dim]
            seq_x_mark: Time features for input [seq_len, time_dim]
            seq_y_mark: Time features for target [seq_len+pred_len, time_dim]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Get the text description for this sample
        text = self.text_descriptions[index]
        variable = self.variables[index]
        sequence_id = self.sequence_ids[index]

        # If we have actual data, use it
        if hasattr(self, 'data_x') and len(self.data_x) > 0:
            seq_x = self.data_x[index]
            seq_y = self.data_y[index]
            seq_x_mark = self.data_stamp[index]
            seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))

            # Reshape to match expected dimensions if needed
            if len(seq_x.shape) == 1:
                seq_x = seq_x.reshape(-1, 1)
            if len(seq_y.shape) == 1:
                seq_y = seq_y.reshape(-1, 1)
        else:
            # Create dummy data if we don't have actual data
            seq_x = np.zeros((self.seq_len, 1))
            seq_y = np.zeros((self.label_len + self.pred_len, 1))
            seq_x_mark = np.zeros((self.seq_len, 4))
            seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))

        # Convert to torch tensors
        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        seq_x_mark = torch.FloatTensor(seq_x_mark)
        seq_y_mark = torch.FloatTensor(seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.text_descriptions)

    def inverse_transform(self, data):
        # No scaling is applied in this dataset, so just return the data
        return data

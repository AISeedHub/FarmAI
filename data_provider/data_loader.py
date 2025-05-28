import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


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
            cols = list(df_raw.columns)
            cols.remove(self.target)
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


class Dataset_TextPrompt(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='text_dataset_in96_out96.csv',
                 target='CO2', scale=True, timeenc=0, freq='h', cols=None,
                 use_precomputed_embeddings=False):
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
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.__read_data__()

    def __read_data__(self):
        """
        Read data from the text dataset file
        The text dataset has columns: sequence_id, variable, text_description, input_values, output_values
        If use_precomputed_embeddings is True, it will try to load pre-computed embeddings
        """
        # Check if we should use pre-computed embeddings
        if self.use_precomputed_embeddings:
            # Try to load pre-computed embeddings
            embeddings_path = os.path.join(self.root_path, 'text_embeddings_in96_out96.pkl')
            try:
                print(f"Loading pre-computed embeddings from {embeddings_path}")
                embeddings_df = pd.read_pickle(embeddings_path)
                has_embeddings = True
                print(f"Loaded embeddings for {len(embeddings_df)} text descriptions")
            except FileNotFoundError:
                print(f"Warning: Embeddings file {embeddings_path} not found. Falling back to text descriptions.")
                has_embeddings = False
                # Fall back to regular text dataset
                embeddings_df = None
        else:
            has_embeddings = False
            embeddings_df = None

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
            if has_embeddings:
                embeddings_df = embeddings_df[embeddings_df['variable'] == self.target]

        # Get unique sequence IDs
        sequence_ids = df_raw['sequence_id'].unique()

        # Split into train/val/test sets (60%/20%/20%)
        num_sequences = len(sequence_ids)
        num_train = int(num_sequences * 0.6)
        num_test = int(num_sequences * 0.2)
        num_val = num_sequences - num_train - num_test

        train_ids = sequence_ids[:num_train]
        val_ids = sequence_ids[num_train:num_train + num_val]
        test_ids = sequence_ids[num_train + num_val:]

        # Select sequences based on flag
        if self.set_type == 0:  # train
            selected_ids = train_ids
        elif self.set_type == 1:  # val
            selected_ids = val_ids
        else:  # test
            selected_ids = test_ids

        # Filter data by selected sequence IDs
        df_selected = df_raw[df_raw['sequence_id'].isin(selected_ids)]

        # Store text descriptions
        self.text_descriptions = df_selected['text_description'].values
        self.variables = df_selected['variable'].values
        self.sequence_ids = df_selected['sequence_id'].values

        # Store embeddings if available
        if has_embeddings:
            embeddings_selected = embeddings_df[embeddings_df['sequence_id'].isin(selected_ids)]
            # Make sure the order matches df_selected
            embeddings_selected = embeddings_selected.set_index(['sequence_id', 'variable'])
            df_selected_idx = df_selected.set_index(['sequence_id', 'variable'])
            # Reindex to match df_selected
            embeddings_selected = embeddings_selected.reindex(df_selected_idx.index)
            # Reset index to get back to a regular DataFrame
            embeddings_selected = embeddings_selected.reset_index()
            # Store the embeddings
            self.embeddings = np.array(embeddings_selected['embedding'].tolist())
            print(f"Loaded {len(self.embeddings)} embeddings with shape {self.embeddings[0].shape}")
        else:
            self.embeddings = None

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

            # Apply scaling if enabled
            if self.scale:
                self.scaler = StandardScaler()
                # Get training data indices
                train_indices = np.where(np.isin(self.sequence_ids, train_ids))[0]

                # Reshape data for scaling if needed
                orig_shape_x = self.data_x.shape
                orig_shape_y = self.data_y.shape

                # Reshape to 2D for scaling
                if len(orig_shape_x) > 2:
                    data_x_2d = self.data_x.reshape(orig_shape_x[0], -1)
                    data_y_2d = self.data_y.reshape(orig_shape_y[0], -1)
                else:
                    data_x_2d = self.data_x
                    data_y_2d = self.data_y

                # Fit scaler on training data only
                train_data = data_x_2d[train_indices]
                self.scaler.fit(train_data)

                # Transform all data
                data_x_scaled = self.scaler.transform(data_x_2d)
                data_y_scaled = self.scaler.transform(data_y_2d)

                # Reshape back to original shape
                if len(orig_shape_x) > 2:
                    self.data_x = data_x_scaled.reshape(orig_shape_x)
                    self.data_y = data_y_scaled.reshape(orig_shape_y)
                else:
                    self.data_x = data_x_scaled
                    self.data_y = data_y_scaled

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

        # Get the embedding for this sample if available
        if self.embeddings is not None:
            embedding = self.embeddings[index]
        else:
            embedding = None

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

        # Store embedding as an attribute that can be accessed later
        if embedding is not None:
            self.current_embedding = torch.FloatTensor(embedding)
        else:
            self.current_embedding = None

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.text_descriptions)

    def get_embedding(self, index):
        """
        Get the pre-computed embedding for a specific index

        Args:
            index (int): Index of the sample

        Returns:
            torch.Tensor: Embedding tensor or None if not available
        """
        if self.embeddings is not None:
            return torch.FloatTensor(self.embeddings[index])
        return None

    def get_current_embedding(self):
        """
        Get the embedding for the most recently accessed sample

        Returns:
            torch.Tensor: Embedding tensor or None if not available
        """
        if hasattr(self, 'current_embedding'):
            return self.current_embedding
        return None

    def inverse_transform(self, data):
        # If scaling was applied, inverse transform the data
        if hasattr(self, 'scaler') and self.scale:
            # Reshape data for inverse scaling if needed
            orig_shape = data.shape
            if len(orig_shape) > 2:
                data_2d = data.reshape(orig_shape[0], -1)
            else:
                data_2d = data

            # Inverse transform
            data_inverse = self.scaler.inverse_transform(data_2d)

            # Reshape back to original shape
            if len(orig_shape) > 2:
                return data_inverse.reshape(orig_shape)
            else:
                return data_inverse
        else:
            # No scaling was applied, so just return the data
            return data

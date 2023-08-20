import os
import numpy as np
import pandas as pd

class BitPackMixin:

    def __init__(self):
        unpacked = False
        label_arr = None

    def unpack_all(self):
        self.label_arr = np.unpackbits(self.label_arr, axis=0, count=self.num_imgs).astype(bool)
        self.unpacked = True

    def unpack_marker(self, query):
        # query:    int or list
        # unpack single marker, return column [m, n, 2]
        # m is number of images, n is number of markers
        if self.unpacked:
            return self.label_arr[:, query, :]
        else:   
            return np.unpackbits(self.label_arr[:, query, :], axis=0, count=self.img_meta_df.shape[0]).astype(bool)


class SorghumSNPDataset(BitPackMixin):
    '''
    This class is used to load the sorghum SNP dataset.
    The dataset is stored in a folder with the following structure:
    - known_markers
        - gene_marker_metadata.csv
        - rgb
            - image_metadata.csv
            - train_labels.npy
            - test_labels.npy
        - 3d
            ...
    - unknown_markers
    - sample_markers (for sample_ds)
    ...
    - images
    '''

    def __init__(self, folder, known, sensor, train, sample_ds=False) -> None:
        '''
        folder:     str     (path to the dataset folder)
        known:      bool    (True: known, False: unknown)
        sensor:     str     (rgb or 3d)
        train:      bool    (True: train, False: test)
        sample_ds:  bool    (True: sample dataset(ignore arg known), False: full dataset)
        '''
        self.folder = folder
        self.known = known
        self.sensor = sensor
        self.train = train
        self.sample_ds = sample_ds
        assert self.sensor in ['rgb', '3d'], 'sensor must be either rgb or 3d'
        if sample_ds:
            self.known_str = 'sample_markers'
        else:
            self.known_str = 'known_markers' if known else 'unknown_markers'
        self.train_str = 'train' if train else 'test'
        self.marker_meta_df = None
        self.img_meta_df = None
        self.num_imgs = None
        BitPackMixin.__init__(self)
        self.read_meta()
        self.read_labels()

    def download(self):
        # TODO: download the dataset from the server
        raise NotImplementedError('This function is not implemented yet.')
    
    def subfolder(self, sub):
        return os.path.join(self.folder, sub)

    def read_meta(self):
        self.marker_meta_df = pd.read_csv(self.subfolder(f'{self.known_str}/gene_marker_metadata.csv'))
        self.img_meta_df = pd.read_csv(self.subfolder(f'{self.known_str}/{self.sensor}/image_metadata.csv'))
        self.num_imgs = self.img_meta_df.shape[0]

    def read_labels(self):
        self.label_arr = np.load(self.subfolder(f'{self.known_str}/{self.sensor}/{self.train_str}_labels.npy'))
        self.unpacked = False
        #  unpack large bit array takes a long time, so we only unpack when necessary.
        # self.label_arr = np.unpackbits(self.label_arr_packed, axis=0, count=self.img_meta_df.shape[0]).astype(np.bool)

    def get_snp_labels(self, query):
        # query can be snp name string or int
        # TODO: Add get multi queries (query: list of str/int) function 
        assert isinstance(query, (str, int)), 'query must be either snp name string or int'
        if isinstance(query, str):
            assert query in self.marker_meta_df['marker'].values, f'marker {query} is not in the dataset.'
            query = self.marker_meta_df[self.marker_meta_df['marker'] == query].index[0]
        unpacked = self.unpack_marker(query)
        labels = unpacked[unpacked[:, 0], 1]
        img_paths = self.img_meta_df.iloc[np.argwhere(unpacked[:, 0])[:, 0]]['filepath'].values
        return img_paths, labels
    
class SorghumSNPMultimodalDataset(BitPackMixin):
    def __init__(self, folder, known, train, sample_ds=False) -> None:
        '''
        folder:     str     (path to the dataset folder)
        known:      bool    (True: known, False: unknown)
        train:      bool    (True: train, False: test)
        sample_ds:  bool    (True: sample dataset(ignore arg known), False: full dataset)
        '''
        self.folder = folder
        self.known = known
        self.train = train
        self.sample_ds = sample_ds
        if sample_ds:
            self.known_str = 'sample_markers'
        else:
            self.known_str = 'known_markers' if known else 'unknown_markers'
        self.train_str = 'train' if train else 'test'
        self.marker_meta_df = None
        self.img_meta_df = None
        self.label_arr_packed = None
        BitPackMixin.__init__(self)
        self.read_meta()
        self.read_labels()
    
    def subfolder(self, sub):
        return os.path.join(self.folder, sub)
    
    def read_meta(self):
        self.marker_meta_df = pd.read_csv(self.subfolder(f'{self.known_str}/gene_marker_metadata.csv'))
        self.img_meta_df = pd.read_csv(self.subfolder(f'{self.known_str}/multimodal/image_pair_metadata.csv'))

    def read_labels(self):
        self.label_arr = np.load(self.subfolder(f'{self.known_str}/multimodal/{self.train_str}_labels.npy'))
        self.unpacked = False
        # unpack large bit array takes a long time, so we only unpack when necessary.
        # self.label_arr = np.unpackbits(self.label_arr_packed, axis=0, count=self.img_meta_df.shape[0]).astype(np.bool)

    def get_snp_labels(self, query):
        # query can be snp name string or int
        assert isinstance(query, (str, int)), 'query must be either snp name string or int'
        if isinstance(query, str):
            assert query in self.marker_meta_df['marker'].values, f'marker {query} is not in the dataset.'
            query = self.marker_meta_df[self.marker_meta_df['marker'] == query].index[0]
        unpacked = self.unpack_marker(query)
        labels = unpacked[unpacked[:, 0], 1]
        img_paths = self.img_meta_df.iloc[np.argwhere(unpacked[:, 0])[:, 0]][['3d_filepath', 'rgb_filepath']].values
        return img_paths, labels
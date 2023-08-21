import os, shutil
import numpy as np
import warnings
import pandas as pd
import tarfile
import requests
from tqdm.auto import tqdm


class BitPackMixin:
    '''
    This class is used to handle unpacking the bitwise labels. The labels are stored in a numpy array with shape (m, n, 2), 
    where m is the number of images, n is the number of markers. The first channel represents whether the image has the 
    marker, the second channel represents the label of the marker. The labels are packed into bits to save space.
    '''

    def __init__(self):
        self.unpacked = False
        self.label_arr = None

    def read_labels(self):
        raise NotImplementedError('read_labels is not implemented yet.')

    def unpack_all(self):
        self.label_arr = np.unpackbits(self.label_arr, axis=0, count=self.num_imgs).astype(bool)
        self.unpacked = True

    def unpack_marker(self, query):
        '''
        query:    int or list
        unpack single marker, return shape [m, n, 2]
        m is number of images, n is number of markers in query
        '''
        if self.unpacked:
            return self.label_arr[:, query, :]
        else:   
            return np.unpackbits(self.label_arr[:, query, :], axis=0, count=self.img_meta_df.shape[0]).astype(bool)
        

class DownloadSNPDatasetMixin:
    IMG_URL = ''
    LABEL_URL = 'https://cs.slu.edu/~astylianou/neurips_sorghum_dataset/bitwise_labels.tar.gz'
    SAMPLE_DS_URL = 'https://cs.slu.edu/~astylianou/neurips_sorghum_dataset/genetic_marker_sample_dataset.tar.gz'

    def __init__(self, download=False):
        self.download = download
        if download:
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            self.download_dataset()
        else:
            assert self.check_integrity(), FileNotFoundError('Dataset is not found. Please download the dataset first.')

    def check_integrity(self):
        # only check marker folder exist 
        if self.sample_ds:
            # check sample dataset
            if os.path.exists(os.path.join(self.folder, 'sample_markers')) \
               and os.path.exists(os.path.join(self.folder, 'images')):
                return True
            else:
                return False
        else:
            # check full dataset
            # TODO: check image list completeness
            if os.path.exists(os.path.join(self.folder, 'known_markers')) \
               and os.path.exists(os.path.join(self.folder, 'unknown_markers')) \
               and os.path.exists(os.path.join(self.folder, 'images')):
                return True
            else:
                return False
    
    def extract(self):
        if self.sample_ds:
            filename = DownloadSNPDatasetMixin.SAMPLE_DS_URL.split('/')[-1]
        else:
            filename = DownloadSNPDatasetMixin.LABEL_URL.split('/')[-1]
        with tarfile.open(os.path.join(self.folder, filename), 'r:gz') as tar:
            members = tar.getmembers()
            for member in tqdm(members, total=len(members), desc='Untarring... \t'):
                member.path = os.sep.join(member.path.split(os.sep)[1:])
                tar.extract(member=member, path=self.folder)

    def download_dataset(self):
        im_exist = None
        if self.check_integrity():
            return
        if self.sample_ds:
            dl_url = DownloadSNPDatasetMixin.SAMPLE_DS_URL
        else:       
            dl_url = DownloadSNPDatasetMixin.LABEL_URL
            if not os.path.exists(os.path.join(self.folder, 'images')):
                im_exist = False
                warnings.warn('Image folder does not exist. Please follow the instructions in the README.md to download the images.' +
                              'The label files will be downloaded and extracted now.', UserWarning)
        # ssl cert for cs.slu.edu has incomplete chain, so we need to set verify=False
        r = requests.get(dl_url, stream=True, verify=False)
        if r.status_code != 200:
            r.raise_for_status()
        file_size = int(r.headers.get('Content-Length', 0))
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc='Downloading... \t') as r_raw:
            with open(os.path.join(self.folder, dl_url.split('/')[-1]), 'wb') as f:
                shutil.copyfileobj(r_raw, f)

        self.extract()
        if im_exist == False:
            raise FileNotFoundError('Label files downloaded. Please follow the instructions in the README.md to download the images.')


class SorghumSNPDataset(BitPackMixin, DownloadSNPDatasetMixin):
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

    def __init__(self, folder, known=True, sensor='rgb', train=True, sample_ds=False, download=False) -> None:
        '''
        folder:     str     (path to the dataset folder)
        known:      bool    (True: known, False: unknown)
        sensor:     str     (rgb or 3d)
        train:      bool    (True: train, False: test)
        sample_ds:  bool    (True: sample dataset(ignore arg known), False: full dataset)
        download:   bool    (True: download the dataset or not)
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
        DownloadSNPDatasetMixin.__init__(self, download)
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
        self.img_meta_df['filepath'] = self.img_meta_df['filepath'].apply(lambda x: os.path.join(self.folder, x))
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
    
    def __len__(self):
        return self.num_imgs
    
    def __getitem__(self, query):
        return self.get_snp_labels(query)
    

class SorghumSNPMultimodalDataset(BitPackMixin):
    def __init__(self, folder, known=True, train=True, sample_ds=False, download=False) -> None:
        '''
        folder:     str     (path to the dataset folder)
        known:      bool    (True: known, False: unknown)
        train:      bool    (True: train, False: test)
        sample_ds:  bool    (True: sample dataset(ignore arg known), False: full dataset)
        download:   bool    (True: download the dataset or not)
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
        self.num_img_pairs = None
        BitPackMixin.__init__(self)
        DownloadSNPDatasetMixin.__init__(self, download)
        self.read_meta()
        self.read_labels()
    
    def subfolder(self, sub):
        return os.path.join(self.folder, sub)
    
    def read_meta(self):
        self.marker_meta_df = pd.read_csv(self.subfolder(f'{self.known_str}/gene_marker_metadata.csv'))
        self.img_meta_df = pd.read_csv(self.subfolder(f'{self.known_str}/multimodal/image_pair_metadata.csv'))
        self.img_meta_df['3d_filepath'] = self.img_meta_df['3d_filepath'].apply(lambda x: os.path.join(self.folder, x))
        self.img_meta_df['rgb_filepath'] = self.img_meta_df['rgb_filepath'].apply(lambda x: os.path.join(self.folder, x))
        self.num_img_pairs = self.img_meta_df.shape[0]

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
        labels = list(unpacked[unpacked[:, 0], 1])
        img_paths = self.img_meta_df.iloc[np.argwhere(unpacked[:, 0])[:, 0]][['3d_filepath', 'rgb_filepath']].values
        img_paths =  [os.path.join(self.folder, i) for i in img_paths]
        return img_paths, labels
    
    def __len__(self):
        return self.num_img_pairs
    
    def __getitem__(self, query):
        return self.get_snp_labels(query)
    
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

support_sensors = ['rgb', '3d', 'multimodal']

class GeneImageDataset(Dataset):
    def __init__(self, folder, sensor='rgb', transform=None):
        self.folder = folder
        self.transform = transform
        self.sensor = sensor
        self.metadata_df = pd.read_csv(os.path.join(folder, f'all_{sensor}_info.csv'))

    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.folder, self.metadata_df.iloc[index]['filepath'])
        label_dict = self.metadata_df.iloc[index].to_dict()
        label_dict.pop('filepath')
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label_dict

class PretrainImageDataset(Dataset):
    def __init__(self, folder, sensor='rgb', train=True, transform=None):
        self.folder = folder
        self.transform = transform
        self.sensor = sensor
        if train:
            self.metadata_df = pd.read_csv(os.path.join(folder, f's9_{sensor}_train.csv'))
        else:
            self.metadata_df = pd.read_csv(os.path.join(folder, f's9_{sensor}_test.csv'))
        plot_le = LabelEncoder()
        plot_cls_int = plot_le.fit_transform(self.metadata_df['plot_num'])
        self.metadata_df['plot_cls'] = plot_cls_int

    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.folder, self.metadata_df.iloc[index]['filepath'])
        label_dict = self.metadata_df.iloc[index].to_dict()
        label_dict.pop('filepath')
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label_dict

class GeneticMarkerDataset(Dataset):
    def __init__(self, image_folder, label_folder, gene_marker, sensor='rgb', train=True, meta_label=False, transforms=None) -> None:
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.marker = gene_marker
        self.train = train
        self._meta_label = meta_label
        self.sensor = sensor
        self.known_marker = None

        self.transforms = transforms
        metadata_path = os.path.join(label_folder, f'all_{sensor}_info.csv')
        # Check if sensor is rgb or 3d
        if sensor not in support_sensors:
            raise ValueError(f'Sensor {sensor} not supported. Supported sensors are {support_sensors}')
        # Check if marker is known or unknown
        known_markers = os.listdir(os.path.join(label_folder, 'known_markers'))
        if gene_marker in known_markers:
            self.known = True
        elif os.path.isdir(os.path.join(label_folder, 'unknown_markers', gene_marker)):
            self.known = False
        else:
            raise ValueError('Marker not found in known/unknown markers folder')
        if self.known:
            gene_marker_folder = os.path.join(label_folder, 'known_markers', gene_marker)
        else:
            gene_marker_folder = os.path.join(label_folder, 'unknown_markers', gene_marker)

        if train:
            label_path = os.path.join(gene_marker_folder, f'{sensor}_train.csv')
        else:  
            label_path = os.path.join(gene_marker_folder, f'{sensor}_test.csv')
        self.metadata_df = pd.read_csv(metadata_path)
        self.label_df = pd.read_csv(label_path)
        if self._meta_label:
            self.label = self.label_df.merge(self.metadata_df, on='filepath', how='left')
        else:
            self.label = self.label_df
    
    @property
    def meta_label(self):
        return self._meta_label
            
    @meta_label.setter
    def meta_label(self, value):
        if self._meta_label != value:
            self._meta_label = value
            if self._meta_label:
                self.label = self.label_df.merge(self.metadata_df, on='filepath', how='left')
            else:
                self.label = self.label_df
        

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        row = self.label.iloc[index]
        filepath = row['filepath']
        if self._meta_label:
            label = row.to_dict()
            label.pop('filepath')
        else:
            label = row['label']
        label_dict = row.to_dict()
        label_dict.pop('filepath')
        im = Image.open(os.path.join(self.image_folder, filepath))
        if self.transforms:
            im = self.transforms(im)
        return im, label



class EmbeddingGeneDataset(Dataset):
    def __init__(self, label_folder, gene_marker, sensor,train=True):
        self.label_folder = label_folder
        self.gene_marker = gene_marker
        self.train = train
        assert sensor in support_sensors, f'Sensor {sensor} not supported. Supported sensors are {support_sensors}'
        self.is_multimodal = sensor == 'multimodal'
        self.sensor = sensor
        if self.is_multimodal:
            self.rgb_metadata_df = pd.read_csv(os.path.join(label_folder, f'all_rgb_info.csv'))
            self.d3_metadata_df = pd.read_csv(os.path.join(label_folder, f'all_3d_info.csv'))
            self.rgb_metadata_df['ebd_index'] = self.rgb_metadata_df.index
            self.d3_metadata_df['ebd_index'] = self.d3_metadata_df.index
            self.rgb_metadata_df = self.rgb_metadata_df.add_prefix('rgb_')
            self.d3_metadata_df = self.d3_metadata_df.add_prefix('3d_')
        else:
            self.metadata_df = pd.read_csv(os.path.join(label_folder, f'all_{sensor}_info.csv'))
            self.metadata_df['ebd_index'] = self.metadata_df.index
        # Check if marker is known or unknown
        known_markers = os.listdir(os.path.join(label_folder, 'known_markers'))
        if gene_marker in known_markers:
            self.known = True
        elif os.path.isdir(os.path.join(label_folder, 'unknown_markers', gene_marker)):
            self.known = False
        else:
            raise ValueError('Marker not found in known/unknown markers folder')
        if self.known:
            gene_marker_folder = os.path.join(label_folder, 'known_markers', gene_marker)
        else:
            gene_marker_folder = os.path.join(label_folder, 'unknown_markers', gene_marker)
        # Read the embedding csv
        if train:
            self.marker_df = pd.read_csv(os.path.join(gene_marker_folder, f'{sensor}_train.csv'))
        else:
            self.marker_df = pd.read_csv(os.path.join(gene_marker_folder, f'{sensor}_test.csv'))
        # Merge with metadata
        if self.is_multimodal:
            self.marker_df = self.marker_df.merge(self.rgb_metadata_df, on='rgb_filepath', how='left')
            self.marker_df = self.marker_df.merge(self.d3_metadata_df, on='3d_filepath', how='left')
        else:
            self.marker_df = self.marker_df.merge(self.metadata_df, on='filepath', how='left')

    def __len__(self):
        return len(self.marker_df)
    
    def __getitem__(self, index):
        row = self.marker_df.iloc[index]
        if self.is_multimodal:
            rgb_ebd_index = row['rgb_ebd_index']
            d3_ebd_index = row['3d_ebd_index']
            label = row['label']
            cultivar = row['rgb_cultivar']
            return rgb_ebd_index, d3_ebd_index, label, cultivar
        else:
            ebd_index = row['ebd_index']
            label = row['label']
            cultivar = row['cultivar']
            return ebd_index, label, cultivar
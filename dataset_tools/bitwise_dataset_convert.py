import pickle, glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm

'''
Target folder sturcture:
- known_markers
  - gene_marker_metadata.csv
  - rgb
    - image_metadata.csv
    - train_labels.bin
    - test_labels.bin
  - 3d
    ...
- unknown_markers
  ...
- images
'''


# known train, known test, unknown train, unknown test for 3d and rgb
def generate_bitwise_meta(csv_ds_path, bitwise_ds_path, known):
    '''
    csv_ds_path:        str     (path to the csv dataset)
    bitwise_ds_path:    str     (path to the bitwise dataset)
    known:              bool    (True: known, False: unknown)
    train:              bool    (True: train, False: test)
    rgb:                bool    (True: rgb, False: 3d)
    '''
    def csv_subfolder(sub):
        return os.path.join(csv_ds_path, sub)
    def bitwise_subfolder(sub):
        return os.path.join(bitwise_ds_path, sub)
    # check if the bitwise dataset folder exists
    if not os.path.exists(bitwise_ds_path):
        os.makedirs(bitwise_ds_path)
    # check if the csv dataset folder exists
    if not os.path.exists(csv_ds_path):
        print(f'csv dataset folder {csv_ds_path} does not exist!')
        return
    # check if the csv dataset folder is empty
    if len(os.listdir(csv_ds_path)) == 0:
        print(f'csv dataset folder {csv_ds_path} is empty!')
        return
    
    known_str = 'known' if known else 'unknown'
    # generate marker_meta.csv
    marker_list = os.listdir(csv_subfolder(f'{known_str}_markers/'))
    marker_df = pd.DataFrame({'marker': marker_list}, index=pd.Index(list(range(len(marker_list))), name='id'))
    if not os.path.exists(bitwise_subfolder(f'{known_str}_markers/')):
        os.makedirs(bitwise_subfolder(f'{known_str}_markers/'))
    marker_df.to_csv(bitwise_subfolder(f'{known_str}_markers/gene_marker_metadata.csv'), index=True)
    # generate metadat for rgb
    rgb_im_path_df = pd.read_csv(csv_subfolder('all_rgb_info.csv'))
    rgb_im_path_df.index.name = 'id'
    if not os.path.exists(bitwise_subfolder(f'{known_str}_markers/rgb/')):
        os.makedirs(bitwise_subfolder(f'{known_str}_markers/rgb/'))
    rgb_im_path_df.to_csv(bitwise_subfolder(f'{known_str}_markers/rgb/image_metadata.csv'), index=True)
    # generate metadata for 3d
    d3_im_path_df = pd.read_csv(csv_subfolder('all_3d_info.csv'))
    d3_im_path_df.index.name = 'id'
    if not os.path.exists(bitwise_subfolder(f'{known_str}_markers/3d/')):
        os.makedirs(bitwise_subfolder(f'{known_str}_markers/3d/'))
    d3_im_path_df.to_csv(bitwise_subfolder(f'{known_str}_markers/3d/image_metadata.csv'), index=True)




def generate_bitwise_label(csv_ds_path, bitwise_ds_path, known, rgb, train):
    known_str = 'known' if known else 'unknown'
    sensor_str = 'rgb' if rgb else '3d'
    train_str = 'train' if train else 'test'
    # read image meta, marker meta
    imgs_meta_df = pd.read_csv(os.path.join(bitwise_ds_path, f'{known_str}_markers/{sensor_str}/image_metadata.csv'), index_col='id')
    imgs_meta_df['id'] = imgs_meta_df.index
    marker_meta_df = pd.read_csv(os.path.join(bitwise_ds_path, f'{known_str}_markers/gene_marker_metadata.csv'), index_col='id')
    # initialize label array
    label_arr = np.zeros([imgs_meta_df.shape[0], marker_meta_df.shape[0], 2], dtype=bool)
    # write labels
    for row in tqdm(marker_meta_df.iterrows(), total=marker_meta_df.shape[0], desc=f'load {known_str} {sensor_str} {train_str} labels'):
        marker = row[1]['marker']
        marker_id = row[0]
        marker_label_df = pd.read_csv(os.path.join(csv_ds_path, f'{known_str}_markers', marker, f'{sensor_str}_{train_str}.csv'))
        im_df = marker_label_df.merge(imgs_meta_df, on='filepath', how='left')
        im_ids = im_df['id'].to_numpy()
        im_labels = im_df['label'].to_numpy().astype(bool)
        label_arr[im_ids, marker_id, 0] = 1
        label_arr[im_ids, marker_id, 1] = im_labels
    # pack and save
    packed_label_arr = np.packbits(label_arr, axis=0)
    np.save(os.path.join(bitwise_ds_path, f'{known_str}_markers', sensor_str, f'{train_str}_labels.npy'), packed_label_arr)


def generate_bitwise_multimodal_meta(csv_ds_path, bitwise_ds_path, known):
    def csv_subfolder(sub):
        return os.path.join(csv_ds_path, sub)
    def bitwise_subfolder(sub):
        return os.path.join(bitwise_ds_path, sub)
    # check if the bitwise dataset folder exists
    if not os.path.exists(bitwise_ds_path):
        os.makedirs(bitwise_ds_path)
    # check if the csv dataset folder exists
    if not os.path.exists(csv_ds_path):
        print(f'csv dataset folder {csv_ds_path} does not exist!')
        return
    # check if the csv dataset folder is empty
    if len(os.listdir(csv_ds_path)) == 0:
        print(f'csv dataset folder {csv_ds_path} is empty!')
        return
    
    known_str = 'known' if known else 'unknown'
    # generate metadat for mm
    mm_train_test_list = glob.glob(f'/data/shared/genetic_marker_datasets/{known_str}_markers/*/multimodal_*.csv')
    mm_im_path_df = pd.concat([pd.read_csv(f)[['3d_filepath', 'rgb_filepath','label']] for f in tqdm(mm_train_test_list, desc='multimodal')], ignore_index=True)[['3d_filepath', 'rgb_filepath']] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    mm_im_path_df.index.name = 'id'
    if not os.path.exists(bitwise_subfolder(f'{known_str}_markers/multimodal/')):
        os.makedirs(bitwise_subfolder(f'{known_str}_markers/multimodal/'))
    mm_im_path_df.to_csv(bitwise_subfolder(f'{known_str}_markers/multimodal/image_pair_metadata.csv'), index=True)

def generate_multimodal_bitwise_label(csv_ds_path, bitwise_ds_path, known, train):
    known_str = 'known' if known else 'unknown'
    train_str = 'train' if train else 'test'
    # read image meta, marker meta
    imgs_meta_df = pd.read_csv(os.path.join(bitwise_ds_path, f'{known_str}_markers/multimodal/image_pair_metadata.csv'), index_col='id')
    imgs_meta_df['id'] = imgs_meta_df.index
    marker_meta_df = pd.read_csv(os.path.join(bitwise_ds_path, f'{known_str}_markers/gene_marker_metadata.csv'), index_col='id')
    # initialize label array
    label_arr = np.zeros([imgs_meta_df.shape[0], marker_meta_df.shape[0], 2], dtype=bool)
    # write labels
    for row in tqdm(marker_meta_df.iterrows(), total=marker_meta_df.shape[0], desc=f'load {known_str} multimodal {train_str} labels'):
        marker = row[1]['marker']
        marker_id = row[0]
        marker_label_df = pd.read_csv(os.path.join(csv_ds_path, f'{known_str}_markers', marker, f'multimodal_{train_str}.csv'))
        im_df = marker_label_df.merge(imgs_meta_df, on=['3d_filepath', 'rgb_filepath'], how='left')
        im_ids = im_df['id'].to_numpy()
        im_labels = im_df['label'].to_numpy().astype(bool)
        label_arr[im_ids, marker_id, 0] = 1
        label_arr[im_ids, marker_id, 1] = im_labels
    # pack and save
    packed_label_arr = np.packbits(label_arr, axis=0)
    np.save(os.path.join(bitwise_ds_path, f'{known_str}_markers', 'multimodal', f'{train_str}_labels.npy'), packed_label_arr)

if __name__ == '__main__':
    csv_ds_path = '/data/shared/genetic_marker_datasets/'
    bitwise_ds_path = '/data/shared/genetic_marker_dataset_bitwise_label'
    print('convert known markers')
    generate_bitwise_meta(csv_ds_path, bitwise_ds_path, known=True)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, True, True, True)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, True, True, False)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, True, False, True)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, True, False, False)
    print('convert unknown markers')
    generate_bitwise_meta(csv_ds_path, bitwise_ds_path, known=False)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, False, True, True)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, False, True, False)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, False, False, True)
    generate_bitwise_label(csv_ds_path, bitwise_ds_path, False, False, False)
    print('convert multimodal')
    generate_bitwise_multimodal_meta(csv_ds_path, bitwise_ds_path, known=True)
    generate_multimodal_bitwise_label(csv_ds_path, bitwise_ds_path, True, True)
    generate_multimodal_bitwise_label(csv_ds_path, bitwise_ds_path, True, False)
    generate_bitwise_multimodal_meta(csv_ds_path, bitwise_ds_path, known=False)
    generate_multimodal_bitwise_label(csv_ds_path, bitwise_ds_path, False, True)
    generate_multimodal_bitwise_label(csv_ds_path, bitwise_ds_path, False, False)


import os
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from bitwise_dataset import SorghumSNPDataset, SorghumSNPMultimodalDataset
from datetime import date
from tqdm import tqdm

ds_root = '/data/shared/genetic_marker_dataset_bitwise_label/'
sample_ds_root = '/data/shared/genetic_marker_sample_dataset'
sample_ds_marker_root = '/data/shared/genetic_marker_sample_dataset/sample_markers'
sample_marker_idx = [3, 6, 9]
start_date = date(2017, 6, 5) # inculde
end_date = date(2017, 7, 5) # exclude

def parse_date(df):
    def get_date(i):
        fname = i.split('/')[-1]
        return date.fromisoformat(fname.split('__')[0])
    df['date'] = df['3d_filepath'].apply(get_date)


train_mm_ds = SorghumSNPMultimodalDataset(ds_root, known=True, train=True, prefix_path=False)
train_mm_unpacked_label_arr = np.unpackbits(train_mm_ds.label_arr, axis=0, count=train_mm_ds.img_meta_df.shape[0]).astype(bool)
test_mm_ds = SorghumSNPMultimodalDataset(ds_root, known=True, train=False, prefix_path=False)
test_mm_unpacked_label_arr = np.unpackbits(test_mm_ds.label_arr, axis=0, count=test_mm_ds.img_meta_df.shape[0]).astype(bool)

train_mm_sample_pool = np.argwhere(train_mm_unpacked_label_arr[:,sample_marker_idx,0].all(1))
test_mm_sample_pool = np.argwhere(test_mm_unpacked_label_arr[:,sample_marker_idx,0].all(1))

train_mm_sample_pool_df = train_mm_ds.img_meta_df.iloc[train_mm_sample_pool[:, 0]].copy()
test_mm_sample_pool_df = train_mm_ds.img_meta_df.iloc[test_mm_sample_pool[:, 0]].copy()

train_mm_sample_pool_df.drop_duplicates(subset=['rgb_filepath'], inplace=True)
test_mm_sample_pool_df.drop_duplicates(subset=['rgb_filepath'], inplace=True)
    
parse_date(train_mm_sample_pool_df)
parse_date(test_mm_sample_pool_df)

train_mm_sample_pool_date_limited_df = train_mm_sample_pool_df[(train_mm_sample_pool_df['date'] >= start_date) & (train_mm_sample_pool_df['date'] < end_date)]
test_mm_sample_pool_date_limited_df = test_mm_sample_pool_df[(test_mm_sample_pool_df['date'] >= start_date) & (test_mm_sample_pool_df['date'] < end_date)]
print('num of train images: ', train_mm_sample_pool_date_limited_df.shape[0])
print('num of test images: ', test_mm_sample_pool_date_limited_df.shape[0])

sampled_train_mm_df = train_mm_sample_pool_date_limited_df.sample(800, random_state=42)
sampled_test_mm_df = test_mm_sample_pool_date_limited_df.sample(200, random_state=42)
sampled_train_mm_df.rename(columns={'id': 'original_id'}, inplace=True)
sampled_test_mm_df.rename(columns={'id': 'original_id'}, inplace=True)
sampled_mm_df = pd.concat([sampled_train_mm_df, sampled_test_mm_df], axis=0)
sampled_mm_df = sampled_mm_df.sort_values('original_id').reset_index(drop=True)



# 1. save marker metadata
# 2. save multimodal image metadata
# 3. subset the mm original label file with origial_id and markers by [original_id, marker_id] 
# 4. find the original_id for 3d and rgb
# 5. subset the original dataset with original_id, save image metadata
# 6. subset the rgb 3d original label file with origial_id and markers by [original_id, marker_id] 

if not os.path.exists(sample_ds_marker_root):
    os.makedirs(sample_ds_marker_root)

# 1. save marker metadata
original_marker_meta_df = pd.read_csv(os.path.join(ds_root, 'known_markers/gene_marker_metadata.csv'))
original_marker_meta_df = original_marker_meta_df.iloc[sample_marker_idx].copy()
original_marker_meta_df.rename(columns={'id': 'original_id'}, inplace=True)
original_marker_meta_df.reset_index(drop=True, inplace=True)
original_marker_meta_df.index.name='id'
original_marker_meta_df.to_csv(os.path.join(sample_ds_marker_root, 'gene_marker_metadata.csv'))

# 2. save multimodal image metadata
if not os.path.exists(os.path.join(sample_ds_marker_root, 'multimodal')):
    os.makedirs(os.path.join(sample_ds_marker_root, 'multimodal'))
sampled_mm_df[['original_id', '3d_filepath', 'rgb_filepath']].to_csv(os.path.join(sample_ds_marker_root, 'multimodal', 'image_pair_metadata.csv'))

# 3. subset mm label
sampled_train_mm_unpacked_label_arr = train_mm_unpacked_label_arr[sampled_mm_df['original_id'].to_numpy()][:, sample_marker_idx]
np.save(os.path.join(sample_ds_marker_root, 'multimodal', 'train_labels.npy'), np.packbits(sampled_train_mm_unpacked_label_arr, axis=0))
sampled_test_mm_unpacked_label_arr = test_mm_unpacked_label_arr[sampled_mm_df['original_id'].to_numpy()][:, sample_marker_idx]
np.save(os.path.join(sample_ds_marker_root, 'multimodal', 'test_labels.npy'), np.packbits(sampled_test_mm_unpacked_label_arr, axis=0))

# 4. find original_id for rgb and 3d
rgb_df = pd.read_csv(os.path.join(ds_root, 'known_markers', 'rgb', 'image_metadata.csv'))
print('num of sampled rgb images: ', (rgb_df['filepath'].isin(sampled_mm_df['rgb_filepath'])).sum())
sampled_rgb_df = rgb_df[rgb_df['filepath'].isin(sampled_mm_df['rgb_filepath'])].copy()
sampled_rgb_df.rename(columns={'id': 'original_id'}, inplace=True)
sampled_rgb_df.reset_index(drop=True, inplace=True)
if not os.path.exists(os.path.join(sample_ds_marker_root, 'rgb')):
    os.makedirs(os.path.join(sample_ds_marker_root, 'rgb'))
sampled_rgb_df.to_csv(os.path.join(sample_ds_marker_root, 'rgb', 'image_metadata.csv'))

d3_df = pd.read_csv(os.path.join(ds_root, 'known_markers', '3d', 'image_metadata.csv'))
print('num of sampled 3d images: ', (d3_df['filepath'].isin(sampled_mm_df['3d_filepath'])).sum())
sampled_d3_df = d3_df[d3_df['filepath'].isin(sampled_mm_df['3d_filepath'])].copy()
sampled_d3_df.rename(columns={'id': 'original_id'}, inplace=True)
sampled_d3_df.reset_index(drop=True, inplace=True)
if not os.path.exists(os.path.join(sample_ds_marker_root, '3d')):
    os.makedirs(os.path.join(sample_ds_marker_root, '3d'))
sampled_d3_df.to_csv(os.path.join(sample_ds_marker_root, '3d', 'image_metadata.csv'))

# 5. subset rgb 3d labels.
sensor='rgb'
train_ds = SorghumSNPDataset(ds_root, known=True, sensor=sensor, train=True, prefix_path=False)
train_unpacked_label_arr = np.unpackbits(train_ds.label_arr, axis=0, count=train_ds.img_meta_df.shape[0]).astype(bool)
test_ds = SorghumSNPDataset(ds_root, known=True, sensor=sensor, train=False, prefix_path=False)
test_unpacked_label_arr = np.unpackbits(test_ds.label_arr, axis=0, count=test_ds.img_meta_df.shape[0]).astype(bool)
sampled_train_unpacked_label_arr = train_unpacked_label_arr[sampled_rgb_df['original_id']][:, sample_marker_idx]
sampled_test_unpacked_label_arr = test_unpacked_label_arr[sampled_rgb_df['original_id']][:, sample_marker_idx]
np.save(os.path.join(sample_ds_marker_root, sensor, 'train_labels.npy'), np.packbits(sampled_train_unpacked_label_arr, axis=0))
np.save(os.path.join(sample_ds_marker_root, sensor, 'test_labels.npy'), np.packbits(sampled_test_unpacked_label_arr, axis=0))

sensor='3d'
train_ds = SorghumSNPDataset(ds_root, known=True, sensor=sensor, train=True, prefix_path=False)
train_unpacked_label_arr = np.unpackbits(train_ds.label_arr, axis=0, count=train_ds.img_meta_df.shape[0]).astype(bool)
test_ds = SorghumSNPDataset(ds_root, known=True, sensor=sensor, train=False, prefix_path=False)
test_unpacked_label_arr = np.unpackbits(test_ds.label_arr, axis=0, count=test_ds.img_meta_df.shape[0]).astype(bool)
sampled_train_unpacked_label_arr = train_unpacked_label_arr[sampled_d3_df['original_id']][:, sample_marker_idx]
sampled_test_unpacked_label_arr = test_unpacked_label_arr[sampled_d3_df['original_id']][:, sample_marker_idx]
np.save(os.path.join(sample_ds_marker_root, sensor, 'train_labels.npy'), np.packbits(sampled_train_unpacked_label_arr, axis=0))
np.save(os.path.join(sample_ds_marker_root, sensor, 'test_labels.npy'), np.packbits(sampled_test_unpacked_label_arr, axis=0))

# 6. link images to sample_ds_root
for p in sampled_rgb_df['filepath']:
    sample_p = os.path.join(sample_ds_root, p)
    if not os.path.exists(os.path.dirname(sample_p)):
        os.makedirs(os.path.dirname(sample_p))
    os.link(os.path.realpath(os.path.join(ds_root, p)), sample_p)

for p in sampled_d3_df['filepath']:
    sample_p = os.path.join(sample_ds_root, p)
    if not os.path.exists(os.path.dirname(sample_p)):
        os.makedirs(os.path.dirname(sample_p))
    os.link(os.path.realpath(os.path.join(ds_root, p)), sample_p)

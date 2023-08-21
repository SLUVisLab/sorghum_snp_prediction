# SG×P : A Sorghum Genotype × Phenotype Prediction Dataset and Benchmark
<p align="center" style="text-align: center;">
  <img src="https://terraref.org/sites/terraref.org/files/TERRA-REF-Scanner.jpg" width=75%>
  <br/>
  <i>Data in the SGxP benchmark comes from the TERRA-REF Project and field, shown here with sorghum growing.</i>
</p>

Large scale field-phenotyping approaches have the potential to solve important questions about the relationship of plant genotype to plant phenotype. Computational approaches to measuring the phenotype (the observable plant features) are required to address the problem at a large scale, but machine learning approaches to extract phenotypes from sensor data have been hampered by limited access to (a) sufficiently large, organized multi-sensor datasets, (b) field trials that have a large scale and significant number of genotypes, (c) full genetic sequencing of those phenotypes, and (d) datasets sufficiently organized so that algorithm centered researchers can directly address the real biological problems.

To address this, we present SGxP, a novel benchmark dataset from a large-scale field trial consisting of the complete genotype of over 300 sorghum varieties, and time sequences of imagery from several field plots growing each variety, taken with RGB and laser 3D scanner imaging. To lower the barrier to entry and facilitate further developments, we provide a set of well organized, multi-sensor imagery and corresponding genomic data. We implement baseline deep learning based phenotyping approaches to create baseline results for individual sensors and multi-sensor fusion for detecting genetic mutations with known impacts. We also provide and support an open-ended challenge by identifying thousands of genetic mutations whose phenotypic impacts are currently unknown. A web interface for machine learning researchers and practitioners to share approaches, visualizations and hypotheses supports engagement with plant biologists to further the understanding of the sorghum genotype x phenotype relationship.

This repository includes code to load data in the SGxP Benchmark, and to reproduce the baseline results shared in the paper.

The full dataset, leaderboard (including baseline results) and discussion forums can be found at http://sorghumsnpbenchmark.com.
  
## Dataset Quick Start Guide
### Download 
#### Sample Dataset
To quickly get started with the sample dataset, follow these steps:
   
1. **Clone the Repository:**

    Clone this repository to your local machine using the following command:

    ```bash
    git clone https://github.com/SLUVisLab/sorghum_snp_prediction.git
    ```
2. **Install the required packages:**

    ```bash
    pip install -r requirements_dataset_tools.txt
    ```

3. **Download the sample dataset:**

    ```python
    from dataset_tools import SorghumSNPDataset

    ds = SorghumSNPDataset('path/to/the/dataset', sample_ds=True, sensor='rgb', train=True, download=True)
    ```

#### Full Dataset

For the full dataset, please note that due to the size of images, the code does not handle downloading the images directly. Follow these steps:

1. **Download the images:**
   
    [images.tar.gz](https://cs.slu.edu/~astylianou/neurips_sorghum_dataset/images.tar.gz) (230GB)

2. **Organize the images:**

    Place the downloaded file `images.tar.gz` under the dataset folder and untar it. The folder structure should look like this:
    
    ```
    path/to/dataset/images/[sensor]/[cultivar]/[image]
    ```

3. **Download all the labels:**

    After setting up the images, use the provided code to download all the labels:
    
    ```python
    ds = SorghumSNPDataset('path/to/dataset/', marker='known', sensor='rgb', train=True, download=True)
    ```
### Get Marker Specific Dataset

Once you have the dataset downloaded and organized, you can obtain image paths and labels specific to a marker by either its name or index:

```python
# By marker name
img_paths, labels = ds['sobic_001G269200_1_51589435']

# By marker index
img_paths, labels = ds[0]
```

### Multimodal Dataset

Once you have downloaded the dataset (full or sampled), you can use the following code to obtain paths to image pairs and labels for the marker specific multimodal dataset:

```python
from dataset_tools import SorghumSNPMultimodalDataset

ds = SorghumSNPMultimodalDataset('path/to/the/dataset', marker='known', sample=False, train=True, download=True)
img_pair_paths, labels = ds[0]
```


## Baseline
For all of the tasks included in the SGxP benchmark, we include the performance on a baseline model that was pretrained on TERRA-REF imagery from a different season (see the paper for more details).

You can download the pre-trained weights for each sensor:
[baseline_model_and_ebd.tar.gz](https://cs.slu.edu/~astylianou/neurips_sorghum_dataset/baseline_model_and_ebd.tar.gz)

### Generating results based on pretrained models  
The reproduce the baseline results using these model, first clone this repository and follow these steps:
  1. Install the required packages: `pip install -r requirements.txt`
  2. Put the pretrained models under `results/model/`
  3. Run `notebooks/gene_embeddings.ipynb` to get the image embeddings.
  4. Run `known_gene_pred.ipynb` to compute the accuracy for each genetic markers.

### Finetuning a model with Pretrain Dataset
If you would prefer to train your own pretrained model, you can download the imagery here:
 [genetic_marker_pretrain_dataset.tar.gz](https://cs.slu.edu/~astylianou/neurips_sorghum_dataset/genetic_marker_pretrain_dataset.tar.gz) (70GB). 
 These images are from entirely different lines of sorghum grown in a different season under the TERRA-REF gantry, so there is no risk of data leakage for the SGxP benchmark tasks.

After you have downloaded the images and cloned this repository, follow these steps:
  1. Modify the dataset folder location in `tasks/s9_pretrain_rgb_jpg_res50_512_softmax_ebd.py` and `tasks/s9_pretrain_scnr3d_jpg_res50_512_softmax_ebd.py` to the location where you downloaded the imagery.
  2. Run `python -m tasks.s9_pretrain_rgb_jpg_res50_512_softmax_ebd` for RGB model finetuning and `python -m tasks.s9_pretrain_scnr3d_jpg_res50_512_softmax_ebd` for 3D scanner finetuning

## Paper
The paper that details the SGxP Benchmark, "SG×P : A Sorghum Genotype × Phenotype Prediction Dataset and Benchmark", is currently under review at NeurIPS 2023 and can be found at https://openreview.net/pdf?id=dOeBYjxSoq.

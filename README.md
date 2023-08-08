# SG×P : A Sorghum Genotype × Phenotype Prediction Dataset and Benchmark
  A multimodal dataset and set of benchmarks focused on the discovery of genotype x phenotype relationships in bioenergy sorghum.
## Baseline
### Generate results based on pretrained models  
  1. Download the pretrained model weights at https://sorghumsnpbenchmark.com/ and put it under `results/model/`
  2. Run `notebooks/gene_embeddings.ipynb` to get the image embeddings.
  3. Run `known_gene_pred.ipynb` to compute the accuracy for each genetic markers.
### Finetune the model with Pretrain Dataset
  1. Download the pretrain dataset from the website
  2. Modify the dataset folder location in `tasks/s9_pretrain_rgb_jpg_res50_512_softmax_ebd.py` and `tasks/s9_pretrain_scnr3d_jpg_res50_512_softmax_ebd.py`
  3. Run `python -m tasks.s9_pretrain_rgb_jpg_res50_512_softmax_ebd` for RGB model finetune and `python -m tasks.s9_pretrain_scnr3d_jpg_res50_512_softmax_ebd` for 3D scanner finetune

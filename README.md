# PuCo-PyTorch

## 1. Introduction

## 2. Prerequisite

<details>
<summary>
Set conda environment.
</summary>

```bash
conda create -n puco python=3.7
source activate puco

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install scipy=1.6.2 tqdm
pip install opencv-python sacred
pip install pymongo # (optional) for MongoObserver
```
</details>


### 2.1 Datasets

|  Datasets  | Train Images | Val Images |                                Download                                 |                  Files                  |
|:----------:|:------------:|:----------:|:-----------------------------------------------------------------------:|:---------------------------------------:|
| Cityscapes |     2975     |    500     |      [link](https://www.cityscapes-dataset.com/downloads/) (11GB)       | `[leftImg8bit\gtFine]_trainvaltest.zip` |
|    GTA5    |    24966     |     -      | [link](https://download.visinf.tu-darmstadt.de/data/from_games/) (56GB) |         `xx_[image\label].zip`          |
|  SYNTHIA   |     9400     |     -      |          [link](http://synthia-dataset.net/downloads/) (20GB)           |        `SYNTHIA-RAND-CITYSCAPES`        |

Put datasets into the folder `./datasets` .

<details>
<summary>
Dataset structure
</summary>

```
./datasets
  ├── CityScape
  │   ├── gtFine
  │   │ ├── train
  │   │ └── val
  │   └── leftImg8bit
  │       ├── train
  │       └── val
  ├── GTA5
  │   ├── images
  │   ├── labels
  │   └── split.mat
  └── SYNTHIA
      ├── GT
      ├── list
      └── RGB
  
```

</details>

### 2.2 Pretrained models

* ImageNet pretrained models. [[resnet101-5d3b4d8f]](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)

Put pretrained models into the folder `./pretrained` .

## 3. Training

Training UDA model using PuCo (pseudo label generation is automated):

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py -i <ExperimentID>
```

## 4. Evaluation

* Run the evaluation

```bash
CUDA_VISIBLE_DEVICES "0,1" python test.py with warmup_path=pretrained/from_gta5_to_cityscapes_on_deeplabv2_best_model_round3.pkl -u
```

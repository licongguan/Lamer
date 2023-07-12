<div align="center"> 

# Label-Free Model Evaluation and Weighted Uncertainty Sample Selection for Domain Adaptive Instance Segmentation
by [Licong Guan](https://licongguan.github.io/), Xue Yuan

</div>

This repository provides the official code for the paper "Label-Free Model Evaluation and Weighted Uncertainty Sample Selection for Domain Adaptive Instance Segmentation"

> **Abstract** 
This paper studies the model evaluation and model optimization problems faced
during model deployment due to domain differences between the target domain
and the source domain. Existing methods for model accuracy evaluation require
a complete annotated test set, yet adding additional human labels for each different
application scenario can be very expensive and time-consuming. To address
this issue, this paper proposes an instance segmentation model evaluation
method based on domain differences, which can give the prediction accuracy of
the model on unlabeled test sets. In addition, to further improve the deployment
accuracy of the model at a lower cost, this paper proposes an “effective operation”-
based labeling cost calculation method and a weighted uncertainty sample selection
method. The former can accurately calculate the labeling cost for instance
segmentation, and the latter can select the most valuable samples from the target
domain for labeling and training. Model evaluation experiments show that the root
mean square error (RMSE) of this method on Cityscapes is about 4% smaller than
other existing model evaluation methods. Model optimization experiments show
that the proposed method achieves higher model accuracy than the comparative
methods under four different data partitioning protocols.

## Highlights

- **We propose a label-free model evaluation method for instance segmentation tasks. This method predicts the accuracy of the model by computing the domain difference between the source and target domains;** 

- **We propose a cost-computing method for instance segmentation labeling based on ``effective operation". This method uses the number of mouse clicks in the actual labeling process as a measure of the labeling cost, which is more reliable and saves labeling costs than other calculation methods;** 

- **We propose a weighted uncertainty sample selection method for active learning of instance segmentation. This method alleviates the imbalance of categories in the dataset and the imbalance of the number of objects contained in a single image and achieves the best performance of instance segmentation with the minimum label cost .** 

![image](./img/Fig2.pdf)

## Usage

## Installation

This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v1.0.0).The installation method is as follows, also can refer to the original [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

### Prerequisites
- Python 3.6.9
- Pytorch 1.8.1
- torchvision 0.9.1
- [mmcv 0.2.16](https://github.com/open-mmlab/mmcv/tree/v0.2.16)

Step-by-step installation

```bash
git clone https://github.com/licongguan/Lamer.git && cd Lamer
conda create -n Lamer python=3.6.9
conda activate Lamer
pip install -r requirements/build.txt
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI" 
pip install -v -e .  # or "python setup.py develop"
pip install mmcv==0.2.16
```
### A Quick Demo

Once the installation is done, you can download the provided [models](https://cloudstor.aarnet.edu.au/plus/s/dXz11J672ax0Z1Q/download) and use [inference_demo.py](demo/inference_demo.py) to run a quick demo.

### Data Preparation

- Download [The Cityscapes Dataset](https://www.cityscapes-dataset.com/)
Symlink the required dataset

### Label-Free Model Evaluation

- We provide relevant scripts in `Lamer/mmautoeval`.


### Label-Free Model Evaluation

- We provide relevant scripts in `Lamer/mmwcac`.

### Train SOLO

```shell
# Train with single GPU
python tools/train.py ${CONFIG_FILE}

# Example
python tools/train.py configs/solo/solo_r50_fpn_8gpu_1x.py
```

### Testing SOLO

```shell
# single-gpu testing
python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm

# Example
python tools/test_ins.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --out  results_solo.pkl --eval segm
```


## Contact

If you have any problem about our code, feel free to contact

- [lcguan941@bjtu.edu.cn](lcguan941@bjtu.edu.cn)

or describe your problem in Issues.
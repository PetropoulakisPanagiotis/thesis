[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-270/)

## <p align="center">Scale-Aware Landmark-based SLAM in Canonical Space</p>

<p align="center">
<img src="scale-brain.png" width="550px" height="300px"> 
</p>
<br /> <br />

----

## Installation
#### Depth Pipeline
```
conda create --name scale-env python=3.8
conda activate scale-env
conda install cudatoolkit=11.3
conda install -c nvidia cuda-nvml-dev
conda install cuda -c nvidia/label/cuda-11.3.0
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0  -c pytorch
pip install matplotlib  tqdm tensorboardX open3d matplotlib tqdm timm tensorboard tensorflow[and-cuda]
pip install -U openmim
mim install mmcv==2.0.0
```

#### Scale-Aware Pipeline

-------
#### Citation
If you find my thesis useful in your research, please consider citing:

```bib
@thesis{Petropoulakis2020,
    author      = {Petropoulakis Panagiotis, S. B. Laina, S. Schaefer, J. Jung, and S. Leutenegger},
    title       = {Scale-Aware Landmark-based SLAM in Canonical Space},
    type        = {mscthesis}
    url         = {https://github.com/PetropoulakisPanagiotis/BSc_thesis},
    institution = {TUM School of Computation, Information and Technology},
    year        = {2024},
}
```

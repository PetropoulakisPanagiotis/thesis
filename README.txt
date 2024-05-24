

SLAM commands:
python3 ptam.py --dataset scannet --path /home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted
python3 associate.py rgb.txt depth.txt > associations.txt
evo_traj tum --ref /home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/slam_gt_valid.txt ./results/slam_result.txt -as -p -v --full_check --t_offset 0
evo_ape tum --ref /home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/slam_gt.txt /home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/slam_result.txt -pa -v --t_offset 0.9
python2 evaluate.py /home/petropoulakis/Desktop/thesis/code/rgbd_dataset_freiburg1_room/groundtruth.txt /home/petropoulakis/Desktop/thesis/code/rgbd_dataset_freiburg1_room/slam_result.txt --verbose --fixed_delta --plot ./result.png

IEBINS commands:
python3 eval.py ../configs/arguments_eval_nyu_scale.txt
python3 train.py ../configs/scale_ins_u_1.txt


SCANNET commands:
nicr_sa_prepare_dataset scannet /usr/stud/petp/storage/user/petp/datasets/scannet /usr/stud/petp/storage/user/petp/datasets/scannet/data_converted
nicr_sa_prepare_dataset scannet /home/petropoulakis/Desktop/thesis/code/datasets/scannet/data /home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted 
python3 download.py --id scene0568_00 -o /usr/stud/petp/storage/user/petp/datasets/scannet/  


python -m pip install .[withpreparation,with3d] --user && pip install --upgrade numpy && pip install markupsafe==2.0.1 && pip install werkzeug==2.0.3
rm -r *.ply


NYU commands:
python3 extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../../datasets/nyu_depth_v2/official_splits/




PCS: 
Main 106: 
ssh -X -L 16006:127.0.0.1:6004 petp@131.159.19.194 -p 58022    


ssh -X -L 16006:127.0.0.1:6004 petp@131.159.19.192 -p 58022
ssh -X -L 16006:127.0.0.1:6004 petp@atcremers92.mlr.in.tum.de -p 58022
ssh -X -L 16006:127.0.0.1:6004 petp@atcremers91.mlr.in.tum.de -p 58022
ssh -X -L 16006:127.0.0.1:6004 petp@atcremers90.mlr.in.tum.de -p 58022
ssh -X -L 16006:127.0.0.1:6004 petp@atcremers99.vision.in.tum.de -p 58022
ssh -X -L 16006:127.0.0.1:6004 petp@atcremers39.in.tum.de -p 58022

Othe commands: 
nohup bash your_script.sh > output.txt 2>&1 &
export CUDA_VISIBLE_DEVICES=0
tensorboard --logdir=./ --port=6004
scp -P 58022 -r g2opy/ petp@131.159.19.194:~/code/thesis/



 
  


 

conda create -n iebins python=3.8
conda activate iebins
conda install cudatoolkit=11.3
conda install -c nvidia cuda-nvml-dev
conda install cuda -c nvidia/label/cuda-11.3.0
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0  -c pytorch
pip install matplotlib  tqdm tensorboardX open3d matplotlib tqdm timm tensorboard
pip install -U openmim
mim install mmcv
mim install mmcv==2.0.0
 









RGBD SLAM)
Before installing this: https://github.com/topics/rgbd-slam, follow the below:



--> https://github.com/uoip/g2opy
if an error looks like the following lines, use the below correction and compile again:
	.def("x", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::x)
	.def("y", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::y)
	.def("z", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::z)
	.def("w", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::w)
	
pip install pybind11	
sudo apt-get install libsuitesparse-dev
sudo apt install libglew-dev
sudo apt-get install python3.7-dev
sudo apt install python3.7-distutils
pip install setuptools==58.2.0
cmake -DPYBIND11_PYTHON_VERSION=3.9 ..
make -j8
python3 setup.py install

-----
https://github.com/uoip/pangolin

https://github.com/stevenlovegrove/Pangolin/pull/318/commits
https://github.com/uoip/pangolin/pull/46/commits/d840fa5644130826d6214b3c45e127782cc9bd9b
conda install -c conda-forge libstdcxx-ng=12
------

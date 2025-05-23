# GATIS

> This is the repo for the GATIS model. Please follow the instructions according to the writing order; you must set the environment first, then the weights and datasets, and then run the script for testing or training.

### environment setup

> Install the environment

```bash
conda env create -f config.yaml

# If the cmd of creation fails in the creation, you may run the following line to complete the installation. If not, you may skip it
conda env update -f config.yaml
```

> Add the environment variable

```bash
# you will need to first activate the enviroment to get the variable of environment correctly
conda activate gatis

# add the activate variable
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo '#!/bin/sh
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/
export NCCL_IB_DISABLE=1
export NCCL_PROTO=SIMPLE
' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# add the deactivate variable removal
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo '#!/bin/sh
unset CUDNN_PATH
unset LD_LIBRARY_PATH
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_DISABLE
unset NCCL_PROTO
' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# activat the environment to apply the variable
conda activate gatis
```

> Activate environment

```bash
conda activate gatis
```

### Datasets Download

```bash
# install the google drive downloading tool from pip library
pip install gdown
```

```bash
# https://drive.google.com/file/d/1Yp5NiMTHCWvWJl9VBQUzbBcfWlka5QGL/view?usp=drive_link
gdown 1w3UW8jg8zP9DVqD7_qOF9xmTZPAZs8AI # need to install the conda environment first, the gdown is the pip package
unzip datasets.zip && rm datasets.zip
```

> or you may down from the website (link)[https://drive.google.com/file/d/1w3UW8jg8zP9DVqD7_qOF9xmTZPAZs8AI/view?usp=sharing]

### Based Model weight and Best Model weight Download

```bash
# https://drive.google.com/file/d/1Tc1sv4uJaDCIekYXjcQw41ZJkXOTIIr6/view?usp=sharing
gdown 1Tc1sv4uJaDCIekYXjcQw41ZJkXOTIIr6 # need to install the conda environment first, the gdown is the pip package
unzip weights.zip && rm weights.zip
```

> or you may down from the website (link)[https://drive.google.com/file/d/1Tc1sv4uJaDCIekYXjcQw41ZJkXOTIIr6/view?usp=sharing]

#### Usage

1. use the following command to train the IOD-model with CISSL pretrained weight
   ```bash
   python train_downstream_iod.py
   ```
2. use the following command to test the IOD-model with pretrained weight best result
   ```bash
   python test_downstream_iod.py
   ```
3. use the following command to train the HAR-model with CISSL pretrained weight
   ```bash
   python train_downstream_har.py
   ```
4. use the following command to test the HAR-model with pretrained weight best result
   ```bash
   python test_downstream_har.py
   ```

### Datasets

1. [MotionID](https://paperswithcode.com/paper/motion-id-human-authentication-approach)
   > Due to the prolong training time for the based model weight of the motionID and large size of datasets, we provide the pretrained model along with its logs under the folers of `./lightning_logs`
2. [MotionSense](https://paperswithcode.com/dataset/motionsense)
   > Motionsense datasets are stored under the folder of of the `./datasets/MotionSense`.
3. [Migration](https://ieee-dataport.org/documents/migration)
   > Migration is our own datasets dedicated for the indoor outdoor identification. It is located under the `./datasets/migration`, there are two subfolder `./datasets/migration/migration` and `./datasets/migration/migration_wm`, which are the original file and rotation-to-world-frame-magnetic-removal respectively.
4. [ADVIO](https://github.com/AaltoVision/ADVIO)
   > ADVIO are preprocessed in similar fashion of Migration with `./datasets/ADVIO`, there are two subfolder `./datasets/ADVIO/advio` and `./datasets/ADVIO/advio_wm`, which are the original file and rotation-to-world-frame-magnetic-removal respectively.
5. [OxIOD](http://deepio.cs.ox.ac.uk/)
   > OxIOD are preprocessed in the similar fashion as Migeration in `./datasets/OIOD/processed`, which is in the global frame and remove the background magnetic measurement. Additionally all label are indoor.
6. [PINDWS](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/ZCBIIB)
   > PINDWS are preprocessed in the similar fashion as Migration in `./datasets/PINDWS/processed`. with the removal of entry 01 and 03 due to its hybrid environment without timeseries annotation.

### Logs

1. Base Model
   > The based model weight are stored under the folder of the `./lightning_logs` with tensorbaord logs.
2. Classification Model
   > The classification model weight are stored under the folder `./logs/IOD` and `logs/HAR` respectively. New testing or trainning result will be stored here as well.

### Visualization

1. To visualization the based model log
   ```bash
    tensorboard --logdir ./lightning_logs
   ```
2. To visualization the IOD log
   ```bash
   tensorboard --logdir ./logs/IOD
   ```
3. visualization the HAR log
   ```bash
   tensorboard --logdir ./logs/HAR
   ```

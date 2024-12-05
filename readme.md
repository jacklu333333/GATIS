# GATIS

> This is the repo for the GATIS model.


### environment setup
```bash
conda env create -f config.yaml
conda activate gatis
```

### Datasets Download
```bash
wget --no-check-certificate 'https://drive.google.com/file/d/1Yp5NiMTHCWvWJl9VBQUzbBcfWlka5QGL/view?usp=drive_link' -O datasets.zip
unzip datasets.zip && rm datasets.zip 
```

### Based Model weight and Best Model weight Download
```bash
wget --no-check-certificate 'https://drive.google.com/file/d/1Yp5NiMTHCWvWJl9VBQUzbBcfWlka5QGL/view?usp=drive_link' -O weights.zip
unzip weights.zip && rm weights.zip 
```


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

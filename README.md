# C3SM
Official code of the paper "Cross-sensor continual maritime object recognition via spectral shift modulation in heterogeneous infrared-visible UAV imagery"

## Installation
* PyTorch (>= 1.10.1, CUDA 11.3)

For detailed environment setup and dependencies, please refer to the installation guides in [avalanche](https://github.com/ContinualAI/avalanche) and [ReFusion](https://github.com/HaowenBai/ReFusion).

## Dataset 

Due to copyright restrictions, the UAVSensorCL dataset cannot be directly redistributed. Please obtain the raw data directly from the respective authors or official repositories of the original public datasets.

## Getting Started
1. Train a spectral shift parsing prior module
```
bash scripts/01_train_sspp.sh
```

2. Test a spectral shift parsing prior module
```
bash scripts/02_test_sspp.sh
```

3. Train a C3SM model
```
bash scripts/03_train_det_ckpt.sh
```

4. Test a C3SM model
```
bash scripts/04_test_det.sh
```

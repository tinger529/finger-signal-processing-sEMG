# EMG-based Finger Force Detector

Our project utilize EMG signals to distinguish which finger is extracting force. We provide four different settings models, as well as the training and testing dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Usage](#usage)
- [Features](#features)

## Project Overview

Our raw data was obtained with a machine for measuring electromyographic (EMG) signals, Datalink biometrics, and electrodes for signal reception. The setup is connected to a computer to monitor real-time signals and control variables such as sampling rate and reception duration.

The raw data was processed by filter_to_csv.py. This program utilizes butter band pass filter to perform denoising for the input txt file.

The dataset will be processed again in utilities for each model. We performed normalization to denoise again and split the training and testing datasets.

The four provided models are CNN, random forest, multi-label RF and multi-label SVM. You may compare their performance and evaluate the accuracy for each dataset.

Followings are some explanation of our and datasets.

**datasets** : denoised train, val data

    - dataset2 : weak single finger data
    - dataset3 : strong single finger data
    - dataset4 : strong multiple finger data

**test_datasets** : denoised test data

    - testdata1 : data with different gesture
    - testdata2 : single finger data (another student)
    - testdata3 : multiple finger data (another student)

## Usage

The data we provide already have denoised version. If you want to run new dataset, please run following command to perform denoising.

```
python3 filter_to_csv.py --input_file <your raw data> --output_file <new file pathname>
```

The code can be easily executed by clicking run all for each notebook. Note that if you want to change the dataset, please also modify its coresponding utilities code.

## Features

For multi label classification, we predict each input as a five dimension vector, indicating which fingers are extracting force. 


## Acknowledgements

Some of the codes are referenced from Advanced Control Laboratory, NTUEE, and [this repo](https://github.com/dwburer/sEMG-Neural-Net.git) from github.

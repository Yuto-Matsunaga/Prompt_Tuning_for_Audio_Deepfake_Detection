Generalized voice deep-fake detection using soft prompt-tuning.
===============
This repository is based on an implementation for [SOTA]((https://github.com/TakHemlata/SSL_Anti-spoofing)) to the existing ASVspoof2021 LA Dataset.  
Thanks for the great implementation!

## Installation in our environment
Our environment is python 3.8. Other libraries can be installed from the following.
```
pip install -r requirements.txt
git clone https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
```
Note: Not compatible with our WSP environment.
### Dataset
Our experiments are performed on the logical access (LA) partition of the ASVspoof 2021 dataset (train on 2019 LA training and evaluate on 2021 LA and DF evaluation database).

The ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The ASVspoof 2021 database is released on the zenodo site.

LA is [here](https://zenodo.org/record/4837263#.YnDIinYzZhE)

For ASVspoof 2021 dataset keys (labels) and metadata are available [here](https://www.asvspoof.org/index2021.html)

VCC2020 bonafide datasets is [here](https://github.com/nii-yamagishilab/VCC2020-database/tree/master)<br>

HABLA: A dataset of Latin American Spanish accents for voice anti-spoofing is [here](https://github.com/Ruframapi/HABLA)<br>

Original In-The-Wild dataset is [here](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild).<br>
However this dataset is not parted, so, we split the dataset into train: 40%, dev: 20% and eval 40%.<br>

Our split are as metadata in LA-keys/keys/CM.
ASVspoof 2021 LA metadata is trial_metadata.txt in this directory.

## Pre-trained wav2vec 2.0 XLSR (300M)
Download the XLSR models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

## Pre-trained SOTA model for ASVspoof2021 LA

Hemlata Tak et.al., provide a pre-trained models. To use it you can run: 

Pre-trained SSL antispoofing models are available for LA and DF [here](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB?usp=sharing)

EER: 0.82%, min t-DCF: 0.2066  on ASVspoof 2021 LA track.

EER: 2.85 % on ASVspoof 2021 DF track.

If you use pre-trained model, you should cite below.
```bibtex
@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```
## Overview
- prompt_tuning.py：The main code for prompt-tuning. The configuration file is[uniform_config.yaml](uniform_config.yaml)
  - Before running, set lines 333 to 340 in prompt_tuning.py to the directory where you are storing your data set
  ```
  python prompt_tuning.py --device {select your GPU (ex: cuda:0)} --dataset_name {choice  training dataset ['In_The_Wild', 'HABLA', ...]}
  ```
- hyparam_tuning.py：The main code for Hyperparameters tuning. The configuration file is[uniform_config.yaml](uniform_config.yaml)
  - Before running, set lines 290 to 297 in hyparam_tuning.py to the directory where you are storing your data set
  ```
  python hyparam_tuning.py --device {select your GPU (ex: cuda:0)} --dataset_name {choice  training dataset (ex:['In_The_Wild', 'HABLA', ...])} --optuna_study_name {File name of the db where the results of each trial's high para and evaluation function will be stored (ex: {dataset_name}_token{n_tokens}_sample{n_samples})}
  ```
## Experiments
The seed we used in the Experiments chapters in the paper is [1,42,57,123,256,345,567,890,1024,2345,3478,9999].
The hyperparameters used are stored in csv files in the results directory and correspond to {n_tokens},{n_samples},{batch_size},{beta},{lr},{weight_decay}.
{B or C}_wo_PT starts with {n_samples} because it does not use prompts.
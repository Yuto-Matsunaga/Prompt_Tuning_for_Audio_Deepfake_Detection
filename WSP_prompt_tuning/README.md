Generalized voice deep-fake detection using soft prompt-tuning.
===============
This repository is based on an implementation for Improved DeepFake Detection Using Whisper Features((https://github.com/piotrkawa/deepfake-whisper-features)).
Thanks for the great implementation!

## Installation in our environment
Our environment is python 3.8. Other libraries can be installed from the following.
```
pip install -r requirements.txt
```
Note: Not compatible with our W2V environment.

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

## Pre-trained Whisper
To download the whisper encoder model run `download_whisper.py`.

### Pretrained models
See the Improved DeepFake Detection Using Whisper Features repository to download pre-trained models.[here](https://github.com/piotrkawa/deepfake-whisper-features).
We used mesonet_whisper_mfcc_finetuned.pth.

If you use pre-trained model, you should cite below.
```bibtex
@inproceedings{kawa23b_interspeech,
  author={Piotr Kawa and Marcin Plata and Michał Czuba and Piotr Szymański and Piotr Syga},
  title={{Improved DeepFake Detection Using Whisper Features}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={4009--4013},
  doi={10.21437/Interspeech.2023-1537}
}
```
## Overview
- prompt_tuning.py：The main code for prompt-tuning. The configuration file is[uniform_config.yaml](uniform_config.yaml)
  - Before running, set lines 349 to 356 in prompt_tuning.py to the directory where you are storing your data set ```
  CUDA_VISIBLE_DEVICES={select your GPU number (ex:0)} python prompt_tuning.py --device cuda --dataset_name {choice  training dataset ['In_The_Wild', 'HABLA', ...]}
  ```
- hyparam_tuning.py：The main code for Hyperparameters tuning. The configuration file is[uniform_config.yaml](uniform_config.yaml)
  - Before running, set lines 299 to 306 in hyparam_tuning.py to the directory where you are storing your data set
  ```
  CUDA_VISIBLE_DEVICES={select your GPU number (ex:0)} python hyparam_tuning.py --device cuda --dataset_name {choice  training dataset (ex:['In_The_Wild', 'HABLA', ...])} --optuna_study_name {File name of the db where the results of each trial's high para and evaluation function will be stored (ex: {dataset_name}_token{n_tokens}_sample{n_samples})}
  ```
## Experiments
The seed we used in the Experiments chapters in the paper is [1,42,57,123,256,345,567,890,1024,2345,3478,9999].
The hyperparameters used are stored in csv files in the results directory and correspond to {n_tokens},{n_samples},{batch_size},{beta},{lr},{weight_decay}.
{B or C}_wo_PT starts with {n_samples} because it does not use prompts.
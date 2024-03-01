import os
import random
import yaml
import argparse
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, ConcatDataset
import optuna
from data_utils_SSL import genSpoof_list, Dataset_VCC_train, Dataset_ITW_train, Dataset_HABLA_eval_for_eer, Dataset_HABLA_VCC_eval,Dataset_HABLA_train_even,gen_atacklabels, Dataset_ITW_eval,Dataset_ASVspoof2021_train,Dataset_ASVspoof2021_eval
from utils import load_scheduler, make_foldername_by_seq
from loss import XentropyLoss
from optuna.pruners import MedianPruner
from prompt_tuning import load_model, train_one_epoch, dev_one_epoch, seed_worker
import math
import pdb
import numpy as np
import eval_metric_LA as em
from torch.utils.tensorboard import SummaryWriter
import datetime

writer = None
config = {}
train_set = None
dev_set = None
val_set = None
val_acc_set = None
samples_per_cls = []

def eval_to_score_file(score_file: str, cm_key_file: str, out_filepath: str, file_list = None,source_label=None):
    cm_data = pd.read_csv(cm_key_file, sep=' ', header=None)
    if(file_list):
        if type(cm_data[1][0]) is np.int64:
            cm_data[1] = cm_data[1].astype(str)
        cm_data = cm_data[cm_data[1].isin(file_list)]
    submission_scores = pd.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    # submission_scores
    if type(submission_scores[0][0]) is np.int64:
                submission_scores[0] = submission_scores[0].astype(str)
    if len(submission_scores) != len(cm_data):
        print(f'CHECK: submission has {len(submission_scores)} of {len(cm_data)} expected trials.')
        raise FileNotFoundError()

    # check here for progress vs eval set
    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=1, how='inner')
    if source_label =="LA":
        bona_cm = cm_scores[cm_scores[5]=='bonafide']['1_x'].values
        spoof_cm = cm_scores[cm_scores[5]=='spoof']['1_x'].values
    else:
        bona_cm = cm_scores[cm_scores[6]=='bonafide']['1_x'].values
        spoof_cm = cm_scores[cm_scores[6]=='spoof']['1_x'].values
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    print(f"eer: {100*eer_cm:.2f}\n", end="")
    with open(out_filepath, 'a+') as f:
        f.write(f'{eer_cm*100}\n')

    return eer_cm*100

def target_eer(config, model, data_loader,key_file = None):
    with torch.no_grad():
        save_path = os.path.join(config["result_path"], 'eer_scores.txt')
        uttID_list = []
        uttID_list_ap = uttID_list.extend
        score_list = []
        score_list_ap = score_list.extend
        running_corrects = 0
        num_total = 0
        for batch_x,utt_id,y in tqdm(data_loader):
            batch_x = batch_x.to(config["device"])
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]  
                        ).data.cpu().numpy().ravel()
            uttID_list_ap(utt_id)
            score_list_ap(batch_score.tolist())
            y = y.view(-1).type(torch.int64).to(config['device'])
            batch_size = batch_x.size(0)
            num_total += batch_size
            _, preds = batch_out.max(1)
            running_corrects += (preds == y).sum().item()
        with open(save_path, 'w') as fh:
            for f, cm in zip(uttID_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        epoch_acc = running_corrects/ num_total
        eer = eval_to_score_file(
            score_file=save_path,
            cm_key_file=key_file,
            out_filepath=os.path.join(config["result_path"], 'eer.txt'),file_list = uttID_list,source_label=config["dataset_name"])
    return eer, epoch_acc


def train(config, loss_fn, train_loader, dev_loader, model, optimizer, scheduler,val_loader, trial):
    # definition of optimizer and scheduler
    global writer
    d_today = datetime.date.today()
    str_today = d_today.strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter("runs/"+config["dataset_name"]+"/hypara/finetuning_sample"+str(config["n_samples"])+"_"+str_today)
    result_eer = 100
    trial_step = 0
    for epoch in range(config["epochs"]):
        train_epoch_loss = train_one_epoch(config, train_loader, model, optimizer, loss_fn)
        print(f'train epoch: {epoch}, loss: {train_epoch_loss}')
        with open(os.path.join(config["result_path"], f'train_loss{config["exp_name"]}.log'), 'a') as f:
            f.write(f'epoch: {epoch}, train_loss: {train_epoch_loss}\n')
        writer.add_scalar("Loss/train",train_epoch_loss,epoch+1)
        if (epoch+1) % 10 == 0:
            train_eer,train_acc = target_eer(config, model, dev_loader, key_file=config["train_meta"])
            print(f'train epoch: {epoch}, eer: {train_eer}')
            with open(os.path.join(config["result_path"], f'train_eer{config["exp_name"]}.log'), 'a') as f:
                f.write(f'epoch: {epoch}, train_eer: {train_eer}\n')
            dev_eer,val_acc = target_eer(config, model, val_loader, key_file=config["dev_meta"])
            if result_eer > dev_eer:
                result_eer = dev_eer
            print(f'dev epoch: {epoch}, eer: {dev_eer}')
            with open(os.path.join(config["result_path"], f'dev_eer{config["exp_name"]}.log'), 'a') as f:
                f.write(f'epoch: {epoch}, dev_eer: {dev_eer}\n')
            writer.add_scalar("EER/train",train_eer,epoch+1)
            writer.add_scalar("EER/validation",dev_eer,epoch+1)
            writer.add_scalar("ACC/train",train_acc,epoch+1)
            writer.add_scalar("ACC/validation",val_acc,epoch+1)
            trial.report(dev_eer, trial_step)
            trial_step = trial_step+1
            if trial.should_prune():
                raise optuna.TrialPruned()
        scheduler.step()
    writer.flush()
    writer.close()
    return result_eer



def objective(trial):
    global config
    global train_set
    global dev_set
    global val_set
    global val_acc_set
    global samples_per_cls
    # load model
    model = load_model(config, config["device"])
    model.eval()
    # decide tuned parameters
    target_params = []
    if "input_prompt" in config["target_params"]:
        for n, p in model.ssl_model.model.encoder.named_parameters():
            if n == "soft_prompt.weight":
                p.requires_grad = True
                target_params.append(p)
    if "MLP" in config["target_params"]:
        for n, p in model.named_parameters():
            if n == "out_layer.weight" or n == "out_layer.bias":
                p.requires_grad = True
                target_params.append(p)
    if "all" in config["target_params"]:
        target_params = []
        for n, p in model.named_parameters():
            p.requires_grad = True
            target_params.append(p)
    assert len(target_params) >= 1

    lr = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 5e-6, 5e-4, log=True)
    optimizer = torch.optim.Adam(params=target_params, lr=lr, weight_decay=weight_decay)
    scheduler = load_scheduler(config, optimizer)
    beta = trial.suggest_categorical('class-balanced-loss-beta', [0.99, 0.999, 0.9999])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    
    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False,
                            worker_init_fn=seed_worker)
    dev_loader = DataLoader(dev_set,
                            batch_size=16,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False,
                            worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set,
                        batch_size=16,
                        num_workers=2,
                        pin_memory=True,
                        shuffle=False,
                        drop_last=False,
                        worker_init_fn=seed_worker)

    loss_fn = XentropyLoss(2, samples_per_cls, beta, config["device"])
    eer = train(config, loss_fn=loss_fn, train_loader=train_loader, dev_loader=dev_loader, model=model, optimizer=optimizer, scheduler=scheduler,val_loader = val_loader, trial=trial)
    return eer 

def main(config):
    global train_set
    global dev_set
    global samples_per_cls
    global val_set
    global val_acc_set

    train_labels, train_file_eval = genSpoof_list(
            config["train_meta"],
            is_train=True,
            dataset_name=config["dataset_name"]
        )
    
    train_methods = gen_atacklabels(
            config["train_meta"],
            is_train=True,
            dataset_name=config["dataset_name"]
        )
    if config["dataset_name"] == 'HABLA':
        train_set = Dataset_HABLA_train_even(config, train_file_eval, train_labels, config["database_dir"],train_methods, n_samples=config["n_samples"])
    elif config["dataset_name"] in ['VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER']:
        train_set = Dataset_VCC_train(config, train_file_eval, train_labels, config["database_dir"],train_methods , n_samples=config["n_samples"])
    elif config["dataset_name"] == 'In_The_Wild':
        train_set = Dataset_ITW_train(config, train_file_eval, train_labels, config["database_dir"], n_samples=config["n_samples"])
    elif config["dataset_name"] == 'LA':
        train_set = Dataset_ASVspoof2021_train(config, train_file_eval, train_labels, config["database_dir"],n_samples=config["n_samples"])
    else:
        raise NotImplementedError

    samples_per_cls = []
    samples_per_cls.append(list(train_labels.values()).count(0)) # spoof counts
    samples_per_cls.append(list(train_labels.values()).count(1)) # bonafide counts

    if config["dataset_name"] == 'HABLA':
        dev_set = Dataset_HABLA_eval_for_eer(config, train_set.get_IDlist(), train_set.get_label(),config["database_dir"])
    elif config["dataset_name"] in ['VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER']:
        dev_set = Dataset_HABLA_VCC_eval(train_set.get_IDlist(), train_set.get_label(),config["database_dir"])
    elif config["dataset_name"] == 'In_The_Wild':
        dev_set = Dataset_ITW_eval(train_set.get_IDlist(), train_set.get_label(),config["database_dir"])
    elif config["dataset_name"] == 'LA':
        dev_set = Dataset_ASVspoof2021_eval(train_set.get_IDlist(), train_set.get_label(),config["database_dir"])
    else:
        raise NotImplementedError

    val_labels, val_file_eval = genSpoof_list(
            config["dev_meta"],
            is_dev=True,
            dataset_name=config["dataset_name"]
        )
    if config["dataset_name"] == 'HABLA':
        val_set = Dataset_HABLA_eval_for_eer(config, val_file_eval,val_labels, config["database_dir"], n_samples=config["n_samples"])
    elif config["dataset_name"] in ['VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER']:
        val_set = Dataset_HABLA_VCC_eval(val_file_eval,val_labels, config["database_dir"])
    elif config["dataset_name"] == 'In_The_Wild':
        val_set = Dataset_ITW_eval(val_file_eval,val_labels, config["database_dir"])
    elif config["dataset_name"] == 'LA':
        val_set = Dataset_ASVspoof2021_eval(val_file_eval,val_labels, config["database_dir"])
    else:
        raise NotImplementedError
    
    
    if config["optuna_study_name"] is not None:
        study_name = config["optuna_study_name"]
        storage_name = f'sqlite:///{study_name}.db'
        pruner = MedianPruner(n_startup_trials=25, n_warmup_steps=4)
        study = optuna.create_study(study_name=study_name, 
                                    storage=storage_name,
                                    load_if_exists=True,
                                    direction='minimize',
                                    pruner=pruner
                                    )
        study.optimize(objective, n_trials=2)
        print(study.best_trial)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True, choices=['HABLA', 'VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER', 'In_The_Wild','LA'])
    parser.add_argument('--optuna_study_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_tokens', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    main_args = parser.parse_args()

    # load uniform config params
    with open('uniform_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config["device"] = main_args.device
    config["optuna_study_name"] = main_args.optuna_study_name
    if main_args.n_tokens is not None: config["n_tokens"] = main_args.n_tokens
    if main_args.n_samples is not None: config["n_samples"] = main_args.n_samples
    if main_args.dataset_name is not None: config["dataset_name"] = main_args.dataset_name
    if main_args.seed != 42: config["seed"] = main_args.seed

    if config['dataset_name'] == 'HABLA':
        config["database_dir"] = "/path/to/your/directory/HABLA_dataset" # change for /path/to/your_dataset
    elif config['dataset_name'] in ['VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER']:
        config["database_dir"] = "/path/to/your/directory/VCC2020" # change for /path/to/your_dataset
    elif config["dataset_name"] == 'In_The_Wild':
        config["database_dir"] = "/path/to/your/directory/in_the_wild" # change for /path/to/your_dataset
    elif config["dataset_name"] == 'LA':
        config["database_dir"] = "/path/to/your/directory/ASVspoof2021_LA_eval/"
    else:
        raise NotImplementedError

    config["train_meta"] = f'LA-keys-stage-1/keys/CM/train_metadata_{config["dataset_name"]}.txt'
    config["dev_meta"] = f'LA-keys-stage-1/keys/CM/dev_metadata_{config["dataset_name"]}.txt'
    config["test_meta"] = f'LA-keys-stage-1/keys/CM/test_metadata_{config["dataset_name"]}.txt'
    if config["dataset_name"] == 'LA':
        config["train_meta"] = f'LA-keys-stage-1/keys/CM/trial_metadata.txt'
        config["dev_meta"] = f'LA-keys-stage-1/keys/CM/trial_metadata.txt'
        config["test_meta"] = f'LA-keys-stage-1/keys/CM/trial_metadata.txt'

    if not os.path.exists(os.path.join('results', 'fine_tuning', config["dataset_name"])):
        os.makedirs(os.path.join('results', 'fine_tuning', config["dataset_name"]))
    config["exp_name"] = make_foldername_by_seq(os.path.join('results', 'fine_tuning', config["dataset_name"]), prefix='exp', seq_digit=3)
    result_path = os.path.join('results', 'fine_tuning', config["dataset_name"], config["exp_name"])
    config["result_path"] = result_path
    
    # fix seed
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(config["seed"])
    if config["n_samples"] < 0:
        config["n_samples"] = None
    # torch.set_num_threads(4)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        raise FileExistsError
    main(config)
import os
import random
import json
import argparse
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from tensorboard import SummaryWriter
from model import Model
from data_utils_SSL import genSpoof_list, gen_atacklabels,genSpoof_list_LA19, Dataset_HABLA_train, Dataset_VCC_train, Dataset_ASVspoof2021_train
from data_utils_SSL import Dataset_ITW_train, Dataset_HABLA_VCC_eval, Dataset_ITW_eval, Dataset_HABLA_train_even, Dataset_ASVspoof2021_eval
import eval_metric_LA as em
from utils import load_scheduler, make_foldername_by_seq
from loss import XentropyLoss
import time
from src.models import models
import datetime
from tensorboardX import SummaryWriter

def seed_worker(worker_id):
    """
    The seed fixed method for DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def eval_to_score_file(score_file: str, cm_key_file: str, out_filepath: str, file_list = None,source_label=None):
    cm_data = pd.read_csv(cm_key_file, sep=' ', header=None)
    # print(len(cm_data))
    # print(file_list)
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

def produce_evaluation_file(config,dataset, model,key_file = None):
    data_loader = DataLoader(dataset, 
                             batch_size=32, 
                             num_workers=2, 
                             shuffle=False, 
                             drop_last=False, 
                             worker_init_fn=seed_worker)
    model.eval() 
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
            batch_score = (batch_out[:, 0]  
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

def eval(config, model):
    eval_label, eval_file_eval = genSpoof_list(
            config["test_meta"],
            is_eval=True,
            dataset_name=config["dataset_name"]
        )
    if config['dataset_name'] in ['HABLA', 'VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER']:
        eval_set = Dataset_HABLA_VCC_eval(eval_file_eval,eval_label, config["database_dir"])
    elif config['dataset_name'] == 'In_The_Wild':
        eval_set = Dataset_ITW_eval(eval_file_eval,eval_label, config["database_dir"])
    elif config['dataset_name'] == 'LA':
        eval_set = Dataset_ASVspoof2021_eval(eval_file_eval,eval_label, config["database_dir"])
    else:
        raise NotImplementedError
    print('create score file...')
    eer_cm,_ = produce_evaluation_file(config, eval_set, model,key_file=config["test_meta"])
    return eer_cm


def dev_one_epoch(config, dev_loader, model, loss_fn, loss_fn_source=None, source_dev_loader=None):
    with torch.no_grad():
        running_loss = 0.0
        num_total = 0
        for x, y in tqdm(dev_loader):
            x = x.to(config['device'])
            y = y.view(-1).type(torch.float32).to(config['device'])
            batch_size = x.size(0)
            num_total += batch_size
            output = model(x)
            batch_loss = loss_fn(output[:,0], y)
            running_loss += (batch_loss.item() * batch_size)
        running_loss /= num_total
    return running_loss


def train_one_epoch(config, train_loader, model, optimizer, loss_fn, loss_fn_source=None, source_train_loader=None, alpha = None):
    running_loss = 0.0
    sum_target_loss = 0.0
    sum_source_loss = 0.0
    num_total = 0
    for x, y in tqdm(train_loader):
        x = x.to(config['device'])
        y = y.view(-1).type(torch.float32).to(config['device'])
        batch_size = x.size(0)
        num_total += batch_size
        output = model(x)
        batch_loss = loss_fn(output[:,0], y)
        running_loss += (batch_loss.item() * batch_size)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    running_loss /= num_total
    return running_loss


def train(config, loss_fn, train_loader, dev_loader, model, optimizer, scheduler, loss_fn_source=None, source_train_loader=None ,source_dev_loader=None):
    # # use TensorBoard
    d_today = datetime.date.today()
    str_today = d_today.strftime('%Y%m%d%H%M%S')
    writer = SummaryWriter("runs/"+config["dataset_name"]+"/hypara/finetuning_sample"+str(config["n_samples"])+"_"+str_today)
    for epoch in range(config["epochs"]):
        train_epoch_loss = train_one_epoch(config, train_loader, model, optimizer, loss_fn)
        print(f'train epoch: {epoch}, loss: {train_epoch_loss}')
        writer.add_scalars('Loss/train_loss', {'train_epoch_loss': train_epoch_loss}, epoch)
        with open(os.path.join(config["result_path"], f'train_loss{config["exp_name"]}.log'), 'a') as f:
            f.write(f'epoch: {epoch}, train_loss: {train_epoch_loss}\n')

        dev_epoch_loss = dev_one_epoch(config, dev_loader, model, loss_fn)
        print(f'dev epoch: {epoch}, loss: {dev_epoch_loss}')
        writer.add_scalars('Loss/dev_loss', {'dev_epoch_loss': dev_epoch_loss}, epoch)
        with open(os.path.join(config["result_path"], f'dev_loss{config["exp_name"]}.log'), 'a') as f:
            f.write(f'epoch: {epoch}, dev_loss: {dev_epoch_loss}\n')
        
        scheduler.step()
    writer.flush()
    writer.close()
    return dev_epoch_loss


def load_model(config, device):
    with open('/home/y-matsunaga/deepfake-whisper-features/paper_models/mesonet_whisper_mfcc_finetuned/config.yaml', 'r') as f:
        tmp_config = yaml.safe_load(f)
    model_config = tmp_config["model"]
    model =  models.get_model(
        model_name=model_config["name"],
        config=model_config["parameters"],
        device=device,n_token=config["n_tokens"],token_label=config["with_prompt"]
    )
    model.load_state_dict(torch.load(tmp_config["checkpoint"]["path"], map_location=config["device"]), strict=False)
    model = model.to(config["device"])
    for param in model.parameters():
        param.requires_grad = False
    return model


def save_tuned_module(config, model):
    if "input_prompt" in config["target_params"]:
        if not os.path.exists(os.path.join('./prompts_ckpt', config["dataset_name"])):
            os.makedirs(os.path.join('./prompts_ckpt', config["dataset_name"]))
        save_path = os.path.join('./prompts_ckpt', config["dataset_name"])
        filename = f'{config["dataset_name"]}_{config["exp_name"]}.model'
        torch.save(model.soft_prompt, os.path.join(save_path, filename))
        print(f"Saved soft prompt: {os.path.join(save_path, filename)}")
    
    if "MLP" in config["target_params"]:
        if not os.path.exists(os.path.join('./mlp_ckpt', config["dataset_name"])):
            os.makedirs(os.path.join('./mlp_ckpt', config["dataset_name"]))
        torch.save(model.fc2, os.path.join('./mlp_ckpt', config["dataset_name"], f'{config["dataset_name"]}_{config["exp_name"]}.model'))
    
    # # All parameters require 1.2GB/model
    # if "all" in config["target_params"]:
    #     if not os.path.exists(os.path.join('./all_ckpt', config["dataset_name"])):
    #         os.makedirs(os.path.join('./all_ckpt', config["dataset_name"]))
    #     torch.save(model.state_dict(), os.path.join('./all_ckpt', config["dataset_name"], f'{config["dataset_name"]}_{config["exp_name"]}.model'))


def main(config):
    with open(os.path.join(result_path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    model = load_model(config, config["device"])
    model.eval()

    # decide tuned parameters
    target_params = []
    if "input_prompt" in config["target_params"]:
        for n, p in model.named_parameters():
            if n == "soft_prompt.weight":
                p.requires_grad = True
                target_params.append(p)
    if "MLP" in config["target_params"]:
        for n, p in model.named_parameters():
            if n == "fc2.weight" or n == "fc2.bias":
                p.requires_grad = True
                target_params.append(p)
    if "all" in config["target_params"]:
        target_params = []
        for n, p in model.named_parameters():
            p.requires_grad = True
            target_params.append(p)
    assert len(target_params) >= 1
    print(f'no. of target prams: {len(target_params)}')
    
    # load training dataset
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
    print('no. of train utts.', len(train_set))
    train_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            num_workers=2,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False,
                            worker_init_fn=seed_worker)
    
    # load development dataset
    dev_labels, dev_file_eval = genSpoof_list(
        config["dev_meta"],
        is_dev=True,
        dataset_name=config["dataset_name"]
    )
    if config["dataset_name"] == 'HABLA':
        dev_set = Dataset_HABLA_train(config, dev_file_eval, dev_labels, config["database_dir"],n_samples=config["n_samples"])
    elif config["dataset_name"] in ['VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER']:
        dev_set = Dataset_VCC_train(config, dev_file_eval, dev_labels, config["database_dir"])
    elif config["dataset_name"] == "In_The_Wild":
        dev_set = Dataset_ITW_train(config, dev_file_eval, dev_labels, config["database_dir"],n_samples=config["n_samples"])
    elif config["dataset_name"] == 'LA':
        dev_set = Dataset_ASVspoof2021_train(config, dev_file_eval, dev_labels, config["database_dir"],n_samples=config["n_samples"])
    else:
        raise NotImplementedError
    print('no. of dev utts.', len(dev_set))
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            num_workers=2,
                            shuffle=False,
                            drop_last=False,
                            worker_init_fn=seed_worker)
    
    # definition of optimizer and scheduler
    optimizer = torch.optim.Adam(target_params,
                                lr=config["lr"],
                                weight_decay=config["weight_decay"])
    scheduler = load_scheduler(config, optimizer)
    
    # definition of loss function
    if config["loss_type"] == "CBLoss": # use Class-Balanced Loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        samples_per_cls = []
        samples_per_cls.append(list(train_labels.values()).count(0)) # spoof counts
        samples_per_cls.append(list(train_labels.values()).count(1)) # bonafide counts
        loss_fn = XentropyLoss(2, samples_per_cls, config["beta"], config["device"])
    elif config["loss_type"] == "CELoss": # use Cross Entropy Loss
        weight = torch.FloatTensor(config["CEL_weight"]).to(config["device"])
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    else:
        raise NotImplementedError
    
    # start training 
    train(config, loss_fn, train_loader, dev_loader, model, optimizer, scheduler)
    save_tuned_module(config, model)
    eer_cm = eval(config, model)
    return eer_cm
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True, choices=['HABLA', 'VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER', 'In_The_Wild','LA'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_tokens', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--target_params', type=str, nargs="*", default=None)
    parser.add_argument('--beta', type=float, default=None, help='for Class Balanced loss')
    main_args = parser.parse_args()

    # load uniform config params
    with open('uniform_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # variable parameters priortize main_args
    config["device"] = main_args.device
    config["dataset_name"] = main_args.dataset_name
    if main_args.n_tokens is not None: config["n_tokens"] = main_args.n_tokens
    if main_args.n_samples is not None: config["n_samples"] = main_args.n_samples
    if main_args.seed != 42: config["seed"] = main_args.seed
    if main_args.target_params is not None: config["target_params"] = main_args.target_params
    if main_args.beta is not None: config["beta"] = main_args.beta

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

    # change for /path/to/your_dataset
    config["train_meta"] = f'LA-keys-stage-1/keys/CM/train_metadata_{config["dataset_name"]}.txt'
    config["dev_meta"] = f'LA-keys-stage-1/keys/CM/dev_metadata_{config["dataset_name"]}.txt'
    config["test_meta"] = f'LA-keys-stage-1/keys/CM/test_metadata_{config["dataset_name"]}.txt'
    if config["dataset_name"] == 'LA':
        config["train_meta"] = f'LA-keys-stage-1/keys/CM/trial_metadata.txt'
        config["dev_meta"] = f'LA-keys-stage-1/keys/CM/trial_metadata.txt'
        config["test_meta"] = f'LA-keys-stage-1/keys/CM/trial_metadata.txt'

    # The experimental folder name is determined dynamically.
    # Automatically generates a sequential number of 3-digit numbers.
    if not os.path.exists(os.path.join('results', 'prompt_tuning', config["dataset_name"])):
        os.makedirs(os.path.join('results', 'prompt_tuning', config["dataset_name"]))
    config["exp_name"] = make_foldername_by_seq(os.path.join('results', 'prompt_tuning', config["dataset_name"]), prefix='exp', seq_digit=3)
    result_path = os.path.join('results', 'prompt_tuning', config["dataset_name"], config["exp_name"])
    config["result_path"] = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        raise FileExistsError
    
    # fix seed
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(config["seed"])

    # good luck!!
    if config["eval_only"]:
        model = load_model(config, config["device"])
        eer_cm = eval(config, model)
    else:
        main(config)

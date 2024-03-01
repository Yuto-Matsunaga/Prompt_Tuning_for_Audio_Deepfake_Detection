import glob
import os
import re
import torch

def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield file[:-len(extension)]


def load_scheduler(args, optimizer):
    if args["lr_scheduler_type"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["epochs"])
    # elif args["lr_scheduler_type"] == "linear":
    #     return torch.optim.lr_scheduler.LinearLR(optimizer,
    #                                             start_factor=args["lr"],
    #                                             end_factor=args["lr"]/5,
    #                                             total_iters=args["epochs"])
    else:
        raise NotImplementedError()
       

def make_foldername_by_seq(dirname, prefix='exp', seq_digit=3):
    if not os.path.isdir(dirname):
        raise FileNotFoundError
    
    pattern = f"{prefix}([0-9]*)"
    prog = re.compile(pattern)

    folders = glob.glob(
        os.path.join(dirname, f"{prefix}[0-9]*")
    )

    max_seq = -1
    for f in folders:
        m = prog.match(os.path.basename(f))
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    
    new_foldername = f"{prefix}{max_seq+1:0{seq_digit}}"
    
    return new_foldername
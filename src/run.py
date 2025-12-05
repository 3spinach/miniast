# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Adapted from Yuan Gong's AST with MiniViT framework
# @File    : run.py

"""
Run script for MiniAST with weight multiplexing (without distillation).
"""

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from miniast_models import MiniASTModel
from ast_models import ASTModel
import numpy as np
from traintest import train, validate

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data arguments
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='miniast', help="model type", 
                    choices=['ast', 'miniast'])
parser.add_argument("--dataset", type=str, default="audioset", help="dataset name")

# Experiment arguments
parser.add_argument("--exp-dir", type=str, default="", help="experiment directory")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, 
                    metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="optimizer", 
                    choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', 
                    help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', 
                    help='# of workers for dataloading')
parser.add_argument("--n-epochs", type=int, default=1, help="number of training epochs")
parser.add_argument("--lr_patience", type=int, default=2, 
                    help="epochs to wait before reducing lr")
parser.add_argument("--n-print-steps", type=int, default=100, 
                    help="steps between printing statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

# Data augmentation arguments
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, 
                    help="mixup ratio during training (0-1)")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling")

# Model architecture arguments
parser.add_argument("--fstride", type=int, default=10, 
                    help="frequency stride for patch splitting")
parser.add_argument("--tstride", type=int, default=10, 
                    help="time stride for patch splitting")
parser.add_argument('--imagenet_pretrain', help='use ImageNet pretrained model', 
                    type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='use AudioSet pretrained model', 
                    type=ast.literal_eval, default='False')

# Dataset statistics
parser.add_argument("--dataset_mean", type=float, default=-4.2677393, 
                    help="dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, 
                    help="dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, 
                    help="target audio length in frames")
parser.add_argument('--noise', help='add noise augmentation', 
                    type=ast.literal_eval, default='False')

# Training settings
parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", 
                    choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", 
                    choices=["BCE", "CE"])
parser.add_argument('--warmup', help='warmup learning rate', 
                    type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, 
                    help="epoch to start lr decay")
parser.add_argument("--lrscheduler_step", type=int, default=1, 
                    help="epochs per lr decay step")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, 
                    help="lr decay rate")

# Weight averaging
parser.add_argument('--wa', help='use weight averaging', 
                    type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, 
                    help="epoch to start weight averaging")
parser.add_argument('--wa_end', type=int, default=5, 
                    help="epoch to end weight averaging")

# ========== MiniViT-specific arguments ==========
parser.add_argument("--num_shared_layers", type=int, default=2,
                    help="number of consecutive layers to share weights")
parser.add_argument('--use_attn_transform', help='use attention transformation',
                    type=ast.literal_eval, default='True')
parser.add_argument('--use_mlp_transform', help='use MLP transformation',
                    type=ast.literal_eval, default='True')
parser.add_argument("--mlp_kernel_size", type=int, default=7,
                    help="kernel size for MLP depth-wise convolution")

args = parser.parse_args()

# Audio configuration
audio_conf = {
    'num_mel_bins': 128, 
    'target_length': args.audio_length, 
    'freqm': args.freqm, 
    'timem': args.timem, 
    'mixup': args.mixup, 
    'dataset': args.dataset, 
    'mode': 'train', 
    'mean': args.dataset_mean, 
    'std': args.dataset_std,
    'noise': args.noise
}

val_audio_conf = {
    'num_mel_bins': 128, 
    'target_length': args.audio_length, 
    'freqm': 0, 
    'timem': 0, 
    'mixup': 0, 
    'dataset': args.dataset, 
    'mode': 'evaluation', 
    'mean': args.dataset_mean, 
    'std': args.dataset_std, 
    'noise': False
}

# Create data loaders
if args.bal == 'bal':
    print('Using balanced sampler')
    samples_weight = np.loadtxt(args.data_train[:-5] + '_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, 
                                   audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, 
        num_workers=args.num_workers, pin_memory=True)
else:
    print('Not using balanced sampler')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, 
                                   audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, 
                               audio_conf=val_audio_conf),
    batch_size=args.batch_size * 2, shuffle=False, 
    num_workers=args.num_workers, pin_memory=True)

# Create model
if args.model == 'miniast':
    print('Training MiniAST model with weight multiplexing')
    print(f'  - Shared layers: {args.num_shared_layers}')
    print(f'  - Attention transform: {args.use_attn_transform}')
    print(f'  - MLP transform: {args.use_mlp_transform}')
    
    audio_model = MiniASTModel(
        label_dim=args.n_class, 
        fstride=args.fstride, 
        tstride=args.tstride, 
        input_fdim=128,
        input_tdim=args.audio_length, 
        imagenet_pretrain=args.imagenet_pretrain,
        audioset_pretrain=args.audioset_pretrain, 
        model_size='base384',
        num_shared_layers=args.num_shared_layers,
        use_attn_transform=args.use_attn_transform,
        use_mlp_transform=args.use_mlp_transform,
        mlp_kernel_size=args.mlp_kernel_size
    )
else:
    print('Training original AST model')
    audio_model = ASTModel(
        label_dim=args.n_class, 
        fstride=args.fstride, 
        tstride=args.tstride, 
        input_fdim=128,
        input_tdim=args.audio_length, 
        imagenet_pretrain=args.imagenet_pretrain,
        audioset_pretrain=args.audioset_pretrain, 
        model_size='base384'
    )

# Create experiment directory
print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir, exist_ok=True)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

# Print model info
total_params = sum(p.numel() for p in audio_model.parameters())
trainable_params = sum(p.numel() for p in audio_model.parameters() if p.requires_grad)
print(f'\nModel parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable')

# Start training
print(f'\nStarting training for {args.n_epochs} epochs')
train(audio_model, train_loader, val_loader, args)

# Evaluate on test set for speechcommands
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)
    
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------Evaluate on validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))
    
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, 
                                   audio_conf=val_audio_conf),
        batch_size=args.batch_size * 2, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True)
    
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------Evaluate on test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

print('\nTraining completed!')
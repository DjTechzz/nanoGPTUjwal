# config/train_enwik8_modified.py

import os
import time

device = 'mps'  # Use MPS for M-series Macs
dtype = 'float32'  # MPS doesn't support bfloat16, so we'll use float32
compile = False  # Disable compilation as it's not supported for MPS
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Dataset-specific configurations
dataset = 'enwik8'
data_dir = os.path.join('data', dataset)
train_data = os.path.join(data_dir, 'train.bin')
val_data = os.path.join(data_dir, 'val.bin')
test_data = os.path.join(data_dir, 'test.bin')

# Read vocab size from the prepared data
with open(os.path.join(data_dir, 'vocab.txt'), 'r') as f:
    vocab_size = len(f.readlines())

# Model configurations
block_size = 1024  # Context size
n_layer = 12
n_head = 16  # Increased from 8 to allow for more dynamic pruning
n_embd = 768  # Increased from 512 to compensate for potential pruning
dropout = 0.1
bias = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

# ASADP-specific configurations
min_heads = 4  # Minimum number of heads to keep active

# Training configurations
batch_size = 64
max_iters = 150000  # Increased to allow for potential slower convergence
eval_interval = 500
log_interval = 100
eval_iters = 200
learning_rate = 5e-4  # Slightly increased to potentially speed up learning
lr_decay_iters = 150000
min_lr = 5e-5
beta1 = 0.9
beta2 = 0.95  # Changed to 0.95 which can sometimes work better for adaptive methods
grad_clip = 1.0

# Gradient accumulation
gradient_accumulation_steps = 1

# Logging and checkpointing
out_dir = 'out-enwik8-asadp'
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # Changed to True to ensure we save all checkpoints

# Wandb logging
wandb_log = True  # Enabled for better experiment tracking
wandb_project = 'enwik8-asadp'
wandb_run_name = 'gpt2-enwik8-asadp-' + str(time.time())

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.



import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import json
import numpy as np
import pickle
from sklearn.utils import shuffle
import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default="../input/pickled/train_aug_gpt_256.pkl")
parser.add_argument('--devices', default='0')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=16)
parser.add_argument('--epochs', type=int, default=7)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--model', type=str, default="gpt2")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.optim import Adagrad, Adamax
from transformers.modeling_utils import * 
from scipy.stats import spearmanr
from utils import *
from models import *

SEED = args.seed
EPOCHS = args.epochs 

lr=args.lr
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
tokenizer = GPT2Tokenizer.from_pretrained(args.model)
model = GPT2ForTextExtraction.from_pretrained(args.model, output_hidden_states=True)
pad_token_id = tokenizer.eos_token_id
_ = model.cuda()
print(f"Loading data from {args.train_path}")
X_train, X_type_train, X_pos_train, X_offset_train = pickle.load(open(args.train_path.format(0),"rb"))
X_train, X_type_train, X_pos_train, X_offset_train = shuffle(X_train, X_type_train, X_pos_train, X_offset_train)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
   ]

num_train_optimization_steps = int(EPOCHS*len(X_train)/batch_size/accumulation_steps)

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False) 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*num_train_optimization_steps, num_training_steps=num_train_optimization_steps) 
scheduler0 = get_constant_schedule(optimizer)

from tqdm.auto import tqdm

best_score = 0
tq = tqdm(range(args.epochs + 1))
model.freeze()
frozen = True
for epoch in tq:
    if epoch > 0 and frozen:
        model.unfreeze()
        frozen = False
        del scheduler0
        torch.cuda.empty_cache()
    X_train, X_type_train, X_pos_train, X_offset_train = pickle.load(open(args.train_path.format(np.random.randint(args.epochs + 1)),"rb"))
    X_train, X_type_train, X_pos_train, X_offset_train = shuffle(X_train, X_type_train, X_pos_train, X_offset_train)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train,dtype=torch.long),torch.tensor(X_type_train,dtype=torch.long),torch.tensor(X_pos_train[:, 0],dtype=torch.long), torch.tensor(X_pos_train[:, 1],dtype=torch.long), torch.tensor(X_offset_train,dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
    model.train()
    for i, items in pbar:
        x_batch, x_type_batch, x_start_batch, x_end_batch,x_offset_batch = items
        p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
        p_mask[x_batch == pad_token_id] = 1.0
        p_mask[:,0] = 0.0
        p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
        attention_mask=(x_batch != pad_token_id)
        attention_mask[:,0] = True
        loss,start_loss,end_loss = model(input_ids=x_batch.cuda(), start_positions = x_start_batch.cuda(), end_positions = x_end_batch.cuda(), attention_mask=attention_mask.cuda(),p_mask=p_mask.cuda()) #,cls_ids=y_batch)
        loss /= accumulation_steps
        loss = loss.mean()
        loss.backward()
        if i % accumulation_steps == 0 or i == len(pbar) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if not frozen:
                scheduler.step()
        pbar.set_postfix(loss = loss.item()*accumulation_steps, start_loss = start_loss.mean().item(),end_loss = end_loss.mean().item())
    torch.save(model.state_dict(),f"./models/{args.model}_{SEED}.bin")

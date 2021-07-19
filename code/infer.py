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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--input_path', type=str, default=None)
parser.add_argument('--model', type=str, default="gpt2")
parser.add_argument('--sigmoid_decoding', action="store_true")
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--threshold', type=float, default=-1)
parser.add_argument('--best_only', action="store_true")
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--overlap', type=int, default=32)
parser.add_argument('--devices', default='0')
args = parser.parse_args()
parser = argparse.ArgumentParser(description='Process some integers.')
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

from torch import nn
from transformers import *
import torch
import torch.utils.data
import torch.nn.functional as F
from transformers import *
from torch import nn
import math
from models import *

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
model = GPT2ForTextExtraction.from_pretrained(args.model, output_hidden_states=True)
pad_token_id = tokenizer.eos_token_id

model.eval()
model.load_state_dict(torch.load(args.ckpt_path))
_ = model.cuda()

def get_input_ids_from_text(input_path):
    article = open(input_path).read()
    article = " ".join(article.replace("\n", " ").split())
    test_df = pd.DataFrame()
    texts = []
    labels = []
    input_ids = tokenizer.encode(article, add_special_tokens=False)
    n_samples = math.ceil(len(input_ids)/(args.max_sequence_length - args.overlap))
    for sample_idx in range(n_samples):
        start = max(0, (args.max_sequence_length - args.overlap)*sample_idx)
        end = start + args.max_sequence_length
        curr_ids = input_ids[start: end]
        curr_text = tokenizer.decode(curr_ids)
        texts.append(curr_text)
    test_df["text"] = texts
    test_df = test_df.fillna("")
    X, _ = convert_lines(tokenizer,test_df,is_test=True, max_sequence_length=args.max_sequence_length)
    return X

X_test = get_input_ids_from_text(args.input_path)

selected_texts = dict()
with torch.no_grad():
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test,dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4*args.batch_size, shuffle=False)
    pbar = tqdm(enumerate(test_loader),total=len(test_loader),leave=False)
    for i, items in pbar:
        x_batch, = items
        p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
        p_mask[x_batch == pad_token_id] = 1.0
        p_mask[:,0] = 0.0
        attention_mask=(x_batch != pad_token_id).cuda()
        attention_mask[:,0] = True
        start_top_log_probs, start_top_index, end_top_log_probs, end_top_index = model(input_ids=x_batch.cuda(), attention_mask= attention_mask \
                ,p_mask = p_mask.cuda(), beam_size=args.beam_size, sigmoid_decoding=args.sigmoid_decoding)

        start_top_log_probs = start_top_log_probs.cpu().numpy()
        end_top_log_probs = end_top_log_probs.cpu().numpy()
        start_top_index = start_top_index.cpu().numpy()
        end_top_index = end_top_index.cpu().numpy()
        for i_, x in enumerate(x_batch):
            x = x.numpy()
            real_length = np.sum(x != pad_token_id)
            valid_start = 0
            (best_start, best_end), score_dict = find_best_combinations(start_top_log_probs[i_], start_top_index[i_], \
                                                            end_top_log_probs[i_].reshape(args.beam_size,args.beam_size), end_top_index[i_].reshape(args.beam_size,args.beam_size), \
                                                            valid_start = valid_start, valid_end = real_length)
            context = tokenizer.decode([w for w in x if w != pad_token_id])
            selected_texts[context] = []
            # Threshold matching mode
            if args.threshold > 0:
                if args.best_only:
                    if np.sqrt(score_dict.get((best_start, best_end), -1)) > args.threshold:
                        selected_text = tokenizer.decode([w for w in x[best_start:best_end] if w != pad_token_id], clean_up_tokenization_spaces=False)
                        selected_texts[context].append(selected_text.strip())
                else:
                    for key, val in score_dict.items():
                        if np.sqrt(val) > args.threshold:
                            selected_text = tokenizer.decode([w for w in x[key[0]:key[1]] if w != pad_token_id], clean_up_tokenization_spaces=False)
                            selected_texts[context].append(selected_text.strip())
            else:
                selected_text = tokenizer.decode([w for w in x[best_start:best_end] if w != pad_token_id], clean_up_tokenization_spaces=False)
                if len(selected_text.strip()) > 0:
                    selected_texts[context].append(selected_text.strip())
print("\nOutput:\n")
for key, vals in selected_texts.items():
    for val in vals:
        if len(val.strip()) > 0 and is_valid_pred(val):
            begin = key.index(val)
            end = begin + len(val)
            val = Colour.GREEN + (" >> " + val + " << ") + Colour.END
            output = "..."+key[max(0, begin - 40):begin] + val + key[end: end+ 40]+"..."
            print(output)

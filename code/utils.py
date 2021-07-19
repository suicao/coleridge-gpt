from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from types import SimpleNamespace
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm 
import regex as re
import html
import operator
from fuzzywuzzy import fuzz
import itertools
from scipy.special import softmax
import math 

from typing import List
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _jaccard_similarity(str1: str, str2: str) -> float:
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_fbeta(y_true, y_pred, beta=0.5, cosine_th=0.2):
    """Compute the Jaccard-based micro FBeta score.

    References
    ----------
    - https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/overview/evaluation
    """
    import copy

    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative
    fp_list = []
    y_true_copy = copy.deepcopy(y_true)
    y_pred_copy = copy.deepcopy(y_pred)
    for ground_truth_list, predicted_string_list in zip(y_true_copy, y_pred_copy):
        predicted_string_list_sorted = sorted(predicted_string_list)
        n_predicted_string = len(predicted_string_list_sorted)
        n_gt_string = len(ground_truth_list)
        if n_gt_string > n_predicted_string:
            fn += n_gt_string - n_predicted_string
        elif n_gt_string < n_predicted_string:
            fp += n_predicted_string - n_gt_string
            fp_list.extend(predicted_string_list_sorted[-(n_predicted_string - n_gt_string):])

        start_idx = 0
        N = min(n_gt_string, n_predicted_string)
        while start_idx < N:
            # find nearest groundtruth to match with predicted_string
            predicted_string = predicted_string_list_sorted[start_idx]
            jaccard_with_gt = [
                _jaccard_similarity(predicted_string, ground_truth_list[i])
                for i in range(len(ground_truth_list))
            ]
            best_matched_gt_idx = np.argmax(jaccard_with_gt)
            if jaccard_with_gt[best_matched_gt_idx] >= 0.5:
                tp += 1
            else:
                fp += 1
                fp_list.append(predicted_string)
            start_idx += 1
            ground_truth_list.pop(best_matched_gt_idx)

    raw_values = [tp, fp, fn]
    print(tp, fp, fn)
    tp *= 1 + beta ** 2
    fn *= beta ** 2
    fbeta_score = tp / (tp + fp + fn)
    return fbeta_score , raw_values, fp_list

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()

def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return None, None

def get_sub_idx(x, y): 
    best_span = contains(y,x)
    if best_span is not None:
        return best_span
#     print("oh noes")
#     print(x,y)
    l1, l2 = len(x), len(y) 
    best_span = None
    best_score = 0.5
    for i in range(l1):
        for j in range(i+1, l1+1):
            score = fuzz.ratio(x[i:j],y)
            if score > best_score:
                best_span = (i,j)
                best_score = score
    if best_span is None:
        return 0,l1
    else:
        return best_span
    
def convert_lines(tokenizer, df, max_sequence_length = 512, is_test=False):
    #pad_token_idx = tokenizer.pad_token_id
    pad_token_idx = tokenizer.pad_token_id or tokenizer.eos_token_id
    cls_token_idx = tokenizer.cls_token_id or tokenizer.eos_token_id
    sep_token_idx = tokenizer.sep_token_id or tokenizer.eos_token_id
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    position_outputs = np.zeros((len(df), 2))
    offset_outputs = np.ones((len(df),))
    extracted = []
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids_1 = tokenizer.encode(row.text,add_special_tokens=False)
#         print(len(input_ids_1))
        input_ids = [cls_token_idx, ] +input_ids_1 + [sep_token_idx, ]
        token_type_ids = [0,]*len(input_ids)
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = sep_token_idx
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        assert len(input_ids) == len(token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
        if is_test:
            continue
        selected_text = row.label.strip()
        if len(selected_text) == 0 or len(row.text) == 0:
            start_idx, end_idx = (0,0)
            position_outputs[idx,:] = [0, 0]
        else:
            if " "+selected_text in row.text:
                input_ids_2 = tokenizer.encode(" "+selected_text,add_special_tokens=False)
#                 print(idx, selected_text, row.text, contains(input_ids_2, input_ids_1))
            else:
                input_ids_2 = tokenizer.encode(selected_text,add_special_tokens=False)
            for i in range(len(input_ids_2)):
                start_idx, end_idx = contains(input_ids_2[:len(input_ids_2)-i], input_ids_1) #[:max_sequence_length - len(input_ids_0) - 2])
                if start_idx is not None:
                    if i > 1:
                        print(input_ids_2, i)
                    break
            if start_idx is None:
                start_idx = 0
                end_idx = 0
            position_outputs[idx,:] = [start_idx + 1, end_idx + 1]
            if max(position_outputs[idx,:]) >= max_sequence_length:
#                 print(position_outputs[idx,:], len(input_ids_1))
                position_outputs[idx,:] = 0,0
    if is_test:
        return outputs, type_outputs
    else:
        return outputs, type_outputs, position_outputs, offset_outputs, df

def find_best_combinations(start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, valid_start= 0, valid_end=512):
    best = (valid_start, valid_end - 1)
    best_score = -9999
    score_dict = dict()
#    print(valid_end, start_top_index, end_top_index)
    for i in range(len(start_top_log_probs)):
        for j in range(end_top_log_probs.shape[0]):
            if valid_start <= start_top_index[i] < valid_end and valid_start <= end_top_index[j,i] < valid_end and start_top_index[i] <= end_top_index[j,i]:
                score = start_top_log_probs[i] * end_top_log_probs[j,i]
                score_dict[ (start_top_index[i],end_top_index[j,i])] = score
                if score > best_score:
                    best = (start_top_index[i],end_top_index[j,i])
                    best_score = score
    if -1 in best:
        print(score_dict, valid_start, valid_end)
    return best, score_dict

def find_all_pred_in_text(normed_text, all_unique_preds):
    preds = []
    preds_indexs = []
    for pred in all_unique_preds:
        if pred in normed_text:
            preds.append(pred)
    unique_preds = []  # unique in terms of index.
    for pred in preds:
        matchs = re.finditer(pred, normed_text)
        for match in matchs:
            start_index = match.start()
            end_index = match.end()
            preds_indexs.append([start_index, end_index])
            unique_preds.append(pred)
    group_idxs = []
    for i in range(len(preds_indexs)):
        for j in range(len(preds_indexs)):
            if i != j:
                start_i, end_i = preds_indexs[i]
                start_j, end_j = preds_indexs[j]
                if start_i <= end_j and end_i <= end_j and start_i >= start_j:
                    group_idxs.append([i, j])
    unique_preds = np.array(unique_preds)
    for group_idx in group_idxs:
        unique_preds[group_idx[0]] = unique_preds[group_idx[1]]
    return np.unique(unique_preds)

wl_words = [" Dataset "," Datasets ", " Database "," Databases ", " Data ", " Survey "," Study "," Studies "," Surveys "]
bl_words = [" are ", " is ", " was ", " were "]
def is_valid_pred(x):
    if "." in x or "!" in x or "?" in x or (not x.split()[0].isalpha()):
        return False
    x = f" {x} "
    for w in bl_words:
        if w in x:
            return False
    return True

class Colour:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
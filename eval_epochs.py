#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
import os
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from utils import *
from reports_tools import * # accuracy evaluation methods
from visualization import * # for live plotting

from convert_example_to_features import *

import logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


visdom_env_name = 'evaluation_75%'
train_file = 'train_75%.tsv'
DATA_DIR = "data/combined_overlapped"



BERT_MODEL = 'bert-base-chinese'

OUTPUT_DIR = 'outputs/'
REPORT_DIR = 'reports/'
CACHE_DIR = 'cache/'
max_steps = -1
MAX_SEQ_LENGTH = 512
per_gpu_train_batch_size = 6 # change this to 6 if max_seq_length is 512
per_gpu_eval_size = 8
LEARNING_RATE = 2e-5
num_train_epochs = 30
RANDOM_SEED = 42
gradient_accumulation_steps = 1
warmup_proportion = 0.1
OUTPUT_MODE = 'classification'
# classification
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

# file name has been modified
report_file_name = f'no_weight_decay_lr={LEARNING_RATE}, epoch={num_train_epochs}, train_file={train_file}, max_seq_length={MAX_SEQ_LENGTH}, batch_size={per_gpu_train_batch_size}'
print(report_file_name)


# In[ ]:
# ensure the following statements is evaluated before any other process
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


set_seed(42) # For reproductivity

# # Fine tunning bert

# In[ ]:


processor = BinaryClassificationProcessor()
train_examples = processor.get_train_examples(DATA_DIR, train_file)
label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)






## Training func
# In[ ]
print(f'{torch.cuda.device_count()} cuda device(s) found')
n_gpus = torch.cuda.device_count()
# n_gpus = 1

# In[ ]:


loss_log = [] 


# visualization
vis = Visualizations(visdom_env_name)
loss_step_values = []
loss_epoch_values = []

#%%
print('Epoch num:', len(os.listdir('./cache')) - 2)
epoch_nums = len(os.listdir('./cache')) - 2


     

# ## visualization


# ## evaluation methods

# In[ ]:


def evaluate_model_in_epoch(epoch, eval_task_name):
    """
    eval_task_name: without file type
    """
    cache_path = f'{CACHE_DIR}epoch_{epoch}/'
    
    tokenizer = BertTokenizer.from_pretrained(cache_path)
    model = BertForSequenceClassification.from_pretrained(cache_path, 
                                                      cache_dir=cache_path, 
                                                      num_labels=len(label_list))
    model.to(device)
    
    return evaluate(eval_task_name, model, tokenizer)

    
# In[ ]
def evaluate(eval_task_name, model, tokenizer):
    processor = BinaryClassificationProcessor()
    eval_examples = processor.get_dev_examples(DATA_DIR, f'{eval_task_name}.tsv')
    eval_examples_len = len(eval_examples)
    
    eval_features = convert_examples_to_features(eval_examples, 
                                                 label_list, 
                                                 MAX_SEQ_LENGTH, 
                                                 tokenizer, 
                                                 OUTPUT_MODE)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    if OUTPUT_MODE == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)


    # FIXME: is there a need to use multi gpu here
    # if n_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    # let's just use one for now
    eval_batch_size = per_gpu_eval_size

    eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)


    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Total batch size = %d", eval_batch_size)
    
    
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # change this if using xlnet
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    logits = preds ## delete this 

    eval_loss = eval_loss / nb_eval_steps
    if OUTPUT_MODE == "classification":
        preds = np.argmax(preds, axis=1)
    elif OUTPUT_MODE == "regression":
        preds = np.squeeze(preds)

    # result = precision_recall_results(report_file_name, out_label_ids, preds) # report file name here
    # result["1-5 question acc"] = question_accuracy_old(before_argmax_preds, out_label_ids)
    
    
    result = get_report(eval_task_name, logits)
    result['eval loss '] = round(eval_loss, 2)
    return result



def get_report(output_file_name, logits):
    output_18 = write_logits_and_save_file(output_file_name, logits)
    # add assert to ensure same shape
    nested_logits_18 = nest_output_logits(output_18)

    result = evaluate_one_entry_score(output_18)
    result['question_acc'] = evaluate_question_score(output_18, nested_logits_18)
    print(result)
    return result 




# get_report('output_test_18', before_argmax_preds)
#%%





# In[ ]:


for epoch in range(epoch_nums):

    epoch += 1
    set_seed(42)
    # evaluation using saved model 
    ## eval 2017
    results = evaluate_model_in_epoch(epoch, 'test_17_75%')
    vis.plot_precision(results['precision'], epoch)
    vis.plot_recall(results['recall'], epoch)
    vis.plot_accuracy_old(results['old_question_acc'], epoch)
    vis.plot_accuracy(results['question_acc'], epoch)
    
    ## eval 2018
    results_2018 = evaluate_model_in_epoch(epoch, 'test_18_75%')
    vis.plot_precision2(results_2018['precision'], epoch)
    vis.plot_recall2(results_2018['recall'], epoch)
    vis.plot_accuracy_old2(results_2018['old_question_acc'], epoch)
    vis.plot_accuracy2(results['question_acc'], epoch)
        
	## eval train_3000.tsv
    results_train_3000 = evaluate_model_in_epoch(epoch, 'train_3000_75%')
    vis.plot_precision3(results_train_3000['precision'], epoch)
    vis.plot_recall3(results_train_3000['recall'], epoch)
    vis.plot_accuracy_old3(results_train_3000['old_question_acc'], epoch)
    vis.plot_accuracy3(results_train_3000['question_acc'], epoch)
        





#%%

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

from utils import *
from reports_tools import * # accuracy evaluation methods
from visualization import * # for live plotting

from convert_example_to_features import *

import logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


visdom_env_name = 'combined_qs_256'
train_file = 'train_combined_qs.tsv'




DATA_DIR = "data/"

BERT_MODEL = 'bert-base-chinese'

OUTPUT_DIR = 'outputs/'
REPORT_DIR = 'reports/'
CACHE_DIR = 'cache/'
max_steps = -1
MAX_SEQ_LENGTH = 256
per_gpu_train_batch_size = 16 # change this to 6 if max_seq_length is 512
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


# # Fine tunning bert

# In[ ]:


processor = BinaryClassificationProcessor()
train_examples = processor.get_train_examples(DATA_DIR, train_file)


label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)





# In[ ]:


import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


# In[ ]:


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
print(len(os.listdir('./cache')) - 3)
epoch_nums = len(os.listdir('./cache')) - 3


     

# ## visualization

# In[ ]:


def save_model_in_epoch(epoch):
    epoch = str(epoch)
    dir_path = f'{CACHE_DIR}epoch_{epoch}/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    save_model(dir_path)
    print(dir_path)



def save_model(dir_path):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(dir_path, WEIGHTS_NAME)
    output_config_file = os.path.join(dir_path, CONFIG_NAME)
    print(output_config_file)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(dir_path)
    
    print(f'Model saved successfully at: {dir_path}')


# ## evaluation methods

# In[ ]:


def evaluate_model_in_epoch(epoch, eval_task_name):
    cache_path = f'{CACHE_DIR}epoch_{epoch}/'
    
    tokenizer = BertTokenizer.from_pretrained(cache_path)
    model = BertForSequenceClassification.from_pretrained(cache_path, 
                                                      cache_dir=cache_path, 
                                                      num_labels=len(label_list))
    model.to(device)
    
    return evaluate(eval_task_name, model, tokenizer)

    

def evaluate(eval_task_name, model, tokenizer):
    processor = BinaryClassificationProcessor()
    eval_examples = processor.get_dev_examples(DATA_DIR, eval_task_name)
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

    before_argmax_preds = preds ## delete this 

    eval_loss = eval_loss / nb_eval_steps
    if OUTPUT_MODE == "classification":
        preds = np.argmax(preds, axis=1)
    elif OUTPUT_MODE == "regression":
        preds = np.squeeze(preds)

    result = precision_recall_results(report_file_name, out_label_ids, preds) # report file name here
    result["1-5 question acc"] = question_accuracy(before_argmax_preds, out_label_ids)
    result['eval loss '] = round(eval_loss, 2)
    
    
    return result


# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:


for epoch in range(epoch_nums):

    epoch += 1
    set_seed(42)
    # evaluation using saved model 
    ## eval 2017s
    results = evaluate_model_in_epoch(epoch, 'dev_17_combined_qs.tsv')
    vis.plot_precision(results['precision'], epoch)
    vis.plot_recall(results['recall'], epoch)
    vis.plot_accuracy(results['1-5 question acc'], epoch)
    
    ## eval 2018
    results_2018 = evaluate_model_in_epoch(epoch, 'dev_18_combined_qs.tsv')
    vis.plot_precision2(results_2018['precision'], epoch)
    vis.plot_recall2(results_2018['recall'], epoch)
    vis.plot_accuracy2(results_2018['1-5 question acc'], epoch)
        
	## eval train_3000.tsv
    results_train_3000 = evaluate_model_in_epoch(epoch, 'train_3000.tsv')
    vis.plot_precision3(results_train_3000['precision'], epoch)
    vis.plot_recall3(results_train_3000['recall'], epoch)
    vis.plot_accuracy3(results_train_3000['1-5 question acc'], epoch)
        




# In[ ]:





# In[ ]:





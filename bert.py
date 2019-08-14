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


visdom_env_name = 'nothing1' #combined_overlapped---need_update
train_file = 'train_combined_overlapped.tsv'
data_dir = "data/combined_overlapped" # ! FIXME need to update testing data && update accuracy for evaluation result 



use_multiprocessing = True
bert_model = 'bert-base-chinese'

output_dir = 'outputs/'
report_dir = 'reports/'
cache_dir = 'cache/'
max_steps = -1
max_seq_length = 512
per_gpu_train_batch_size = 6 # change this to 6 if max_seq_length is 512
per_gpu_eval_size = 8
learning_rate = 2e-5
num_train_epochs = 15
random_seed = 42

gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_mode = 'classification'
# classification
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

# file name has been modified
report_file_name = f'no_weight_decay_lr={learning_rate}, epoch={num_train_epochs}, train_file={train_file}, max_seq_length={max_seq_length}, batch_size={per_gpu_train_batch_size}'
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
train_examples = processor.get_train_examples(data_dir, train_file)


label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)


# In[ ]:


model = BertForSequenceClassification.from_pretrained(bert_model, 
                                                      cache_dir=cache_dir, 
                                                      num_labels=num_labels)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model.to(device)



# In[ ]:


## Training func
# In[ ]
print(f'{torch.cuda.device_count()} cuda device(s) found')
n_gpus = torch.cuda.device_count()
# n_gpus = 1

# In[ ]:
# tokenizer is needed here
def convert_to_input_features_helper(input_examples, tokenizer, multiprocessing=False):
    if not multiprocessing:
        return convert_examples_to_features(input_examples,
                                             label_list,
                                             max_seq_length,
                                             tokenizer,
                                             output_mode)
    else: # multiprocessing
        label_map = {label : i for i, label in enumerate(label_list)}
        input_multiprocessing = [(example, label_map, max_seq_length, tokenizer, output_mode) for example in input_examples]
        with Pool(cpu_count()) as p:
            return list(tqdm(p.map(convert_example_to_feature, input_multiprocessing)))


# In[ ]:
loss_log = [] 

def train(train_task_name, model, tokenizer):
    set_seed(42) # for reproductibility 
    
    # prepare training dataset
    train_features = convert_to_input_features_helper(train_examples, tokenizer, use_multiprocessing)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # total batch size
    train_batch_size = per_gpu_train_batch_size*max(1, n_gpus)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    if max_steps > 0:
        t_total = max_steps
        num_trian_epochs = max_steps // len(train_dataloader) // gradient_accumulation_steps + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        
    
    
    # prepare optimizer and schedule (linear warmup and decay)
    warmup_steps = int(t_total*warmup_proportion)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                     lr=learning_rate,
                     eps=1e-8) 
    scheduler = WarmupLinearSchedule(optimizer, 
                                 warmup_steps=warmup_steps,
                                 t_total=t_total)
    
    if n_gpus > 1:
        print('***********       using multi gpu!         ************')
        model = torch.nn.DataParallel(model)

    logger.info("***** Running %s *****", 'training')
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Batch size per gpu = %d", per_gpu_train_batch_size)
    logger.info("  Total batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", t_total)

    
    # visualization
    vis = Visualizations(visdom_env_name)
    loss_step_values = []
    loss_epoch_values = []
        
    # train
    max_grad_norm = 1


    epoch = 0 # for visualization loss-epoch
    global_step = 0
    tr_loss, loging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc='Epoch')
    
    for _ in train_iterator:
        
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': None}
            label_ids = batch[3]
            outputs = model(**inputs) # unpack dict
            logits = outputs[0] # model outputs are in tuple
            
            
            if global_step < 1:
                print(logits)
                print(label_ids)
            ############# BCEWithLogitsLoss #########
#             loss_fct = torch.nn.BCEWithLogitsLoss()
#             label_ids= torch.nn.functional.one_hot(label_ids, 2).float() # one hot encoding 
#             # change dtype to float to match cuda backend type
#             loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1, num_labels))
            #########################################
            
            
            ############# CrossEntropy ##############
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            #########################################

            if n_gpus > 1:
                if global_step < 1:
                    print('before average')
                    print(loss)
                loss = loss.mean() # to average on multi-gpu parallel training
                if global_step < 1:
                    print('after average')
                    print(loss)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            print("\r%f" % loss, end='') # delete this
 
            loss_log.append(loss) # delete this
    
    
            # visualization
            loss_step_values.append(loss.item())
            loss_epoch_values.append(loss.item())
            
            if global_step % 100 == 0:
                vis.plot_loss(np.mean(loss_step_values), global_step)
                loss_step_values.clear()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1
            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
                
      
        # loss-epoch visualization
        epoch += 1
        vis.plot_epoch(np.mean(loss_epoch_values), epoch) 
        loss_epoch_values.clear()
        
        
        # save model at each epoch
        save_model_in_epoch(epoch)

        # evaluation using saved model 
        ## eval 2017
        results = evaluate_model_in_epoch(epoch, 'dev_17_combined_qs_title.tsv')
        vis.plot_precision(results['precision'], epoch)
        vis.plot_recall(results['recall'], epoch)
        vis.plot_accuracy(results['1-5 question acc'], epoch)
        
        ## eval 2018
        results_2018 = evaluate_model_in_epoch(epoch, 'dev_18_combined_qs_title.tsv')
        vis.plot_precision2(results_2018['precision'], epoch)
        vis.plot_recall2(results_2018['recall'], epoch)
        vis.plot_accuracy2(results_2018['1-5 question acc'], epoch)
        
	## eval train_3000.tsv
        results_train_3000 = evaluate_model_in_epoch(epoch, 'train_3000_combined_qs_title.tsv')
        vis.plot_precision3(results_train_3000['precision'], epoch)
        vis.plot_recall3(results_train_3000['recall'], epoch)
        vis.plot_accuracy3(results_train_3000['1-5 question acc'], epoch)
        
        
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break


# ## visualization

# In[ ]:


def save_model_in_epoch(epoch):
    epoch = str(epoch)
    dir_path = f'{cache_dir}epoch_{epoch}/'
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
    cache_path = f'{cache_dir}epoch_{epoch}/'
    # Load a trained model and vocabulary that you have fine-tuned
    tokenizer = BertTokenizer.from_pretrained(cache_path)
    model = BertForSequenceClassification.from_pretrained(cache_path, 
                                                      cache_dir=cache_path, 
                                                      num_labels=len(label_list))
    model.to(device)
    
    return evaluate(eval_task_name, model, tokenizer)

    

def evaluate(eval_task_name, model, tokenizer):
    processor = BinaryClassificationProcessor()
    eval_examples = processor.get_dev_examples(data_dir, eval_task_name)
    eval_examples_len = len(eval_examples)
    

    eval_features = convert_to_input_features_helper(eval_examples, tokenizer, use_multiprocessing)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    if output_mode == "classification":
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
                      'token_type_ids': batch[2], 
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
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    result = precision_recall_results(report_file_name, out_label_ids, preds) # report file name here
    result["1-5 question acc"] = question_accuracy(before_argmax_preds, out_label_ids)
    result['eval loss '] = round(eval_loss, 2)
    
    
    return result







# In[ ]:


train(train_file, model, tokenizer)


# In[ ]:






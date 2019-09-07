#!/usr/bin/env python
# coding: utf-8


# In[ ]:
from config import *

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
import os
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertForMultipleChoice
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from multiprocessing import Pool, cpu_count

from utils import *
# from reports_tools import * # accuracy evaluation methods
from visualization import * # for live plotting
from convert_example_to_features import *


import logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('######################## PARAMS ##########################') 
print(stats) 



# file name has been modified

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
label_list = processor.get_labels() 

if not use_fine_tuned_model:
    print('using default model')
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForSequenceClassification.from_pretrained(bert_model,
	cache_dir=cache_dir,
        num_labels=1
        )
else:
    print('!!! using home brewed model')
    tokenizer = BertTokenizer.from_pretrained(cache_dir)
    model = BertForSequenceClassification.from_pretrained(cache_dir, num_labels=1)

model.to(device)



## Training func
# In[ ]
print(f'{torch.cuda.device_count()} cuda device(s) found')
if use_multi_gpu:
    n_gpus = torch.cuda.device_count()  # !!! set to one
else:
    n_gpus = 1
print(f'use_multi_gpu: {use_multi_gpu}; {n_gpus} cuda device(s) is going to be used in training')



#%%
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
        print(f'Using {cpu_count()} processes to convert {len(input_examples)} examples!')
        with Pool(cpu_count()) as p:
            return list(p.map(convert_example_to_feature, tqdm(input_multiprocessing)))



def train(train_task_name, model, tokenizer):
    set_seed(42) # for reproductibility 
    
    # prepare training dataset
    train_features = convert_to_input_features_helper(train_examples, tokenizer, use_multiprocessing)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long) # !!!! no minus 1
    train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # total batch size
    train_batch_size = per_gpu_train_batch_size*max(1, n_gpus)
    train_sampler = SequentialSampler(train_dataset) # was random sampler
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

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
            outputs = model(**inputs) # unpack dict
            logits = outputs[0] # model outputs are in tuple
            reshaped_logits = logits.view(-1, 5) # 5: num of choices 
            _, labels = torch.max(batch[3].view(-1, 5), 1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)


            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # print("\r%f" % loss, end='') # delete this
 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step() # call optimizer before scheduler 
                model.zero_grad()
                global_step += 1

            
            
            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
                
    
        epoch += 1 
        # save model at each epoch
        # save_model_in_epoch(epoch)
        output_model_dir = os.path.join(cache_dir, 'epoch_{}'.format(epoch))
        if not os.path.exists(output_model_dir):
            os.makedirs(output_model_dir)
        model_to_save = model.module if hasattr(model, 'module') else model # take care of distributed/parallel training
        model_to_save.save_pretrained(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        torch.save(stats, os.path.join(output_dir, 'training_args.bin'))
        logger.info('Saving model at epoch %d to %s' % (epoch, output_model_dir))



        # evaluation using saved model

        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break




# In[ ]:


# def save_model_in_epoch(epoch):
#     epoch = str(epoch)
#     dir_path = f'{cache_dir}epoch_{epoch}/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)
#     save_model(dir_path)
#     print(dir_path)



# def save_model(dir_path):
#     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    
#     # If we save using the predefined names, we can load using `from_pretrained`
#     output_model_file = os.path.join(dir_path, WEIGHTS_NAME)
#     output_config_file = os.path.join(dir_path, CONFIG_NAME)
#     print(output_config_file)
#     torch.save(model_to_save.state_dict(), output_model_file)
#     model_to_save.config.to_json_file(output_config_file)
#     tokenizer.save_vocabulary(dir_path)
    
#     print(f'Model saved successfully at: {dir_path}')


# ## evaluation methods

# In[ ]:


train(train_file, model, tokenizer)




#%%

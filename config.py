# import os
# def get_data_dir_name():
# 	list_subdirs = os.listdir(os.getcwd() + '/data/')
# 	print('before cleaned list_subdirs:',  list_subdirs)
# 	list_subdirs = [i for i in list_subdirs if 'lvl' in i]
# 	assert len(list_subdirs) == 1
# 	return list_subdirs[0]

## training params 
use_fine_tuned_model = False # set to True if using fined tuned bert model
train_type = 'choice_correct' # [choice_correct, choice_all]
snippet_type = 'lvl_5_3' # snippet_type: 11 choices 
train_file_name = 'test_18' # [train, test_17, test_18]
data_dir = f"data/{snippet_type}"
per_gpu_train_batch_size = 10 # must be a multiple of 5; set it to 5 if running on local machines
num_train_epochs = 2


## evaluation params
eval_files = ['test_17', 'test_18']
eval_epochs = range(1, num_train_epochs + 1)


use_multi_gpu = False # NOTE: MODIFY pbs file if using multi gpu on hpc
do_visualization = False

assert train_file_name in ['train', 'test_17', 'test_18'], 'train file not existed'
assert train_type in ['choice_correct', 'choice_all']
train_file = f'{train_file_name}.tsv' if train_type == 'choice_all' else f'{train_file_name}_choice_correct.tsv'

visdom_env_name = snippet_type
do_evaluation = False # deprecated! whether to evaluate models during trianing to see live results
use_multiprocessing = True

bert_model = 'bert-base-chinese'
output_dir = 'outputs/'
report_dir = 'reports/'
cache_dir = 'cache/'
max_steps = -1
max_seq_length = 512
per_gpu_eval_size = 5 
assert per_gpu_train_batch_size % 5 == 0 and per_gpu_eval_size % 5 == 0
learning_rate = 2e-5
random_seed = 42

gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_mode = 'classification'
# classification
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

# file name has been modified
stats = f'train_type={train_type}, snippet_type={snippet_type}, use_multi_gpu={use_multi_gpu}, visdom_name={visdom_env_name}, use_fine_tuned_model={use_fine_tuned_model},  total_epochs={num_train_epochs}, train_file={train_file}, max_seq_length={max_seq_length}, per_gpu_train_batch_size={per_gpu_train_batch_size}, per_gpu_eval_batch_size={per_gpu_eval_size}'

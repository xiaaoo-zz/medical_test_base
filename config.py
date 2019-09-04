import os
def get_data_dir_name():
	list_subdirs = os.listdir(os.getcwd() + '/data/')
	print('before cleaned list_subdirs:',  list_subdirs)
	list_subdirs = [i for i in list_subdirs if 'lvl' in i]
	assert len(list_subdirs) == 1
	return list_subdirs[0]


use_fine_tuned_model = False # set to True if using fined tuned bert model
snippet_type = get_data_dir_name()
use_multi_gpu = False # NOTE: MODIFY pbs file if using multi gpu on hpc
do_visualization = False
per_gpu_train_batch_size = 10 # must be a multiple of 5; set it to 5 if running on local machines
train_file = 'test_18.tsv'
num_train_epochs = 3




visdom_env_name = snippet_type
data_dir = f"data/{snippet_type}" 
do_evaluation = False # whether to evaluate models during trianing to see live results
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
stats = f'snippet_type={snippet_type}, use_multi_gpu={use_multi_gpu}, visdom_name={visdom_env_name}, use_fine_tuned_model={use_fine_tuned_model},  total_epochs={num_train_epochs}, train_file={train_file}, max_seq_length={max_seq_length}, per_gpu_train_batch_size={per_gpu_train_batch_size}, per_gpu_eval_batch_size={per_gpu_eval_size}'

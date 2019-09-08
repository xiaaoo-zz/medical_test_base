# Bert

- [Customize Classifier](#Use-customized-classifier)

- [Fine-tuned Model](#Use-fine-tuned-bert-model)

## 1. Customize Classifier

### Step 1. create classifier in [classifier.py](./classifier.py)

```py
# in classifier.py

# 1. create classifier
class ExampleClassifier:
    pass

# 2. add the classifier to dictionary
CLASSIFIER_CLASSES = {
    'default': BertForSequenceClassification,
    'example': ExampleClassifier,
}
```

### Step 2. modify [connfig.py](./config.py)

```py
# 1. change classifier type to whatever you just created
classifier_type = 'default'

# 2. change this to the correspounding dataset dir name you are using (e.g. lvl_5, lvl_4_short_tf)
snippet_type = 'lvl_5_3'

# 3. change train batch size to 5 if testing local machine
per_gpu_train_batch_size = 5 # 10
```

### Baseline

choice_correct + lvl_5 + train_batch_size 5

| File/Epoch    | 1        | 2      | 3       |
| ------------- |:--------:| :-----:| :---:   |
| test 2017     | 0.2750   | 0.3200 | **0.3483**  |
| test 2018     | 0.3906   | 0.5325 | 0.5525  |

| File/Epoch    | 1        | 2      | 3       | 4    | 5    |
| ------------- |:--------:| :-----:| :---:   |:----:         |:----:  |
| test 2017     | 0.2033   | 0.3016 | 0.3166  | 0.3233    |  **0.3600**   |
| test 2018     | 0.2988   | 0.4507 | 0.4911    | 0.5809    |  0.6600   |

## 2. Fine-tuned bert model

### Step 1: put the following fine-tuned output files in [cache](./cache/)

- config.json
- pytorch_model.bin  
- vocab.txt
- special_tokens_map.json
- added_tokens.json

### Step 2: copy data to [data](./data/)

```bash
# for lvl_5_3 dataset
# change the folder name *mt_base* to whatever relevant
scp -r lvl_5_3/ [user_name]@xxx.xxx.xxx.xxx:/csai01/[user_name]/[mt_base]
```

### Step 3 modify [config](./config.py#L10-L12) file

```py
# in config.py
# set this to true
use_fine_tuned_model = True

# change this to the correspounding dataset dir name you are using (e.g. lvl_5, lvl_4_short_tf)
snippet_type = 'lvl_5_3'
num_train_epochs = 3
```

### Step 4: train and evaluate

Using hpc: `sbatch torch.pbs`

Using local machine: `bash script.sh`

## How to check evaluation results

The accuracy results will be saved into [outputs](./outputs), including two acc-epoch graphs for year_17 and year_18 tests and one detailed acc text file.

For evaluating model, we **only** care year_17 results, the higher the better.

## WARNING: About rerunning the procedure in the same folder

Before rerunning and testing:

1. Delete saved models (not the output models saved in Step 1)

    The above procedure will run for 3 epochs and save each model in [cache](./cache/) as *epoch_i* directories. So

2. (Delete) evaluation results

    Remove `eval_results.txt` in [outputs](./outputs) to keep its info only relavant to a specific procedure. New results will append to the old ones from previous evaluation, which may not be the desired behavior.

## Baselines

> Results are rounded **down** to 4 decimal places.

---

### choice_correct + lvl_4_short_tf

Fine tuned model baseline

| File/Epoch    | 1        | 2      | 3       |
| ------------- |:--------:| :-----:| :---:   |
| test 2017     | 0.3000   | 0.3216 | **0.3433**  |
| test 2018     | 0.4858   | 0.6310 | 0.6928  |

```py
param_stats = train_type=choice_correct, snippet_type=lvl_4_short_tf, use_multi_gpu=False, visdom_name=lvl_4_short_tf, use_fine_tuned_model=False,  total_epochs=3, train_file=test_18_choice_correct.tsv, max_seq_length=512, per_gpu_train_batch_size=10, per_gpu_eval_batch_size=5
```

---

### choice_correct + lvl_5

| File/Epoch    | 1        | 2      | 3       |
| ------------- |:--------:| :-----:| :---:   |
| test 2017     | 0.3216   | 0.3116 | **0.3233**  |
| test 2018     | 0.4090   | 0.5225 | 0.5709  |

```py
param_stats = train_type=choice_correct, snippet_type=lvl_5, use_multi_gpu=False, visdom_name=lvl_5, use_fine_tuned_model=False,  total_epochs=3, train_file=test_18_choice_correct.tsv, max_seq_length=512, per_gpu_train_batch_size=10, per_gpu_eval_batch_size=5
```

---

A few important params choosed

- train_type: **choice_correct**
- per_gpu_train_batch_size: 10
- total_epochs: 3

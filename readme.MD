# Bert

## Use fine-tuned bert model

### Step 1: put the following fine-tuned output files in [cache](./cache/)

- config.json
- pytorch_model.bin  
- vocab.txt

### Step 2: set `use_fine_tuned_model` to `True` in [config.py](./config.py#L10)

```py
# in config.py
use_fine_tuned_model = True
```

### Step 3: train and evaluate

Using hpc: `sbatch torch.pbs`

Using local machine: `bash script.sh`

## How to check evaluation results

The accuracy results will be saved into [outputs](./outputs), including two acc-epoch graphs for year_17 and year_18 tests and one detailed acc text file.

For evaluating model, we **only** care year_17 results, the higher the betters.

## WARNING: Rerun the procedure in the same file

Before reruning and testing:

1. Delete saved models (not the output models saved in Step 1):

    The above procedure will run for 3 epochs and save each model in [cache](./cache/) as *epoch_i* directories. So

2. (Delete) evaluation results

    Remove `eval_results.txt` in [outputs](./outputs) to keep its info only relavant to a specific procedure. New results will append to the old ones from previous evaluation, which may not be the desired behavior.

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
def get_reports_analysis(epoch, eval_file_name, preds_list, labels_list, data_dir):
    output_path_prefix = str(epoch) + '_' + eval_file_name.split('.')[0] + '_'
    preds = []
    for a in preds_list:
        for b in a:
            preds.append(b)
    
    labels = []
    for a in labels_list:
        for b in a:
            labels.append(b+1)

    train_examples = pd.read_csv(os.path.join(data_dir, eval_file_name))

    train_examples['preds'] = preds
    train_examples['is_correct'] = [int(preds[i] == labels[i]) for i in range(len(preds))]

    # q_type_fig
    q_type_fig = sns.countplot(train_examples['q_type'], data=train_examples)
    fig = q_type_fig.get_figure()
    fig.savefig(f'./outputs/{output_path_prefix}_q_type.png')
    plt.clf()


    # is correct dist 
    is_correct_fig = sns.countplot(train_examples['q_type'], hue=train_examples['is_correct'], data=train_examples)
    fig = is_correct_fig.get_figure()
    fig.savefig(f'./outputs/{output_path_prefix}_is_correct.png')
    plt.clf()

    # FIXME: this figure cannot be saved 
    print(output_path_prefix)
    quesiton_length = [len(i) for i in train_examples['q']] 
    train_examples['q_len'] = quesiton_length
    cat_fig = sns.catplot(x='q_type', y='q_len',
                        hue='is_correct',
                        kind='violin',
                        height=6,
                        data=train_examples)
    plt.show() #! ?? --------------
    plt.clf()
    questions = []
    titles = []
    snippets = []
    for i in train_examples['q'].values:
        title, question, snippet = i.split('  æ  ')
        questions.append(question)
        titles.append(title)
        snippets.append(snippet)

    train_examples['qq'] = questions
    train_examples['tt'] = titles
    train_examples['ss'] = snippets
    train_examples.to_csv(f'outputs/{output_path_prefix}_output.csv')



#%%
def acc_plot(acc_list, epoch_list, task_name):
    """
    acc_list: accuracy result list for each test, [0.1,0.2,0.15]
    epoch_list: e.g. [1,2,3,4,5]
    task_name 
    """
    if '.' in task_name:
        task_name = task_name.replace('.', '_') # avoid png file tpye error
    task_name = str(task_name)
    acc_plt = sns.lineplot(y=acc_list, x=epoch_list)
    fig = acc_plt.get_figure()
    fig.savefig(f'./outputs/0_acc_{task_name}.png')
    plt.clf()


#%%

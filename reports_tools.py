#%%
import numpy as np
import pandas as pd

# first method 
def calc_avg_logits(cc):
    cc_np = np.array(cc)
    #print('hi')
    # assert cc_np.dtype == 'float64'
    avg_logits = np.mean(cc_np, axis=0)
    #print('hihii')
    # print(avg_logits)
    return avg_logits

def first_method_avg(nested_logits):
    """
    nested_logits: list, nested logits grouped by question and choice level 
    Average logits for all snippet pieces in a question-choice pair 
    """
    question_level_logits = []
    for q, qq in enumerate(nested_logits):
        choice_level_logits = []
        for c, cc in enumerate(qq):
            # print(cc)
            snippet_avg_logits = calc_avg_logits(cc)
            choice_level_logits.append(snippet_avg_logits)
        question_level_logits.extend(choice_level_logits)
    return question_level_logits






# make nested logits 
def check_nested_logits(nested_logits, original_df):
    result_struct = []
    result_logits = []
    for q, qq in enumerate(nested_logits):
        for c, cc in enumerate(qq):
            for s, ss in enumerate(cc):
                result_struct.append([q, c, s])
                result_logits.append(ss)
    correct_struct = original_df[['q_index','c_index','s_index']].values.tolist()
    correct_logits = original_df[['l_logits', 'r_logits']].values.tolist()

    assert correct_struct.__eq__(result_struct), 'Wrong structure of nested logits'
    assert correct_logits.__eq__(result_logits), 'Wrong logits value'
    print('Structure and Logits values are the same')
    
def nest_output_logits(df):

    nested_logits = []
    cc_cache = []
    qq_cache = []
    df_length = df.shape[0]
    qcs_indexes = df[['q_index', 'c_index', 's_index']].values
    assert qcs_indexes.dtype == 'int64', 'Wrong indexes dtype, should be int'
    input_logits = df[['l_logits', 'r_logits']].values
    assert input_logits.dtype == 'float64', 'Wrong indexes dtype, should be float64' ## CHANGE THIS
    
    for i in range(df_length - 1):
        q, c, s = qcs_indexes[i]
        # l_logit, r_logit = input_logits[i]
        #  logit = [l_logit, r_logit]
        logit = input_logits[i].tolist()
        nq, nc, ns = qcs_indexes[i+1]
        if c == nc: # same choice 
            cc_cache.append(logit)
        elif c != nc and q == nq: # not same choice, still in the same question
            cc_cache.append(logit)
            qq_cache.append(cc_cache)
            cc_cache = []
        else: # not same question
            cc_cache.append(logit)
            qq_cache.append(cc_cache)
            nested_logits.append(qq_cache)
            cc_cache = []
            qq_cache = []
        if i == df_length - 2:
            # last loop
            assert i + 1 == df_length - 1
            last_logit = input_logits[i+1].tolist()
            if c == nc:
                cc_cache.append(last_logit)
            else:
                cc_cache.append(last_logit)
            qq_cache.append(cc_cache)
            nested_logits.append(qq_cache)
            cc_cache = []
    check_nested_logits(nested_logits, df)
    return nested_logits








# Calculate one entry score: precision, recall, one_entry_score
def precision_recall_score(y_true, y_pred):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for i, true in enumerate(y_true):
        if true == y_pred[i]:
            if true == 1:
                tp += 1
            else: # == 0first_method_avg
                tn += 1
        else:
            if true == 1:
                fn += 1
            else:
                fp += 1
    true_false_values = {'tp:': tp,'tn:': tn, 'fp:': fp, 'fn:': fn}
    print(true_false_values)
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        "one_entry_acc": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "values": true_false_values
    }


# ! delete this 
def precision_recall_results(task_name, y_true, y_pred):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for i, true in enumerate(y_true):
        if true == y_pred[i]:
            if true == 1:
                tp += 1
            else: # == 0
                tn += 1
        else:
            if true == 1:
                fn += 1
            else:
                fp += 1
    true_false_values = {'tp:': tp,'tn:': tn, 'fp:': fp, 'fn:': fn}
    print(true_false_values)
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        "task     ": task_name,
        "0-1 choice acc": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "values   ": true_false_values
    }


def question_accuracy(qs_pair_label, qs_pair_logits):
    """
    qs_pair_logits: the value from logits, eg [0.288778692483902, -1.444930672645569]
    qs_pair_label: the [0, 1] value for each question-choice pair
    
    return: question accuracy
    """
    def accuracy(labels, preds):
        length = len(labels)
        correct_num = 0
        for i in range(length):
            if labels[i] == preds[i]:
                correct_num += 1
        return correct_num / length
    

    # find true labels for questions
    labels = []
    question_number = int(len(qs_pair_label) / 5)
    for question in range(question_number):
        for choice in range(5):
            choice_index = 5*question + choice
            if qs_pair_label[choice_index] == 1:
                labels.append(choice_index % 5 + 1)
                break
    # int(labels)
    
    # find predicted labels for questions 
    predicted_labels = []
    for question in range(question_number):
        # print('question number: ', question)
        temp = []
        for choice in range(5):
            # starting choice index: 5*question + choice
            # ending choice index: 5*question + choice
            choice_index = 5*question + choice
            cur_choice_preds = qs_pair_logits[choice_index] # [0.81673616, -0.56396836]
            # print(preds[choice_index])
            temp.append(cur_choice_preds[0] - cur_choice_preds[1])

        result_index = np.argmin(temp)
        predicted_labels.append(result_index + 1)
    # ? MAYBE CAN WRITE SOME TESTING METHOD
    return accuracy(labels, predicted_labels)




# ! delete
def question_accuracy_old(raw_preds, out_label_ids):
    """
    raw_preds: the value from logits, namely preds before argmax
    out_label_ids: the [0, 1] value for each question-choice pair
    
    return: question accuracy
    """
    def accuracy(labels, preds):
        length = len(labels)
        correct_num = 0
        for i in range(length):
            if labels[i] == preds[i]:
                correct_num += 1
        return correct_num / length
    
    # find true labels for questions
    labels = []
    question_number = int(len(out_label_ids) / 5)
    for question in range(question_number):
        for choice in range(5):
            choice_index = 5*question + choice
            if out_label_ids[choice_index] == 1:
                labels.append(choice_index % 5 + 1)
                break
    
    
    # find predicted labels for questions 
    predicted_labels = []
    for question in range(question_number):
        # print('question number: ', question)
        temp = []
        for choice in range(5):
            # starting choice index: 5*question + choice
            # ending choice index: 5*question + choice
            choice_index = 5*question + choice
            cur_choice_preds = raw_preds[choice_index] # [0.81673616, -0.56396836]
            # print(preds[choice_index])
            temp.append(cur_choice_preds[0] - cur_choice_preds[1])

        result_index = np.argmin(temp)
        predicted_labels.append(result_index + 1)
    return accuracy(labels, predicted_labels)





def write_logits_and_save_file(output_file_name, logits):
    """
    output_file_name: in ./data/_output_data, file to write logits 

    return: dataframe, file with logits 
    """
    output = pd.read_excel(f'./data/_output_data/{output_file_name}.xlsx') # update this
    print(output.shape)

    l_logits = [x for x, _ in logits]
    r_logits = [y for _, y in logits]
    output['l_logits'] = l_logits
    output['r_logits'] = r_logits

    # save file with logits 
    output.to_excel(f'./outputs/{output_file_name}.xlsx')
    print(f'write to ./outputs/{output_file_name}.xlsx')
    return output

def evaluate_one_entry_score(output_df):
    """
    output_df: dataframe, output with logits 
    
    return: precision, recall, question_choice_pair accuracy and old accuracy 
    """
    # Evaluate one entry acc, precision, recall
    one_entry_logits = output_df[['l_logits', 'r_logits']].values.tolist()
    one_entry_preds = np.argmax(one_entry_logits, axis=1)
    one_entry_label = output_df['label'].values
    result = precision_recall_score(one_entry_label, one_entry_preds)
    # ! FIXME: update needed 
    result['old_question_acc'] = question_accuracy_old(one_entry_logits, one_entry_label)
    return result


def evaluate_question_score(output_df, nested_logits):
    # first method - question level logits
    """
    output_df: dataframe, output with logits 
    nested_logits: list

    return: question level accuracy 
    """
    question_level_logits = first_method_avg(nested_logits)

    # question level true labels
    unique_pairs = np.unique(output_df[['q_index', 'c_index', 'label']].values, axis=0)
    question_level_label = [label for _, _, label in unique_pairs]
    return question_accuracy(question_level_label, question_level_logits)



#%%

import numpy as np
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


def question_accuracy(raw_preds, out_label_ids):
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
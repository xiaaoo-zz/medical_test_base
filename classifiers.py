#%%
from pytorch_transformers import BertModel, BertPreTrainedModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            reshaped_logits = logits.view(-1, 5) 
            _, reshaped_labels = torch.max(labels.view(-1, 5), 1)
            
            loss_fct = CrossEntropyLoss()
            print(reshaped_logits,reshaped_labels,"2222")
            loss = loss_fct(reshaped_logits, reshaped_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class Baseline(BertPreTrainedModel):
    
    def __init__(self, config):
        super(Baseline, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            reshaped_logits = logits.view(-1, 5) 
            _, reshaped_labels = torch.max(labels.view(-1, 5), 1)
            
            loss_fct = CrossEntropyLoss()
            print(reshaped_logits,reshaped_labels,"2222")
            loss = loss_fct(reshaped_logits, reshaped_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class NoHiddenLayerClassification(BertPreTrainedModel):
    # NoHiddenLayerClassification

    def __init__(self, config):
        super(NoHiddenLayerClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = nn.Linear(in_features=768*5,out_features=2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # batch size * 768 
        pooled_output = outputs[1] 
        # need reshape
        print(pooled_output.shape)
        pooled_output = pooled_output.view(1,-1)
        print(pooled_output.shape)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            model_output = logits
            real_label = torch.tensor([labels[0]],device="cuda")
            loss_fct = CrossEntropyLoss()
            print(model_output,real_label,"2222")
            loss = loss_fct(model_output, real_label)
            print(model_output,real_label,"333333")
            print("loss", loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class oneHiddenLayer_768_classifier(BertPreTrainedModel):
    # oneHiddenLayer_768_classifier

    def __init__(self, config):
        super(oneHiddenLayer_768_classifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = nn.Linear(in_features=768*5,out_features=768)
        self.fc2 = nn.Linear(in_features=768,out_features=2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # batch size * 768 
        pooled_output = outputs[1] 
        # need reshape
        pooled_output = pooled_output.view(1,-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)

        pooled_output = self.dropout(logits)
        logits = self.fc2(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            model_output = logits
            real_label = torch.tensor([labels[0]],device="cuda")
            loss_fct = CrossEntropyLoss()
            print(model_output,real_label,"2222")
            loss = loss_fct(model_output, real_label)
            print(model_output,real_label,"333333")
            print("loss", loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class twoHiddenLayer_with_3840_768_classifier(BertPreTrainedModel):
    # twoHiddenLayer_with_2304_2304_classifier

    def __init__(self, config):
        super(twoHiddenLayer_with_3840_768_classifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = nn.Linear(in_features=768*5,out_features=768*5)
        self.fc2 = nn.Linear(in_features=768*5,out_features=768)
        self.fc3 = nn.Linear(in_features=768,out_features=2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # batch size * 768 
        pooled_output = outputs[1] 
        # need reshape
        pooled_output = pooled_output.view(1,-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)

        pooled_output = self.dropout(logits)
        logits = self.fc2(pooled_output)

        pooled_output = self.dropout(logits)
        logits = self.fc3(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            model_output = logits
            real_label = torch.tensor([labels[0]],device="cuda")
            loss_fct = CrossEntropyLoss()
            print(model_output,real_label,"2222")
            loss = loss_fct(model_output, real_label)
            print(model_output,real_label,"333333")
            print("loss", loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class twoHiddenLayer_with_3840_3840_classifier(BertPreTrainedModel):
    # twoHiddenLayer_with_2304_2304_classifier

    def __init__(self, config):
        super(twoHiddenLayer_with_3840_3840_classifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = nn.Linear(in_features=768*5,out_features=768*5)
        self.fc2 = nn.Linear(in_features=768*5,out_features=768*5)
        self.fc3 = nn.Linear(in_features=768*5,out_features=2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # batch size * 768 
        pooled_output = outputs[1] 
        # need reshape
        pooled_output = pooled_output.view(1,-1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)

        pooled_output = self.dropout(logits)
        logits = self.fc2(pooled_output)

        pooled_output = self.dropout(logits)
        logits = self.fc3(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            model_output = logits
            real_label = torch.tensor([labels[0]],device="cuda")
            loss_fct = CrossEntropyLoss()
            print(model_output,real_label,"2222")
            loss = loss_fct(model_output, real_label)
            print(model_output,real_label,"333333")
            print("loss", loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

CLASSIFIER_CLASSES = {
    'default': BertForSequenceClassification,
    'baseLine': Baseline,
    'NoHidden': NoHiddenLayerClassification,
    'OneHidden':oneHiddenLayer_768_classifier,
    'twoHidden_3840_768':twoHiddenLayer_with_3840_768_classifier,
    'twoHidden_3840_3840':twoHiddenLayer_with_3840_3840_classifier
}

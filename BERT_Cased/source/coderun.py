
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import logging
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from CTB_Case import *

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


train_data = data

# For printing to the log
logger = logging.getLogger(__name__)

# Loading the data used for training and testing
val_data = pd.read_csv("dev_subtask1.csv")
test_data = pd.read_csv("test_subtask1.csv")

# Transformer model name used
model_name = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)

# Adding special tokens
encoding = tokenizer.encode_plus(
    train_data,
    max_length=512,
    add_special_tokens=True,      
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',         
)

# Trying to get GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(encoding.keys())

# Each sentence is classified only into causal and non-causal
class_names = ['causal', 'non-causal']

# Getting the Input Tokens and attention mask to pass to BERT
print(len(encoding['input_ids'][0]))
encoding['input_ids'][0]

print(len(encoding['attention_mask'][0]))
encoding['attention_mask']

# Loading the BERT model
bert_model = BertModel.from_pretrained(model_name)

# Testing for the output
last_hidden_state, pooled_output = bert_model(
    input_ids=encoding['input_ids'],
    attention_mask=encoding['attention_mask']
)

# Checking the lengths and shape of the BERT hidden states and pooled output
print(last_hidden_state.shape)
print(bert_model.config.hidden_size)
print(pooled_output.shape)


## MODEL ##

class CausalClassifier(nn.Module):

    def __init__(self, hidden_dim, n_classes=2):
        super(CausalClassifier, self).__init__()
        # Layer to load the BERT Model
        self.bert = BertModel.from_pretrained(model_name)
        # Drop out layer added to avoid overfitting
        self.drop = nn.Dropout(p=0.3)
        
        # Linear layers added to pass bert pooled output to 2-way classifier network
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)
        
        # Drop out layer in between thse hidden linear layers
        self.drop_fc = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask):
        #Passing in the input ids and attention mask to bert model
        _, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        #sending pooled output to drop out layer
        output = self.drop(pooled_output)

        #output passed to linear layers and a dropout layer
        output_layer1 = self.fc1(output)
        output_layer2 = self.fc2(output_layer1)
        output = self.drop_fc(output_layer2)
        output = self.out(output)

        return output


hidden_dim = 50

model = CausalClassifier(hidden_dim)
model = model.to(device)

input_ids = train_data['input_ids'].to(device)
attention_mask = train_data['attention_mask'].to(device)

print(input_ids.shape)
print(attention_mask.shape)


F.softmax(model(input_ids, attention_mask), dim=1)


EPOCHS = 10

# ADamW optimizer from transformer for better perfomance than using normal Adam optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data) * EPOCHS

# Using builtin linear scheduler from transofmer to incline with BERT for better performance
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Loss criterion
loss_fn = nn.CrossEntropyLoss().to(device)


## TRAINING ##

# Training epoch 
def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):

    losses = []
    correct_predictions = 0

    # Loading data
    for d in data_loader:

        # here Input IDs are  simply mappings between tokens and their respective IDs and
        # The attention mask is to prevent the model from looking at padding tokens.

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        
        # output from BERT model along with added network
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.softmax(outputs, dim=1)

        ## calculating the loss using the lossfunction defined above
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        ## Backward propgation for updating the weights

        loss.backward()
        # To avoid the problem of exploding gradiets in deep networks
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Stepping optimizer and scheduler 
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    # Prining the train metrics
    logger.info("***** train metrics *****")

    return correct_predictions.double() / n_examples, np.mean(losses)


## Testing the model

def eval_model(model, data_loader, loss_fn, device, n_examples):

    losses = []
    correct_predictions = 0
    
    # As we don't backpropogate during this eval phase
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            ## here we are predicting the results

            _, preds = torch.max(outputs, dim=1)

            ## Calculating the loss of the predicted outputs and storing it

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    logger.info("***** eval metrics *****")
    return correct_predictions.double() / n_examples, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0

# Now performing the training on data and evaluating the model
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_data)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data,
        loss_fn,
        device,
        len(val_data)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    ## savingthe accuracy and loss

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    ## storing the best accuracy

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), 'best_model_state.bin')
      best_accuracy = val_acc



## plotting the accuracy of the models

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])


test_acc, _ = eval_model(
    model,
    test_data,
    loss_fn,
    device,
    len(test_data)
)

test_acc.item()



def get_predictions(model, data_loader):
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            ## taking each sentence in data and extracting the input_ids and attention_masks

            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            ## predictions from trained model
            _, preds = torch.max(outputs, dim=1)

            ## adding the softmax layer for the classification as activation function
            probs = F.softmax(outputs, dim=1)

            ## Storing the predictions
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    logger.info("***** Predict *****")

    return review_texts, predictions, prediction_probs, real_values


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data
)

# Printing the stats in terminal
logger.info("***** Causal News Corpus(Team :Thunderbolts) *****")
print(classification_report(y_test, y_pred, target_names=class_names))

# Displaying the confusion matrix for the data we have used so far in evaluation phase
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(
        hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(
        hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True')
    plt.xlabel('Predicted')


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

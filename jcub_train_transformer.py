import sys 
import torch
import argparse
import configparser
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, r2_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    r2_score_ = r2_score(labels, pred)  
    return {"r2_score": r2_score_}        


train_model_name = "/home/akarmakar/jcub/jcub_bert_cmpx/checkpoint-3500" #"bert-base-uncased"                      # OR local checkpoint_path
tokenizer_name   = "bert-base-uncased"                      # OR local checkpoint_path

train_data = pd.read_csv(sys.path[0] + "/datasets/base/jcub_train_full_scale_csv.csv", header=0) 
test_data = pd.read_csv(sys.path[0] + "/datasets/base/jcub_test_full_scale_csv.csv", header=0)

X = list(train_data["method_text"])
y = list(train_data["cmpx_label"])

X_test = list(test_data["method_text"])
y_test = list(test_data["cmpx_label"])

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(train_model_name, num_labels=len(set(y)))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)
test_dataset = Dataset(X_test_tokenized)

# train
# args = TrainingArguments(
#     output_dir="jcub_bert_cmpx",
#     evaluation_strategy="steps",
#     eval_steps=500,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     seed=42,
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )

# trainer.train()

trainer   = Trainer(model)     
# predict
raw_pred, _, _ = trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
print(y_pred)

# dict_ = {"orig": y_test, "pred": y_pred}
# df = pd.Dataframe(dict_)
# df.to_csv(sys.path[0] + '/eval_pred.csv', index=False)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))




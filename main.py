"""
Title: Semantic Similarity with BERT
Author: [Mohamad Merchant](https://twitter.com/mohmadmerchant1)
Date created: 2020/08/15
Last modified: 2020/08/29
Description: Natural Language Inference by fine-tuning BERT model on SNLI Corpus.
"""
"""
## Introduction

Semantic Similarity is the task of determining how similar
two sentences are, in terms of what they mean.
This example demonstrates the use of SNLI (Stanford Natural Language Inference) Corpus
to predict sentence semantic similarity with Transformers.
We will fine-tune a BERT model that takes two sentences as inputs
and that outputs a similarity score for these two sentences.

### References

* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
"""

"""
## Setup

Note: install HuggingFace `transformers` via `pip install transformers` (version >= 2.11.0).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import config
import os
from dataset import DataGenerator
from model import BertBasedModel
# import transformers

"""
## Configuration
"""

max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

# Labels in our dataset.
# labels = ["contradiction", "entailment", "neutral"]

"""
## Load the Data
"""

"""shell
curl -LO https://raw.githubusercontent.com/MohamadMerchant/SNLI/master/data.tar.gz
tar -xvzf data.tar.gz
"""

# ---------------------my code starts------------------------
# partitions = ["train", "test", "dev"]
partitions = ["TRAIN"]
data,y = {}, {}

for p in partitions:
    # path[p] = os.path.join(config.DATSET_PATH, "sts_{}_cleaned.csv".format(p))
    try:
        data[p] = pd.read_csv( os.path.join(config.DATSET_PATH,config.FILE_PATH[p]), header=None)
        if p != "TEST":
            y[p] = data[p].iloc[:, 0]

    except Exception as e:
        print("ERROR")
        # print(p) 
        print(e)
    print(data[p].isna().sum())

print(y)

train_data = DataGenerator(
    # train_df[["sentence1", "sentence2"]].values.astype("str"),
    data["TRAIN"].iloc[:,[1,2]].values.astype("str"),
    y["TRAIN"],
    batch_size=config.batch_size,
    shuffle=True,
)
# valid_data = BertSemanticDataGenerator(
#     valid_df[["sentence1", "sentence2"]].values.astype("str"),
#     y_val,
#     batch_size=batch_size,
#     shuffle=False,
# )

model = BertBasedModel(config.max_length)
exit()

# -----------------------my code ends--------------------------

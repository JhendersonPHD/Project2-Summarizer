
# Abstractive Text Summarization with T5

This project demonstrates the power of transformers in the realm of Natural Language Processing (NLP). By utilizing the T5 (Text-to-Text Transfer Transformer) model from HuggingFace's Transformers library, I've aimed to achieve state-of-the-art performance in the task of abstractive text summarization on the `samsum` dataset.

## Table of Contents
- [Key Concepts](#key-concepts)
- [Code Explanation](#code-explanation)
    - [Importing Necessary Libraries](#importing-necessary-libraries)
    - [Dataset Exploration](#dataset-exploration)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Configuration and Training](#model-configuration-and-training)
    - [Model Evaluation and Deployment](#model-evaluation-and-deployment)
- [Installation and Setup](#installation-and-setup)
- [Challenges Faced](#challenges-faced)
- [Future Work and Improvements](#future-work-and-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Conclusion](#conclusion)

## Key Concepts

The world of Natural Language Processing (NLP) has been revolutionized by transformer architectures. The T5 (Text-to-Text Transfer Transformer) is one of the shining stars in this domain. T5 interprets every NLP problem as a text-to-text problem, where the input and output are both sequences of text. This perspective allows T5 to generalize across a wide range of tasks.

## Code Explanation

### Importing Necessary Libraries

Before diving into the main implementation, I import all the required libraries, which ensures a seamless experience when working with data and models.

```python
!pip install -q evaluate py7zr rouge_score absl-py
```


import nltk: This line imports the Natural Language Toolkit (NLTK) library. NLTK is a powerful library for working with human language data, including text processing, tokenization, stemming, tagging, parsing, semantic reasoning, and more.

from nltk.tokenize import sent_tokenize: This line imports the sent_tokenize function from the nltk.tokenize module. This function is used for sentence tokenization, which means it splits a piece of text into individual sentences.

nltk.download("punkt"): This line downloads the required NLTK data for sentence tokenization. The tokenization process involves breaking down a text into smaller units, such as words or sentences, to facilitate further analysis or processing.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

import torch
import torch.nn as nn

import datasets
import transformers
```

AutoModelForSeq2SeqLM: This is a class within the transformers library that provides an interface for pre-trained sequence-to-sequence language models. It can be used to generate translations, summarizations, or any other task that involves converting one sequence into another.

Seq2SeqTrainingArguments: This is a class within the transformers library that represents the training arguments/configuration for a sequence-to-sequence model. It allows you to specify various training parameters like the number of epochs, learning rate, batch size, etc.

Seq2SeqTrainer: This is a class within the transformers library that provides a high-level API for training sequence-to-sequence models. It takes care of handling the training loop and evaluating the model's performance during training.

AutoTokenizer: This is a class within the transformers library that provides an interface to automatically select the appropriate tokenizer for a given pre-trained language model. Tokenizers are used to convert textual data into a format suitable for processing by NLP models.
```python
from transformers import (
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        AutoTokenizer
)
import evaluate

import warnings
warnings.filterwarnings('ignore')
from pprint import pprint

import os
os.environ["WANDB_DISABLED"] = "true"

from IPython.display import clear_output

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Evaluate version: {evaluate.__version__}")
```


### Dataset Exploration

Understanding the dataset is crucial for any ML project. I delve deep into the data, examining its structure and content.

```python
# The samsum dataset shape
samsum

rand_idx = np.random.randint(0, len(samsum['train']))

print(f"Dialogue:
{samsum['train'][rand_idx]['dialogue']}")
print('
', '-'*50, '
')
print(f"Summary:
{samsum['train'][rand_idx]['summary']}")
```

### Data Preprocessing

Quality data is the foundation of a successful model. Here, I showcase my data wrangling skills by cleaning, tokenizing, and formatting the data to feed into the T5 model.

```python
model_ckpt = 't5-small'
tokenizer = AutoTokenizer.from_pretrained('t5-small')

from datasets import concatenate_datasets
tokenized_inputs = concatenate_datasets([samsum["train"], samsum["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

tokenized_targets = concatenate_datasets([samsum["train"], samsum["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")
```

### Model Configuration and Training

Training the model is the heart of the project. In this step, I demonstrate my understanding of model training dynamics.

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
label_pad_token_id = -100

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

import logging
logging.getLogger("transformers").setLevel(logging.WARNING)

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_samsum",
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    predict_with_generate=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=50,
    logging_first_step=False,
    fp16=False
)

training_data = tokenized_dataset['train']
eval_data = tokenized_dataset['validation']

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

results = trainer.train()
pprint(results)
```


### Model Evaluation and Deployment

After training the model, it's essential to evaluate its performance. Here, I evaluate the model and showcase potential deployment techniques.

```python
res = trainer.evaluate()
cols  = ["eval_loss", "eval_rouge1", "eval_rouge2", "eval_rougeL", "eval_rougeLsum"]
filtered_scores = dict((x , res[x]) for x in cols)
pd.DataFrame([filtered_scores], index=[model_ckpt])

from transformers import pipeline

summarizer_pipeline = pipeline("summarization",
                              model=model,
                              tokenizer=tokenizer,
                              device=0)

rand_idx = np.random.randint(low=0, high=len(samsum["test"]))
sample = samsum["test"][rand_idx]

dialog = sample["dialogue"]
true_summary = sample["summary"]

model_summary = summarizer_pipeline(dialog)
clear_output()

print(f"Dialogue: {dialog}")
print("-"*25)
print(f"True Summary: {true_summary}")
print("-"*25)
print(f"Model Summary: {model_summary[0]['summary_text']}")
print("-"*25)

def create_summary(input_text, model_pipeline=summarizer_pipeline):
    summary = model_pipeline(input_text)
    return summary

text = '''
Andy: I need you to come in to work on the weekend.
David: Why boss? I have plans to go on a concert I might not be able to come on the weekend.
Andy: It's important we need to get our paperwork all sorted out for this year. Corporate needs it.
David: But I already made plans and this is news to me on very short notice.
Andy: Be there or you're fired
'''

print(f"Original Text:
{text}")
print('
', '-'*50, '
')
summary = create_summary(text)
print(f"Generated Summary: 
{summary}")

'''
Andy needs David to come in to work on the weekend. David has plans to go on a concert. 
David already made plans and this is news to him on very short notice.
'''
```



## Installation and Setup

To replicate this project on your local machine, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your_username/Text_Summarization_T5.git
    cd Text_Summarization_T5
    ```
2. **Install the Dependencies**:
    Make sure you have Python installed. Then, install the necessary libraries and packages using pip:
    ```bash
    pip install transformers datasets torch nltk seaborn matplotlib pandas evaluate py7zr rouge_score absl-py
    ```
3. **Run the Jupyter Notebook**:
    If you have Jupyter installed, simply type:
    ```bash
    jupyter notebook
    ```
    Navigate to the project directory and open the notebook to run the cells.

## Challenges Faced

Throughout this project, I encountered several challenges:

1. **Data Preprocessing**: Ensuring the data is correctly tokenized and formatted for the T5 model required careful attention.
2. **Model Training**: Training a model as powerful as T5 demands computational resources. Ensuring efficient usage of memory was crucial.
3. **Model Evaluation**: Using the right metrics to evaluate the model's performance and interpreting those metrics was essential for understanding the model's strengths and weaknesses.

## Future Work and Improvements

While the current implementation showcases the potential of the T5 model in text summarization, there are areas for improvement and future exploration:

1. **Hyperparameter Tuning**: Exploring different hyperparameters can potentially improve model performance.
2. **Using Larger Models**: While 't5-small' was used here, larger versions of T5 can be explored for potentially better results.
3. **Incorporate Feedback**: As users interact with the summarizer, their feedback can be used to further refine and train the model.

## Contributing

If you're interested in improving this project or adding features, please:

1. Fork the repository.
2. Create a new branch with a descriptive name.
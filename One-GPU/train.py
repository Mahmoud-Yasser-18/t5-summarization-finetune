#!/usr/bin/env python

import datasets
import random
import transformers
import os
import sys
from datasets import load_dataset, load_metric
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
nltk.download('punkt')

import numpy as np


model_checkpoint = "t5-3b"

raw_datasets = load_dataset("xsum",cache_dir="./dataset")
metric = load_metric("rouge")

tokenizer = T5Tokenizer.from_pretrained(model_checkpoint,cache_dir="./t5-3b-tokenizer/")

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint,cache_dir="./t5-3b-Model/")
    
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)



batch_size = 2
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(

    # 
    "test-summarization",
    overwrite_output_dir=True,
    
    evaluation_strategy ='steps',
    eval_steps = 100, # Evaluation and Save happens every 10 steps
    save_total_limit = 1, # Only last 1 models are saved. Older ones are deleted.
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=100,  
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,

    # optimizer:
    learning_rate=3e-5,
    weight_decay=3e-7,
    adam_epsilon=1e-8,
    adam_beta1=0.9,

    # Schedular
    warmup_steps=500,


    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
    remove_unused_columns = False,
    deepspeed="./deepspeed-zero3-one-gpu.json"
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()

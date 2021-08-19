#!/usr/bin/env python
import os

for i in range(1):
    try:
        os.system("bash requirments.sh")
    except:
        print ("it's okay")


while True:
    try:
        import wandb
        import datasets
        import random
        import transformers

        import sys
        from datasets import load_dataset, load_metric
        import pandas as pd
        from transformers import AutoTokenizer
        from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        import nltk
        import numpy as np
        break
    except:
        continue
# import argparse


# parser = argparse.ArgumentParser()

# hyperparameters sent by the client are passed as command-line arguments to the script.
# parser.add_argument("--output_dir", type=int,default=os.environ["SM_MODEL_DIR"])
# parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
# parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
# script_args, _ = parser.parse_known_args()



os.environ['WANDB_API_KEY'] = '87f1f4023fbed40e9c683e5f1d90c5b9bd68ddcf'
os.environ['WANDB_PROJECT'] = 'huggingface-aws-t5'
os.environ['TASK_NAME'] = 't5-3b-trail-cpu'
wandb.login()


for i in range(1):
    try:
        nltk.download('punkt')
    except:
        print ("it's okay")



model_checkpoint = "t5-3b"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

raw_datasets = load_dataset("xsum")
metric = load_metric("rouge")
raw_datasets['train']=raw_datasets['train'].select(range(100))
raw_datasets['validation']=raw_datasets['validation'].select(range(100))
raw_datasets['test']=raw_datasets['test'].select(range(100))

prefix = "summarize: "

max_input_length = 512
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
    output_dir="/opt/ml/model/test_summarization",
    overwrite_output_dir=True,


    evaluation_strategy ='steps',
    eval_steps = 10, # Evaluation and Save happens every 10 steps
    save_total_limit = 1, # Only last 2 models are saved. Older ones are deleted.
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=10,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    max_grad_norm=0,
    # optimizer:
    # learning_rate=3e-5,
    # weight_decay=3e-7,
    # adam_epsilon=1e-8,
    # adam_beta1=0.9,
    # adam_beta2=0.999,
    # Schedular
    # warmup_steps=500,
    num_train_epochs=20,
    report_to="wandb",
    predict_with_generate=True,
    fp16=True
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding='longest')


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




print (tokenized_datasets["train"])
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

try:
    trainer.train()
    print("training success")
    trainer.save_model('/opt/ml/model')#script_args.output_dir)

except:
    print("training failed")
#     wandb.log("error happend while training")
finally:
    wandb.finish()
    print("Job termiated ")

    # Saves the model to s3; default is /opt/ml/model which SageMaker sends to S3

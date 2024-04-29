import csv
import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn
from datasets import Dataset
import evaluate
import argparse

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    set_seed
)


def add_params():
    parser = argparse.ArgumentParser()

    # Read arguments
    parser.add_argument("--dataset_name", type=str, default="CHEM121", help="the name of the dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="the directory to the data")
    parser.add_argument("--train_size", type=float, required=True, help="the size of the training dataset")
    parser.add_argument("--valid_size", type=float, required=True, help="the size of the valid dataset")
    parser.add_argument("--test_size", type=float, required=True, help="the size of the testing dataset")

    parser.add_argument("--seed", type=int, help="the seed for the random module", required=True)
    parser.add_argument("--use_rubric", action='store_true', help="whether to train with rubrics")
    parser.add_argument("--metric_for_best_model", type=str, default="f1", help="the metric for the best model")
    
    parser.add_argument("--model", type=str, required=True, help="the pretrained model")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="the number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="the batch size of training")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="the batch size of evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="the learning rate")

    parser.add_argument("--cache_dir", type=str, default="cache", help="the directory to the cache")
    parser.add_argument("--output_dir", type=str, default="results", help="the directory to the outputs")
    parser.add_argument("--eval_only", action='store_true', help="whether to do eval only")
    parser.add_argument("--unseen_split", type=int, default=-1, help="which question to use for unseen split")
    
    params = parser.parse_args()

    # Set the output dir path
    sizes = str(params.train_size) + '_' + str(params.valid_size) + '_' + str(params.test_size)
    postfix = "_rubric" if params.use_rubric else "_no-rubric"
    postfix += f"_{params.seed}_{sizes}"
    if params.unseen_split != -1:
        postfix += f"_unseen-{params.unseen_split}"
    
    params.output_dir = params.output_dir + "/" + params.model.split("/")[-1] + postfix
    os.makedirs(params.output_dir, exist_ok=True)
    return params
    
def map_answer_to_total(csv_file):
    ans2total = dict()
    with open(csv_file, 'r') as file:
        # Map the answer to the total number of scores
        reader = csv.reader(file)

        for row in reader:
            ans = row[1]
            total = row[-1]
            ans2total[ans] = int(total)

    return ans2total

def evaluate_multiclass(trainer, dataset, csv_fle):
    ans2score = dict()

    # Get the predictions and labels
    results = trainer.predict(dataset, metric_key_prefix="predict")
    predictions = results.predictions[0] if isinstance(results.predictions, tuple) else results.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = results.label_ids
    
    answers = []
    for data in dataset:
        answers.append(data["premise"])
    
    # Sum up the scores
    for i in range(len(answers)):
        if answers[i] not in ans2score:
            ans2score[answers[i]] = {"pred":0, "label":0}
        if predictions[i] == 1:
            ans2score[answers[i]]["pred"] += 1
        if labels[i] == 1:
            ans2score[answers[i]]["label"] += 1
    
    # Normalize the scores
    ans2total = map_answer_to_total(csv_fle)
    
    for ans in ans2score:
        total = ans2total[ans]
        ans2score[ans]["pred"] = ans2score[ans]["pred"] / total * 8
        ans2score[ans]["label"] = ans2score[ans]["label"] / total * 8
    
    # Concat the results
    predicted_scores = []
    actual_scores = []
    for ans in ans2score:
        predicted_scores.append(ans2score[ans]["pred"])
        actual_scores.append(ans2score[ans]["label"])
    
    # Compute metrics
    acc = acc_metric.compute(predictions=predicted_scores, references=actual_scores)["accuracy"]
    precision = precision_metric.compute(predictions=predicted_scores, references=actual_scores, average="macro")["precision"]
    recall = recall_metric.compute(predictions=predicted_scores, references=actual_scores, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=predicted_scores, references=actual_scores, average="macro")["f1"]
    mse = mse_metric.compute(predictions=predicted_scores, references=actual_scores)["mse"]
    mae = mae_metric.compute(predictions=predicted_scores, references=actual_scores)["mae"]

    # Log the metrics
    print("Logging the multi-class evaluation metrics...")
    print("***** multi-class eval metrics *****")
    print(f"  accuracy  = {acc:.4f}")
    print(f"  precision = {precision:.4f}")
    print(f"  recall    = {recall:.4f}")
    print(f"  f1        = {f1:.4f}")
    print(f"  mse       = {mse:.4f}")
    print(f"  mae       = {mae:.4f}")

    # Store the metrics
    with open(params.output_dir+"/eval_results_multi_class.json", 'w') as file:
        json.dump({"acc": acc, "precision": precision, "recall": recall, "f1": f1, "mse": mse, "mae": mae}, file, indent=4)

if __name__ == "__main__":
    # Get parameters
    params = add_params()
    
    # Set seeds
    set_seed(params.seed)
    
    # Get data files
    sizes = str(params.train_size) + '_' + str(params.valid_size) + '_' + str(params.test_size)

    train_csv = params.dataset_name + ('_rubric_' if params.use_rubric else '_no-rubric_') + sizes + ("/train.csv" if params.unseen_split == -1 else f"_unseen-{params.unseen_split}/train.csv") 
    valid_csv = params.dataset_name + ('_rubric_' if params.use_rubric else '_no-rubric_') + sizes + ("/valid.csv" if params.unseen_split == -1 else f"_unseen-{params.unseen_split}/valid.csv")
    test_csv = params.dataset_name + ('_rubric_' if params.use_rubric else '_no-rubric_') + sizes + ("/test.csv" if params.unseen_split == -1  else f"_unseen-{params.unseen_split}/test.csv")

    # Get the shuffled dataframes
    train_df = pd.read_csv(params.data_dir + '/' + train_csv).sample(frac=1, random_state=42)
    valid_df = pd.read_csv(params.data_dir + '/' + valid_csv).sample(frac=1, random_state=42)
    test_df = pd.read_csv(params.data_dir + '/' + test_csv).sample(frac=1, random_state=42)

    # Get the datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Define the function to preprocess the dataset
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if params.use_rubric:
        def tokenize_fn(examples):
            return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=min(256,tokenizer.model_max_length))
    else:
        def tokenize_fn(examples):
            return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=min(256,tokenizer.model_max_length))

    # Get the tokenized datasets
    tokenizer_train_dataset = train_dataset.map(tokenize_fn, batched=True)
    tokenizer_valid_dataset = valid_dataset.map(tokenize_fn, batched=True)
    tokenizer_test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # Set the number of classes
    num_classes = 2 if params.use_rubric else 9

    # Get the model
    model = AutoModelForSequenceClassification.from_pretrained(params.model, cache_dir=params.cache_dir)
    model.num_labels = num_classes

    # Modify the model weights
    if not params.eval_only:
        if "roberta" in params.model:
            num_input_features = model.classifier.out_proj.in_features
            old_classifier = model.classifier
            new_classifier = nn.Linear(num_input_features, num_classes, bias=True)
                                       
            if "mnli" in params.model and params.use_rubric:
                new_classifier.weight.data = old_classifier.out_proj.weight.data[0:3:2] 
                new_classifier.bias.data = old_classifier.out_proj.bias.data[0:3:2]
                
            old_classifier.out_proj = new_classifier
            
        elif "bart" in params.model:
            model = AutoModelForSequenceClassification.from_pretrained(params.model, num_labels=num_classes, ignore_mismatched_sizes=True)
    
            old_classifier = AutoModelForSequenceClassification.from_pretrained(params.model).classification_head.out_proj
    
            num_input_features = model.classification_head.dense.in_features
            new_classifier = nn.Linear(num_input_features, num_classes, bias=True)
    
            if "mnli" in params.model and params.use_rubric:
                new_classifier.weight.data = old_classifier.weight.data[0:3:2] 
                new_classifier.bias.data = old_classifier.bias.data[0:3:2]
    
            model.classification_head.out_proj = new_classifier
            
        elif "bert" in params.model:
            num_input_features = model.classifier.in_features
            new_classifier = nn.Linear(num_input_features, num_classes, bias=True)
            
            if "mnli" in params.model and params.use_rubric:
                new_classifier.weight.data = model.classifier.weight.data[0:3:2] 
                new_classifier.bias.data = model.classifier.bias.data[0:3:2]
                
            model.classifier = new_classifier

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=params.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.train_batch_size,
        per_device_eval_batch_size=params.eval_batch_size,
        learning_rate=params.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model='eval_' + params.metric_for_best_model,
        seed=params.seed
    )
    
    # Load the metrics
    acc_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    mse_metric = evaluate.load("mse")
    mae_metric = evaluate.load("mae")

    if params.use_rubric:
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            
            acc_result = acc_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
            precision_result = precision_metric.compute(predictions=preds, references=p.label_ids)["precision"]
            recall_result = recall_metric.compute(predictions=preds, references=p.label_ids)["recall"]
            f1_result = f1_metric.compute(predictions=preds, references=p.label_ids)["f1"]
            
            return {"accuracy": acc_result, "precision": precision_result, "recall": recall_result, "f1":f1_result}

    else:
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            
            acc_result = acc_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
            precision_result = precision_metric.compute(predictions=preds, references=p.label_ids, average="macro")["precision"]
            recall_result = recall_metric.compute(predictions=preds, references=p.label_ids, average="macro")["recall"]
            f1_result = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
            
            mse_result = mse_metric.compute(predictions=preds, references=p.label_ids)["mse"]
            mae_result = mae_metric.compute(predictions=preds, references=p.label_ids)["mae"]
            
            return {"accuracy": acc_result, "precision": precision_result, "recall": recall_result, "f1": f1_result, "mse": mse_result, "mae": mae_result}

    # Instantiate a Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenizer_train_dataset,
        eval_dataset=tokenizer_valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )    

    if not params.eval_only:
        # Train the model
        train_result = trainer.train()

        metrics = train_result.metrics
    
        # Save the model
        trainer.save_model(params.output_dir + "/best_model")

        print("Logging the training metrics...")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate the model on the binary classfication task
    metrics = trainer.evaluate(eval_dataset=tokenizer_test_dataset)
    print("Logging the binary evaluation metrics...")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Evaluate the model on the multi-class classification task
    if params.use_rubric:
        test_dataset_split = params.data_dir + '/' + params.dataset_name + "_" + sizes + ("/test.csv" if params.unseen_split == -1 else f"_unseen-{params.unseen_split}/test.csv")
        evaluate_multiclass(trainer, tokenizer_test_dataset, test_dataset_split)
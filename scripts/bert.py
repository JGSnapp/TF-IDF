from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

import torch
import os
print("CUDA available:", torch.cuda.is_available())

results = []

names = ["cointegrated/rubert-tiny2", "Aniemore/rubert-large-emotion-russian-cedr-m7"]
for name in names:
    model = AutoModelForSequenceClassification.from_pretrained(name,
            num_labels=4,
            id2label = { 0: "0", 1: "1", 2: "2", 3: "3"},
            label2id = { "0": 0, "1": 1, "2": 2, "3": 3},    
            ignore_mismatched_sizes=True,       
            problem_type="single_label_classification",
        )
    tok = AutoTokenizer.from_pretrained(name, model_max_length=512)

    train = pd.read_csv("../data/output/train.csv").rename(columns={"type": "labels"})
    test  = pd.read_csv("../data/output/test.csv").rename(columns={"type": "labels"})
    data = DatasetDict({
        "train":  Dataset.from_pandas(train[["text","labels"]]),
        "test": Dataset.from_pandas(test[["text","labels"]])
    })

    def preprocessing(raw):
        batch = tok (raw["text"],
            truncation=True,
            max_length=512,
            padding=False 
        )
        return batch

    tokenized_dataset = data.map(preprocessing, batched = True)

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    out_dir = os.path.join("../models", name)  

    args = TrainingArguments(
        output_dir = out_dir,
        eval_strategy = "epoch",
        save_strategy="epoch",
        logging_strategy="steps", 
        logging_steps=20, 
        logging_first_step = True,
        learning_rate = 2e-5,
        per_device_train_batch_size = 8,
        num_train_epochs = 3,
        weight_decay = 0.01,
        load_best_model_at_end = True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis = -1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["test"],
        tokenizer = tok,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate(tokenized_dataset["test"])
    best_ckpt = trainer.state.best_model_checkpoint

    results.append({
        "model": name,
        "best_checkpoint": best_ckpt,
        "eval_accuracy": round(float(eval_metrics.get("eval_accuracy", float("nan"))), 4),
        "eval_f1": round(float(eval_metrics.get("eval_f1", float("nan"))), 4),
        "eval_loss": round(float(eval_metrics.get("eval_loss", float("nan"))), 4),
        "train_loss": round(float(trainer.state.log_history[-1].get("train_loss", float("nan"))) if trainer.state.log_history else float("nan"), 4),
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv("bert_results.csv", index=False)

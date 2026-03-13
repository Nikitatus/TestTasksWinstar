import argparse
import os
import random
import warnings

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from datasets import Dataset

from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from seqeval.scheme import IOB2


model_name = "google-bert/bert-base-cased"
target_labels = ["O", "B-ANIMAL"]
label2id = {l: i for i, l in enumerate(target_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(target_labels)

def load_model():
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return model

def compute_metrics(predictions):
    predictions, labels = predictions
    predictions = np.argmax(predictions, axis=-1)

    all_predictions = []
    all_labels = []

    for seq_idx in range(len(predictions)):
        predictions_sample = []
        labels_sample = []
        for token_idx in range(len(predictions[seq_idx])):
            label = labels[seq_idx][token_idx]
            prediction = predictions[seq_idx][token_idx]
            if label.item() == -100:
                continue
            predictions_sample.append(id2label[prediction.item()])
            labels_sample.append(id2label[label.item()])

        all_predictions.append(predictions_sample)
        all_labels.append(labels_sample)

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, mode="strict", scheme=IOB2),
        "recall": recall_score(all_labels, all_predictions, mode="strict", scheme=IOB2),
        "f1": f1_score(all_labels, all_predictions, mode="strict", scheme=IOB2),
    }

def load_conll(path):
    sentences = []
    labels = []

    tokens = []
    tags = []

    with open(path) as f:
        for line in f:

            line = line.strip()

            if line == "":
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
                continue

            token, tag = line.split()
            tokens.append(token)
            tags.append(tag)

    return sentences, labels

def encode_labels(labels):
    return [[label2id[token] for token in seq] for seq in labels]

def tokenize_and_align(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []

    for i, label in enumerate(examples["ner_tags"]):

        word_ids = tokenized.word_ids(batch_index=i)
        prev_word = None
        label_ids = []

        for word_id in word_ids:

            if word_id is None:
                label_ids.append(-100)

            elif word_id != prev_word:
                label_ids.append(label[word_id])

            else:
                label_ids.append(label[word_id])

            prev_word = word_id

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

def load_dataset(path):
    tokens, labels = load_conll(path)

    dataset = Dataset.from_dict({
        "tokens": tokens,
        "ner_tags": labels
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = dataset.map(
        lambda x: {"ner_tags": encode_labels(x["ner_tags"])},
        batched=True
    )

    dataset = dataset.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)

    return dataset

def get_optimiser(model, lr):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=0.01)

    return optimizer


def train(args):
    warnings.filterwarnings("ignore", 
                            message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset(args.train_data_path)
    val_dataset = load_dataset(args.val_data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model().to(device)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )

    optimizer = get_optimiser(model, args.learning_rate)

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,

        seed=args.seed,
        report_to='none',
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )   

    print("The training begins!")
    train_result = trainer.train()
    eval_results = trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="models/ner_model")
    parser.add_argument("--logging_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

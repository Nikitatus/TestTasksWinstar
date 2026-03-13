import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

target_labels = ["O", "B-ANIMAL"]
id2label = {i: l for i, l in enumerate(target_labels)}

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def get_animal_words(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    word_ids = tokens.word_ids()

    animals = []

    current_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == current_word_id:
            continue
        current_word_id = word_id

        pred_label = id2label[predictions[idx].item()]
        if pred_label == "B-ANIMAL":
            token_span = tokens.word_to_chars(word_id)
            animals.append(text[token_span.start : token_span.end])

    return animals

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_dir", type=str, default="models/ner_model")

    parser.add_argument("--sentence", type=str, required=True)

    return parser.parse_args()

def predict(args):
    model, tokenizer = load_model(args.model_dir)
    animals = get_animal_words(args.sentence, model, tokenizer)

    if animals:
        print(f"Sentence : {args.sentence}")
        print(f"Animals  : {animals}")
    else:
        print("No animals found")

if __name__ == "__main__":
    predict(parse_args())

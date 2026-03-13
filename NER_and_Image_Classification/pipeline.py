import argparse

from NER_inference import load_model as load_ner_model, get_animal_words
from IMG_inference import load_model as load_img_model, classify_image

def check(image_path, sentence, ner_model, tokenizer, img_model, classes):
    animals = get_animal_words(sentence, ner_model, tokenizer)
    if not animals:
        return False

    image_label, confidence = classify_image(image_path, img_model, classes)

    match = all(a.lower() == image_label.lower() for a in animals)
    return match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--ner_model_dir", type=str, default="models/ner_model")
    parser.add_argument("--img_checkpoint", type=str, default="models/img_model/image_model.pt")
    args = parser.parse_args()

    ner_model, tokenizer = load_ner_model(args.ner_model_dir)
    img_model, classes = load_img_model(args.img_checkpoint)

    result = check(
        args.image, args.sentence,
        ner_model, tokenizer,
        img_model, classes,
    )

    print(f"Sentence : {args.sentence}")
    print(f"Image    : {args.image}")
    print(f"Result   : {result}")


if __name__ == "__main__":
    main()

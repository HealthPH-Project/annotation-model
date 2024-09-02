import torch
import os
import joblib
import pandas as pd
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    RobertaTokenizer,
)
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

output_folder = Path("./outputs")


def annotate_posts(tokenizer, model, post):
    text = post

    # Tokenize and make prediction
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)

    predicted_post = predictions[0]

    annotate = []

    annotations = ["AURI", "PN", "TB", "COVID"]

    for i, a in enumerate(predicted_post):
        if a == 1:
            annotate.append(annotations[i])

    return ",".join(annotate) if len(annotate) > 0 else "X"


def annotate_by_electra(file_path):
    electra_df = pd.DataFrame(pd.read_csv(file_path), columns=["posts"])

    electra_model = ElectraForSequenceClassification.from_pretrained(
        "./models/electra/ELECTRA_model"
    )
    electra_tokenizer = ElectraTokenizer.from_pretrained(
        "./models/electra/ELECTRA_save_tokenizer"
    )

    electra_model.eval()

    electra_df["annotations"] = electra_df.progress_apply(
        lambda x: annotate_posts(electra_tokenizer, electra_model, x["posts"]), axis=1
    )

    electra_df.to_csv("./outputs/electra_df.csv", index=False)


def annotate_by_roberta(file_path):
    roberta_df = pd.DataFrame(pd.read_csv(file_path), columns=["posts"])

    roberta_model = joblib.load("./models/roberta/roBERTa_model.pkl")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    roberta_model.eval()

    roberta_df["annotations"] = roberta_df.progress_apply(
        lambda x: annotate_posts(roberta_tokenizer, roberta_model, x["posts"]), axis=1
    )

    roberta_df.to_csv("./outputs/roberta_df.csv", index=False)


if __name__ == "__main__":

    os.makedirs(output_folder, exist_ok=True)

    annotate_by_electra("./inputs/test_raw_data.csv")
    annotate_by_roberta("./inputs/test_raw_data.csv")

import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import os
import sys
from torch.nn.functional import softmax
import shap
import pickle
import os
from tqdm import tqdm

def main(chunk, logits):
    silent = False if os.name == "nt" else True
    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    classifier = pipeline(
        'sentiment-analysis',
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True,
        max_length=512,
        top_k=None
    )

    explainer_probs = shap.Explainer(classifier, silent=silent)
    explainer_logits = shap.Explainer(shap.models.TransformersPipeline(classifier, rescale_to_logits=True), silent=silent)

    def find_file(chunk):
        for root, dirs, files in os.walk(os.path.join("datasets", "chunked")):
            for file in files:
                if f"chunk_{chunk}_" in file:  # Check if "chunk_0" is in the file name
                    return os.path.join(root, file)  # Return the first match
        return None  # Return None if no file matches


    # Usage
    dataset = find_file(chunk)
    if dataset:
        print(f"Found file: {dataset}")
    else:
        print(f"No file with 'chunk_{chunk}' found.")
    print(dataset)
    
    df = pd.read_csv(dataset)
    
    print(f"Running explainer with chunk {chunk} and logits {logits}.")

    if logits == "0":
        shap_values_probs = explainer_probs(df['text'])
        with open(os.path.join("shap_results", f"sv_probs_{os.path.basename(dataset)}.pkl"), "wb") as f:
            pickle.dump(shap_values_probs, f)
    elif logits == "1":
        shap_values_logits = explainer_logits(df['text'])
        with open(os.path.join("shap_results", f"sv_logits_{os.path.basename(dataset)}.pkl"), "wb") as f:
            pickle.dump(shap_values_logits, f)
    else:
        raise Exception("logits must be 0 or 1.")

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        raise Exception(f'Usage: {sys.argv[0]} chunk logits')

    if sys.argv[2] not in ["0", "1"]:
        raise Exception("logits must be 0 or 1.")
    
    main(sys.argv[1], sys.argv[2])

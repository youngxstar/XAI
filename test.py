import nlp
import numpy as np
import scipy as sp
import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification

import shap

# load a BERT sentiment analysis model
tokenizer = BertTokenizer.from_pretrained(r'./Models/google-bert/bert-base-uncased/tokenizer.json')
model = BertForSequenceClassification.from_pretrained('./Models/google-bert/bert-base-uncased', num_labels=2).cuda()


# define a prediction function
def f(x):
    tv = torch.tensor(
        [
            tokenizer.encode(v, padding="max_length", max_length=500, truncation=True)
            for v in x
        ]
    ).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
    return val


# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)

# explain the model's predictions on IMDB reviews
# imdb_train = nlp.load_dataset("imdb")["train"]

s = [
    "In this picture, there are four persons: my father, my mother, my brother and my sister."
]

# explainer = shap.Explainer(model, tokenizer)

shap_values = explainer(s)

# shap_values = explainer(imdb_train[:10], fixed_context=1)
# plot the first sentence's explanation
shap.plots.text(shap_values[3])
# plot the first sentence's explanation
shap.plots.text(shap_values[:3])
shap.plots.bar(shap_values.abs.sum(0))
shap.plots.bar(shap_values.abs.max(0))
shap.plots.bar(shap_values[:, "but"])
shap.plots.bar(shap_values[:, "but"])
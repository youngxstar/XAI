# %% [markdown]
# # Evaluation of XAI methods for Bert Models
# #### By Zhang, Yang
# ### Model: BERT
# ### Task: Sentiment Classification
# ### XAI: LIM, SHAP, captum
# ### TestMethod: Change sentences with different genders, times, occassions.

# ### Ref:
# https://captum.ai/
# https://captum.ai/tutorials/
# https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html
# https://huggingface.co/google-bert/bert-base-uncased/tree/main
# https://textattack.readthedocs.io/en/latest/2notebook/Example_5_Explain_BERT.html
# https://towardsdatascience.com/interpreting-the-prediction-of-bert-model-for-text-classification-5ab09f8ef074

# %%
import torch
import torch.nn as nn

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig

from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# We need to split forward pass into two part: 
# 1) embeddings computation
# 2) classification

# %%
def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)    


class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

# %%
bert_model_wrapper = BertModelWrapper(model)
ig = IntegratedGradients(bert_model_wrapper)

# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []

# %%
def interpret_sentence(model_wrapper, sentence, label=1):
    
    model_wrapper.eval()
    model_wrapper.zero_grad()
    
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
    input_embedding = model_wrapper.model.bert.embeddings(input_ids)
    
    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)

    print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy().tolist())
    add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label, delta, vis_data_records_ig)
    
    
def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()
    
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            str(label),
                            attributions.sum(),       
                            tokens[:len(attributions)],
                            delta))    

# %%
def generate_sentences(sentiment):
    sentences = []

    # Gender variations
    genders = ["She", "He", "They"]
    for gender in genders:
        sentence = f"{gender} feels thrilled about the news." if sentiment == "positive" else f"{gender} is devastated by the news."
        sentences.append(sentence)

    # Time variations
    # times = ["yesterday", "today", "tomorrow"]
    # for time in times:
    #     sentence = f"She was thrilled to receive the award {time}." if sentiment == "positive" else f"He received the news of the loss {time}."
    #     sentences.append(sentence)

    # Occasion variations
    occasions = ["surprise party", "upcoming vacation", "failure"]
    for occasion in occasions:
        sentence = f"He is enjoyable by the {occasion}." if sentiment == "positive" else f"They are distressed by the {occasion}."
        sentences.append(sentence)

    # Print the generated sentences
    for idx, sentence in enumerate(sentences):
        print(f"Sentence {idx + 1}: {sentence}")
    
    return sentences


# %%
vis_data_records_ig = []
sentences = generate_sentences('positive')
sentences = ["This movie was fantastic!"]
for setence_i in sentences:
    interpret_sentence(bert_model_wrapper, sentence=setence_i, label=0)
visualization.visualize_text(vis_data_records_ig)

# %%
vis_data_records_ig = []
sentences = generate_sentences('negative')
for setence_i in sentences:
    interpret_sentence(bert_model_wrapper, sentence=setence_i, label=1)
visualization.visualize_text(vis_data_records_ig)



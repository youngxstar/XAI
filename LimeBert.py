from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients
import shap
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# 设置环境变量和加载模型
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
tokenizer = BertTokenizer.from_pretrained(r'./Models/google-bert/bert-base-uncased/tokenizer.json')
model = BertForSequenceClassification.from_pretrained('./Models/google-bert/bert-base-uncased', num_labels=2)

# 将模型移到 GPU 上（如果可用）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
text = "This movie is bad!"
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入数据移动到正确的设备上
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)
print("Predicted class:", predictions.item())


# 定义情感分析函数
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    # Convert input_ids to torch.long type
    inputs = {key: value.long().to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits.cpu(), dim=1).numpy()
    return probabilities

# define a prediction function
def predict(x):
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

# 创建 XAI 解释器
lime_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
shap_explainer = shap.Explainer(predict, tokenizer)
ig = IntegratedGradients(model)

# 定义要解释的文本
text_to_explain = "This movie was fantastic! I loved it."

# 使用 LIME 生成解释
lime_exp = lime_explainer.explain_instance(text_to_explain, sentiment_analysis, num_features=10)
print("LIME Explanation:")
lime_exp.as_pyplot_figure()
print(lime_exp.as_list())

# 使用 SHAP 生成解释

shap_exp = shap_explainer([text_to_explain])
print("SHAP Explanation:")
shap.plots.text(shap_exp)

# 使用 Integrated Gradients 生成解释
inputs = tokenizer(text_to_explain, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
output = model(**inputs).logits
tv = torch.tensor(
        [
            tokenizer.encode(v, padding="max_length", max_length=500, truncation=True)
            for v in [text_to_explain]
        ]
    ).cuda()
outputs = model(tv).logits.detach().cpu().numpy()

# Convert input_ids to torch.long (int64) type
# Convert input_ids to torch.long type
input_ids = inputs['input_ids'].to(torch.long).cpu()

input_embedding = model_wrapper.model.bert.embeddings(input_ids)
output = torch.argmax(output).to(torch.long)

attributions, delta = ig.attribute(inputs=input_ids,target=output,return_convergence_delta=True)

# attributions = attributions.sum(dim=-1).squeeze(0)
print("Integrated Gradients Attributions:")
# print(attributions)




def generate_counterfactual(text, change_word, replacement_word):
    counterfactual_text = text.replace(change_word, replacement_word)
    original_pred = sentiment_analysis(text)[0]
    counterfactual_pred = sentiment_analysis(counterfactual_text)[0]
    return counterfactual_text, original_pred, counterfactual_pred

# 生成反事实解释
counterfactual_text, original_pred, counterfactual_pred = generate_counterfactual(text_to_explain, "fantastic", "terrible")
print("Original text:", text_to_explain)
print("Counterfactual text:", counterfactual_text)
print("Original prediction:", original_pred)
print("Counterfactual prediction:", counterfactual_pred)


def get_feature_importance(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions[-1].squeeze(0)  # 获取最后一层的注意力权重
    mean_attentions = attentions.mean(dim=0).cpu().detach().numpy()  # 计算平均注意力权重
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return dict(zip(tokens, mean_attentions))

# 获取特征重要性
feature_importance = get_feature_importance(text_to_explain)
print("Feature Importance:")
print(feature_importance)


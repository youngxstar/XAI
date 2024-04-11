# %% Load Model
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients
import torch
from captum.attr import visualization as viz

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = model.to(device)
def forward_func(input, mask=None):   
    input = torch.round(input).long().to(device)
    return model(input_ids=input, attention_mask=mask)[0]

# Choose the layer for which you want to compute attributions
layer = model.distilbert.embeddings
lig = LayerIntegratedGradients(forward_func, layer)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# %% Import Dataset
import os
import pandas as pd
def load_imdb_data(data_dir, num_pos=10, num_neg=10):
    data = []
    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            folder = os.path.join(data_dir, split, sentiment)
            filenames = os.listdir(folder)[:num_pos] if sentiment == 'pos' else os.listdir(folder)[:num_neg]
            for filename in filenames:
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    review = file.read()
                    data.append({'review': review, 'sentiment': sentiment})
    return pd.DataFrame(data)

data_dir = './Datasets/aclImdb'  # 指定IMDb数据集的文件夹路径
imdb_data = load_imdb_data(data_dir, num_pos=1000, num_neg=1000)

# 打印数据集的前几行
print(imdb_data.head())



# %% Import sst2
from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
print(dataset)

def replace_gender_keywords(sentence, gender_keywords):
    for male, female in zip([" he ", " his ", " him ", " man ", " boy ", " male "], \
                            [" she ", " her ", " her ", " woman ", " girl ", " female "]):
        if male in sentence:
            return sentence.replace(male, female)
        elif female in sentence:
            return sentence.replace(female, male)
    return sentence



# %%
import numpy as np
import matplotlib.pyplot as plt

gender_keywords = [" he ", " she ", " his ", " her ", " him ", " man ", " woman ", " boy ", " girl ", " male ", " female "]
# 初始化用于存储属性值的字典
gender_attributions_pos = {keyword: [] for keyword in gender_keywords}
gender_attributions_neg = {keyword: [] for keyword in gender_keywords}


# 遍历数据集
stop = 67349
for example in dataset['train']:
    input_text = example['sentence']
    true_label = example['label']
    stop-=1
    print(stop)
    if stop<=0:
        break
    # 检查句子是否包含性别相关的关键词
    for keyword in gender_keywords:
        if keyword in input_text.lower():
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            

            # Creating a baseline
            baseline_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)
            baseline_mask = torch.ones_like(attention_mask)
            # Mask the beginning and end tokens in the baseline_mask
            baseline_mask[:, 0] = 0  # Mask the first token ([CLS])
            baseline_mask[:, -1] = 0  # Mask the last token ([SEP])

            pred_class = model(input_ids, attention_mask)
            target = pred_class.logits.argmax().detach()

            attributions, _ = lig.attribute(inputs=(input_ids, attention_mask),
                                            baselines=(baseline_ids, baseline_mask),
                                            target=target,
                                            n_steps=200,
                                            return_convergence_delta=True)
            attributions_sum = summarize_attributions(attributions)

            # 获取性别关键词的索引
            token_index = tokenizer.convert_ids_to_tokens(input_ids[0]).index(keyword.strip())

            # 根据真实标签将属性值添加到相应的列表中
            if true_label == 1:
                gender_attributions_pos[keyword].append(attributions_sum[token_index].item())
            else:
                gender_attributions_neg[keyword].append(attributions_sum[token_index].item())

# %% plot
# 计算正面情绪和负面情绪句子中性别相关词汇的平均属性值
# avg_attributions_pos = {keyword: np.mean(values) for keyword, values in gender_attributions_pos.items() if values}
# avg_attributions_neg = {keyword: np.mean(values) for keyword, values in gender_attributions_neg.items() if values}

avg_attributions_pos = {keyword: np.mean(values) if values else 0 for keyword, values in gender_attributions_pos.items()}
avg_attributions_neg = {keyword: np.mean(values) if values else 0 for keyword, values in gender_attributions_neg.items()}


# 可视化结果
keywords = list(avg_attributions_pos.keys())
pos_values = [avg_attributions_pos[keyword] for keyword in keywords]
neg_values = [avg_attributions_neg[keyword] for keyword in keywords]

x = np.arange(len(keywords))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pos_values, width, label='Positive')
rects2 = ax.bar(x + width/2, neg_values, width, label='Negative')

ax.set_ylabel('Average Attribution')
ax.set_title('Average Attribution of Gender Keywords by Sentiment')
ax.set_xticks(x)
ax.set_xticklabels(keywords)
ax.legend()

fig.tight_layout()
plt.xticks(rotation=45)
plt.show()


# %% Version2
import matplotlib.pyplot as plt
import numpy as np

# 计算正面情绪和负面情绪句子中的样本数量
sample_counts_pos = {keyword: len(values) for keyword, values in gender_attributions_pos.items()}
sample_counts_neg = {keyword: len(values) for keyword, values in gender_attributions_neg.items()}

# 可视化结果
keywords = list(avg_attributions_pos.keys())
pos_values = [avg_attributions_pos[keyword] for keyword in keywords]
neg_values = [avg_attributions_neg[keyword] for keyword in keywords]
pos_counts = [sample_counts_pos[keyword] for keyword in keywords]
neg_counts = [sample_counts_neg[keyword] for keyword in keywords]

x = np.arange(len(keywords))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pos_values, width, label='Positive')
rects2 = ax.bar(x + width/2, neg_values, width, label='Negative')

ax.set_ylabel('Average Attribution')
ax.set_title('Average Attribution of Gender Keywords by Sentiment')
ax.set_xticks(x)
ax.set_xticklabels(keywords, rotation=45)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

fig.tight_layout()

# 添加样本数量注释
for rect, count in zip(rects1, pos_counts):
    height = rect.get_height()
    if height < 0:
        # 对于负数，将标签向下移动
        ax.annotate(f'{count}', xy=(rect.get_x() + rect.get_width() / 2, height), 
                    xytext=(0, -3), textcoords='offset points', ha='center', va='top')
    else:
        # 对于正数，将标签向上移动
        ax.annotate(f'{count}', xy=(rect.get_x() + rect.get_width() / 2, height), 
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

for rect, count in zip(rects2, neg_counts):
    height = rect.get_height()
    if height < 0:
        # 对于负数，将标签向下移动
        ax.annotate(f'{count}', xy=(rect.get_x() + rect.get_width() / 2, height), 
                    xytext=(0, -3), textcoords='offset points', ha='center', va='top')
    else:
        # 对于正数，将标签向上移动
        ax.annotate(f'{count}', xy=(rect.get_x() + rect.get_width() / 2, height), 
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

plt.show()

# %% Stat based on gender
# 假设你已经遍历了数据集，并且已经收集了所有性别关键词的属性值
# gender_attributions_pos 和 gender_attributions_neg 应包含由之前遍历得到的属性值

# 将性别关键词分为男性和女性组
male_keywords = [" he ", " his ", " him ", " man ", " boy ", " male "]
female_keywords = [" she ", " her ", " woman ", " girl ", " female "]

# 初始化男性和女性的总体贡献度计数
total_contributions = {
    'male': {
        'positive': 0.0,
        'negative': 0.0,
        'count_positive': 0,
        'count_negative': 0
    },
    'female': {
        'positive': 0.0,
        'negative': 0.0,
        'count_positive': 0,
        'count_negative': 0
    }
}

# 计算每个性别关键词组的总体贡献度
for keyword in gender_attributions_pos:
    if keyword in male_keywords:
        total_contributions['male']['positive'] += np.sum(gender_attributions_pos[keyword])
        total_contributions['male']['count_positive'] += len(gender_attributions_pos[keyword])
    elif keyword in female_keywords:
        total_contributions['female']['positive'] += np.sum(gender_attributions_pos[keyword])
        total_contributions['female']['count_positive'] += len(gender_attributions_pos[keyword])

for keyword in gender_attributions_neg:
    if keyword in male_keywords:
        total_contributions['male']['negative'] += np.sum(gender_attributions_neg[keyword])
        total_contributions['male']['count_negative'] += len(gender_attributions_neg[keyword])
    elif keyword in female_keywords:
        total_contributions['female']['negative'] += np.sum(gender_attributions_neg[keyword])
        total_contributions['female']['count_negative'] += len(gender_attributions_neg[keyword])

# 现在 total_contributions 字典中包含了男性和女性关键词对正面和负面情绪的总贡献度及计数

# 接下来进行可视化
categories = ['Positive', 'Negative']
male_values = [total_contributions['male']['positive']/total_contributions['male']['count_positive'], total_contributions['male']['negative']/total_contributions['male']['count_negative']]
female_values = [total_contributions['female']['positive']/total_contributions['female']['count_positive'], total_contributions['female']['negative']/total_contributions['female']['count_negative']]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, male_values, width, label='Male')
rects2 = ax.bar(x + width/2, female_values, width, label='Female')

# 添加标签显示贡献的样本数量
for rect, label in zip(rects1, ['count_positive', 'count_negative']):
    ax.annotate(f'{total_contributions["male"][label]}',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

for rect, label in zip(rects2, ['count_positive', 'count_negative']):
    ax.annotate(f'{total_contributions["female"][label]}',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_ylabel('Total Average Attribution')
ax.set_title('Total Average Attribution of Gender Keywords by Sentiment')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

fig.tight_layout()
plt.show()



# %%

sentences_with_labels = [
    # 性别相关句子
    ("She is a great leader and very competent.", 1),
    ("He is a great leader and very competent.", 1),
    ("She is emotional and irrational.", 0),
    ("He is emotional and irrational.", 0),
    
    # 场合相关句子
    ("This party is amazing and everyone is having a great time!", 1),
    ("This funeral is solemn and everyone is mourning.", 0),
    ("The meeting was productive and everyone contributed.", 1),
    ("The argument was heated and everyone was upset.", 0),
    
    # 年龄相关句子
    ("The young boy is energetic and full of life.", 1),
    ("The old man is wise and full of experience.", 1),
    ("The teenager is rebellious and unpredictable.", 0),
    ("The elderly woman is frail and forgetful.", 0),
]


# 定义一个映射从类别索引到情感标签
# index_to_label = {1: "Positive", 0: "Negative", -1: "Neutral"}
index_to_label = {1: "Positive", 0: "Negative"}

# Process each sentence
# for input_text, true_label in sentences_with_labels:

# for index, row in imdb_data.iterrows():
#     input_text = row['review']
#     true_label = 1 if row['sentiment'] == 'pos' else 0
    
# 遍历训练集
for example in dataset['train']:
    input_text = example['sentence']
    true_label = example['label']
    
    gender_keywords = [" he ", " she ", " his ", " her ", " him ", " man ", " woman ", " boy ", " girl ", " male ", " female "]
    if any(keyword in input_text.lower() for keyword in gender_keywords)==False:
        continue # if not related to gender, then pass
    
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    # attention_mask[:, 0] = 0  # Mask the first token ([CLS])
    # attention_mask[:, -1] = 0  # Mask the last token ([SEP])
    
    # Creating a baseline
    baseline_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)
    baseline_mask = torch.ones_like(attention_mask)
    # Mask the beginning and end tokens in the baseline_mask
    baseline_mask[:, 0] = 0  # Mask the first token ([CLS])
    baseline_mask[:, -1] = 0  # Mask the last token ([SEP])
    
    pred_class = model(input_ids, attention_mask)
    target = pred_class.logits.argmax().detach()
    
    # 获取模型预测的情感标签
    predicted_label = index_to_label[target.cpu().item()]
    
    # Compute attributions
    attributions, delta = lig.attribute(inputs=(input_ids, attention_mask),
                                        baselines=(baseline_ids, baseline_mask),
                                        target=target,  # Use the predicted class as target
                                        n_steps=200,
                                        return_convergence_delta=True)

    attributions_sum = summarize_attributions(attributions)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    viz.visualize_text([viz.VisualizationDataRecord(
                            word_attributions=attributions_sum,
                            pred_prob= torch.max(pred_class[0]),
                            pred_class= target,
                            true_class=true_label,
                            attr_class=predicted_label,  # 使用预测的情感标签
                            attr_score=attributions_sum.sum(),
                            raw_input_ids=tokens,
                            convergence_score=delta)])

# %%

import matplotlib.pyplot as plt

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().numpy()

plt.bar(tokens, attributions_sum)
plt.xticks(rotation=90)
plt.xlabel('Tokens')
plt.ylabel('Attribution')
plt.show()

cumulative_attributions = attributions_sum.cumsum()

plt.plot(tokens, cumulative_attributions)
plt.xticks(rotation=90)
plt.xlabel('Tokens')
plt.ylabel('Cumulative Attribution')
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

true_labels = [label for _, label in sentences_with_labels]
pred_labels = [model(tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids']).logits.argmax().item() for text, _ in sentences_with_labels]

cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %%

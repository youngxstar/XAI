import numpy as np
import matplotlib.pyplot as plt

# 初始化用于存储属性值的字典
gender_attributions_pos = {keyword: [] for keyword in gender_keywords}
gender_attributions_neg = {keyword: [] for keyword in gender_keywords}

# 遍历数据集
for example in dataset['test']:
    input_text = example['sentence']
    true_label = example['label']

    # 检查句子是否包含性别相关的关键词
    for keyword in gender_keywords:
        if keyword in input_text.lower():
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            pred_class = model(input_ids, attention_mask)
            target = pred_class.logits.argmax().detach()

            attributions, _ = lig.attribute(inputs=(input_ids, attention_mask),
                                            baselines=(baseline_ids, baseline_mask),
                                            target=target,
                                            n_steps=200,
                                            return_convergence_delta=False)
            attributions_sum = summarize_attributions(attributions)

            # 获取性别关键词的索引
            token_index = tokenizer.convert_ids_to_tokens(input_ids[0]).index(keyword.strip())

            # 根据真实标签将属性值添加到相应的列表中
            if true_label == 1:
                gender_attributions_pos[keyword].append(attributions_sum[token_index].item())
            else:
                gender_attributions_neg[keyword].append(attributions_sum[token_index].item())

# 计算正面情绪和负面情绪句子中性别相关词汇的平均属性值
avg_attributions_pos = {keyword: np.mean(values) for keyword, values in gender_attributions_pos.items() if values}
avg_attributions_neg = {keyword: np.mean(values) for keyword, values in gender_attributions_neg.items() if values}

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

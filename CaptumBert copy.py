from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import IntegratedGradients
import torch

# 加载模型和分词器
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# 将模型移到 GPU 上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 适配器：调整模型的输出以符合 IntegratedGradients 的要求
def model_output(input):
    # 将浮点插值点四舍五入到最接近的整数
    input = torch.round(input).long()
    return model(input)[0]




# 使用 Captum 的 Integrated Gradients
ig = IntegratedGradients(model_output)

# 定义要解释的文本
text_to_explain = "I love this movie! It's so amazing."
inputs = tokenizer(text_to_explain, return_tensors='pt', truncation=True, padding=True)
inputs_for_model = {key: value.to(device) for key, value in inputs.items()}

# 获取模型预测的目标类别
output = model(**inputs_for_model)
target_class = output.logits.argmax(dim=1).item()

# 获取模型的输入张量
input_ids = inputs_for_model['input_ids']
input_ids = input_ids.long()

# 计算属性
attributions, delta = ig.attribute(inputs=input_ids, baselines=input_ids * 0, target=target_class, return_convergence_delta=True,allow_unused=True)


# 属性可视化和处理...

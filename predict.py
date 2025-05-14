from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/qwen/Qwen2-1.5B-Instruct/'
lora_path = './qwen2_finetuned/checkpoint-23823' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", trust_remote_code=True)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

def generate(instruction, input_text):
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例测试
print(generate(
    "阅读题目并选择正确答案",
    "生物题：下列关于光合作用光反应阶段的叙述，错误的是：\nA. 发生在叶绿体类囊体薄膜上\nB. 水的光解产生氧气和[H]\nC. 能量转换过程为：光能→ATP中活跃化学能\nD. 直接生成葡萄糖作为产物"
))
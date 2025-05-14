import json

# 学科定制化 Prompt 模板
SUBJECT_PROMPT_TEMPLATES = {
    "语文": "请仔细阅读以下文本，注意语言表达和修辞手法，然后回答问题。\n文本：{context}\n问题：{question}",
    "英语": "Read the following passage and answer the question in English.\nPassage: {context}\nQuestion: {question}",
    "政治": "请结合政治学科知识分析以下材料：\n材料：{context}\n问题：{question}",
    "生物": "根据生物学原理回答以下问题：\n实验/现象：{context}\n问题：{question}",
    "历史": "请从历史角度分析以下史料：\n史料：{context}\n问题：{question}",
    "地质": "根据地学知识回答：\n地质资料：{context}\n问题：{question}"
}

ANSWER_TEMPLATE = "正确答案：{answer}\n解析：{analysis}"

def format_to_qwen2(input_file: str, output_file: str):
    """
    将 COIG 考试数据转换为 Qwen2 微调格式
    Args:
        input_file:  输入的 COIG JSONL 文件路径
        output_file: 输出的 Qwen2 JSONL 文件路径
    """
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line.strip())
                subject = str(data.get("subject", "")).strip()
                context = str(data.get("textbox_q_context", "")).strip()
                question = str(data.get("textbox_question", "")).strip()
                answer = str(data.get("textbox_answer", "")).strip()
                analysis = str(data.get("textbox_answer_analysis", "")).strip()

                # 选择学科模板（默认回退）
                prompt_template = SUBJECT_PROMPT_TEMPLATES.get(
                    subject, 
                    "请回答以下问题：\n上下文：{context}\n问题：{question}"
                )
                
                # 构建指令和输入
                instruction = data.get("textbox_q_instruction", "请根据题目要求回答问题").strip()
                input_text = prompt_template.format(context=context, question=question)
                
                # 构建标准化输出
                output_text = ANSWER_TEMPLATE.format(answer=answer, analysis=analysis)
                
                # 写入JSONL（保留学科信息）
                f_out.write(json.dumps({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "subject": subject  # 可选：用于后续分析
                }, ensure_ascii=False) + "\n")
            
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失败: {line[:50]}... (错误: {e})")
                continue

if __name__ == "__main__":
    # 使用示例
    format_to_qwen2(
        input_file = "./dataset/exam_instructions.jsonl",
        output_file = "./dataset/qwen2_sft_format.jsonl"
    )
    print("✅ 数据转换完成！")

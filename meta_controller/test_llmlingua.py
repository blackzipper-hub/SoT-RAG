from newLlmlingua import PromptCompressor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time

dtype = "half"
model_name = "/mnt/e/graduation_design/effecient_llm/model_project/model_ckpt/self-rag-gf"

# 初始化模型
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if dtype == "half" else torch.float32
)
model.eval()

llm_lingua = PromptCompressor(model=model, tokenizer=tokenizer)

# 从txt文件读取内容
text = "who is joe biden"

# 转换为 StringList（增强处理）
string_list = [
    line.strip()                      # 去除前后空格和换行符
    for line in text.split('\n')      # 按换行符分割
    if line.strip() != ''             # 过滤空段落
    and not line.startswith('#')      # 可选：过滤注释行（以#开头）
]

# 压缩提示（添加错误处理）
try:
    start = time.time()
    compressed_prompt = llm_lingua.compress_prompt(
        context=string_list,
        instruction="",
        question="",
        # rate = 0.2,
        target_token=5,
    )

    end = time.time()
    total_time = end - start
    print("Compression time:", total_time)
    print("压缩后的提示：")
    print(compressed_prompt["compressed_prompt"])
except Exception as e:
    print(f"处理出错: {str(e)}")
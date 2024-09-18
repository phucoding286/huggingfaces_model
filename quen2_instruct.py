from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator, disk_offload

device = "cuda" if torch.cuda.is_available() else "cpu"

# Khởi tạo Accelerator
# accelerator = Accelerator()

# Tải mô hình
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    cache_dir="E:/transformers_cache"
)
# Offload mô hình lên đĩa
# model = disk_offload(model, offload_dir="E:/transformers_cache")

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    cache_dir="E:/transformers_cache"
)

# Tạo prompt và đầu vào cho mô hình
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

while True:
    inp = input("You: ")
    messages.append({"role": "user", "content": inp})

    # Tạo đầu vào cho mô hình
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(device)

    # Sinh văn bản
    generated_ids = model.generate(
        model_inputs.input_ids.to(device),
        attention_mask=model_inputs.attention_mask.to(device),
        max_new_tokens=30,
        min_length=4,
        num_beams=1,
        top_k=10,
        top_p=0.9,
        temperature=0.7,
        early_stopping=True
    )

    # Giải mã kết quả
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(response)

    messages.append({"role": "assistant", "content": response})
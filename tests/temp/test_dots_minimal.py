"""
dots.ocr-1.5 最小测试 - 解决空输出问题
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import zipfile
from io import BytesIO

print('[1/5] Loading model...')
processor = AutoProcessor.from_pretrained('data/model/dots.ocr-1.5', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'data/model/dots.ocr-1.5',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa'
).to('cuda').eval()
print('  [OK] Model loaded')

# 检查显存使用
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f'  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')

print('[2/5] Loading test image...')
with zipfile.ZipFile('data/ocrbenchmark/HME100K/test.zip') as zf:
    with zf.open('test_images/test_2.jpg') as fp:
        image = Image.open(BytesIO(fp.read())).convert('RGB')
print(f'  [OK] Image loaded: {image.size}')

print('[3/5] Building input...')
messages = [{
    'role': 'user',
    'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': 'Recognize this formula:'},
    ],
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f'  Template: {text[:100]}...')

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors='pt'
)

print(f'  Input keys: {list(inputs.keys())}')
print(f'  Input IDs: {inputs["input_ids"].shape}')

# 移动到GPU并过滤 mm_token_type_ids
inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if 'mm_token_type_ids' in inputs:
    print('  [WARN] Filtering mm_token_type_ids')
    inputs = {k: v for k, v in inputs.items() if k != 'mm_token_type_ids'}

print('[4/5] Generating...')
input_len = inputs['input_ids'].shape[1]

# 尝试不同的生成策略
print('  Strategy: min_new_tokens=10 to force generation')
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        min_new_tokens=10,  # 强制生成至少10个token
        max_new_tokens=50,
        do_sample=False
    )

new_tokens = outputs.shape[1] - input_len
print(f'  Output shape: {outputs.shape}')
print(f'  New tokens generated: {new_tokens}')

# 再次检查显存
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f'  GPU Memory after generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')

print('[5/5] Decoding...')
result = processor.decode(outputs[0, input_len:], skip_special_tokens=True)
print(f'  Result: [{result}]')
print(f'  Length: {len(result)}')

if len(result) == 0:
    print('\n[ERROR] EMPTY OUTPUT - Checking token IDs...')
    print(f'  Generated token IDs: {outputs[0, input_len:input_len+10]}')
    print(f'  Decoded with special tokens: {processor.decode(outputs[0, input_len:input_len+10], skip_special_tokens=False)}')
else:
    print(f'\n[SUCCESS] Got non-empty output')

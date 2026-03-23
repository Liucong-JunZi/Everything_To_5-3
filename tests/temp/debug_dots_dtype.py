"""
调试 dots.ocr-1.5 dtype 问题
找出哪里还有 bfloat16
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import zipfile
from io import BytesIO

print('='*60)
print('dots.ocr-1.5 dtype 调试')
print('='*60)

print('\n[1] Loading model...')
processor = AutoProcessor.from_pretrained('data/model/dots.ocr-1.5', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'data/model/dots.ocr-1.5',
    trust_remote_code=True,
    torch_dtype=torch.float32,
    attn_implementation='sdpa'
).to('cpu').float().eval()

print(f'  Model param dtype: {next(model.parameters()).dtype}')

print('\n[2] Preparing inputs...')
with zipfile.ZipFile('data/ocrbenchmark/HME100K/test.zip') as zf:
    with zf.open('test_images/test_2.jpg') as fp:
        image = Image.open(BytesIO(fp.read())).convert('RGB')

messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': 'OCR:'}]}]

# Step 1: process_vision_info
print('\n  Step 1: process_vision_info')
image_inputs, video_inputs = process_vision_info(messages)
print(f'    image_inputs type: {type(image_inputs)}')
if image_inputs:
    for i, img in enumerate(image_inputs):
        img_size = img.size if hasattr(img, 'size') else 'N/A'
        print(f'      [{i}] type={type(img)}, size={img_size}')

# Step 2: processor
print('\n  Step 2: processor')
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')

print(f'    Input keys: {list(inputs.keys())}')
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f'      {k}: dtype={v.dtype}, shape={v.shape}')

# Step 3: Convert dtypes
print('\n  Step 3: Converting dtypes to float32')
for k, v in inputs.items():
    if isinstance(v, torch.Tensor) and v.dtype in (torch.bfloat16, torch.float16):
        inputs[k] = v.float()
        print(f'    Converted {k}: {v.dtype} -> {inputs[k].dtype}')

print('\n  After conversion:')
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f'    {k}: dtype={v.dtype}')

# Step 4: Filter mm_token_type_ids
if 'mm_token_type_ids' in inputs:
    print('\n  Step 4: Filtering mm_token_type_ids')
    inputs = {k: v for k, v in inputs.items() if k != 'mm_token_type_ids'}
    print(f'    Remaining keys: {list(inputs.keys())}')

print('\n[3] Generating...')
try:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    input_len = inputs['input_ids'].shape[1]
    result = processor.decode(outputs[0, input_len:], skip_special_tokens=True)
    print(f'  SUCCESS!')
    print(f'  Result: [{result}]')
    print(f'  Length: {len(result)}')

except Exception as e:
    print(f'  ERROR: {e}')

    # Check if any parameters are still bfloat16
    print('\n  Checking model parameters...')
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f'    {name}: {param.dtype}')
            break
    else:
        print('    All params are float32 ✓')

    # Check inputs again
    print('\n  Checking inputs again...')
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype != torch.float32:
                print(f'    {k}: {v.dtype} <- STILL WRONG!')
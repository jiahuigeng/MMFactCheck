from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoTokenizer

import torch
import torchvision.transforms as T
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests
import base64
from torchvision.transforms.functional import InterpolationMode

# def encode_image(image_id):
#     image = Image.open(requests.get(url, stream=True).raw)

def load_image_llava(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_intern(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_llava16_model(model_size):
    if model_size == "small":
        model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'
    elif model_size == "medium":
        model_id = 'llava-hf/llava-v1.6-vicuna-13b-hf'
    elif model_size == "large":
        model_id = 'llava-hf/llava-1.6-34b-hf'
    else:
        return Exception(f"Invalid model size: {model_size}")
    # cache_dir = "/data/zhuderui/huggingface/hub"
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto",
                                                              low_cpu_mem_usage=True, load_in_4bit=True)
    # model.to("cuda:0")
    return model, processor

def get_gpt_model():
    openai_key_file = "openai_api.txt"
    openai_key = open(openai_key_file, "r").read()
    model = OpenAI(api_key=openai_key)
    return model

def prompt_gpt4(model, prompt, image_id):
    with open(image_id, "rb") as image_file:
        base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    response = model.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}"
                            # "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    res = response.choices[0].message.content
    return res



def get_model_and_processor(args):
    if args.model == "llava":
        if args.model_size == "small":
            model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'
        elif args.model_size == "medium":
            model_id = 'llava-hf/llava-v1.6-vicuna-13b-hf'
        elif args.model_size == "large":
            model_id = 'llava-hf/llava-1.6-34b-hf'
        else:
            return Exception(f"Invalid llava model size: {args.model_size}")
        # cache_dir = "/data/zhuderui/huggingface/hub"
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                  device_map="auto",
                                                                  low_cpu_mem_usage=True, load_in_4bit=True)
        # model.to("cuda:0")
        return model, processor

    elif args.model == "intern":
        if args.model_size == "small":
            model_id = 'OpenGVLab/InternVL2-4B'
        elif args.model_size == "medium":
            model_id = 'OpenGVLab/InternVL2-8B'
        elif args.model_size == "large":
            model_id = 'OpenGVLab/InternVL2-26B'
        else:
            return Exception(f"Invalid intern model size: {args.model_size}")
        # cache_dir = "/data/zhuderui/huggingface/hub"
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

        return model, tokenizer

    elif args.model == "gpt4":
        model = None
        return model, None


# def prompt_llava16(model, processor, image, human_input):
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": human_input},
#                 {"type": "image"},
#             ],
#         },
#     ]
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
#     output = model.generate(**inputs, max_new_tokens=500)
#     output = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[0]
#     return output



def prompt_llava16(model, processor, image_id, human_input):

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": human_input},
                {"type": "image"},
            ],
        },
    ]
    image = load_image_llava(image_id)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=500)
    output = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[0]
    return output



def ensemble_prompt():
    pass



if __name__ == "__main__":
    # model, processor = get_llava16_model("small")
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    # image = load_image_llava(url)
    # image = url
    # query = "What is shown in this image?"
    # print(prompt_llava16(model, processor, image, query))

    model = get_gpt_model()
    res = prompt_gpt4(model, "describe this image", "view.jpg")
    print(res)
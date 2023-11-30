import base64
import requests
from PIL import Image
from io import BytesIO



import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}




GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinationsshould be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not countas necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''


# OpenAI API Key
API_KEY = "YOUR_API_KEY"



def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True




def call_api(prompt, image_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json().keys())
    return response.json()


def get_gpt4v_answer(prompt, image_path):
    while 1:
        try:
            res = call_api(prompt, image_path)
            if "choices" in res.keys():
                return res["choices"][0]["message"]["content"]
            else:
                assert False
        except Exception as e:
            print("retry")
            # pass
    # return call_api(prompt, image_path)


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data_path", type=str, default="COCO_2014/val2014/", help="data path")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")

parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
args = parser.parse_known_args()[0]



os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)




img_files = os.listdir(args.data_path)
random.shuffle(img_files)

base_path = "log/gpt4v-eval"
if not os.path.exists(base_path + f"/{args.model}"):
    os.mkdir(base_path + f"/{args.model}")

gpt_answer_records = {}
assistant_answer_records = {}
avg_hal_score_1 = 0
avg_hal_score_2 = 0
avg_det_score_1 = 0
avg_det_score_2 = 0
num_count = 0

for idx in range(50):
    img = img_files[idx]
    image_path = args.data_path + img
    raw_image = Image.open(image_path)
    raw_image = raw_image.convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)
    qu = "Please describe this image in detail."

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)
    assistant_answer_records[str(img)] = {}

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu}, 
                use_nucleus_sampling=False, 
                num_beams=5,
                max_new_tokens=512,
            )
    model_response_1 = out[0]
    assistant_answer_records[str(img)]["assistant_1"] = model_response_1
    print("Beam-5 output:") 
    print(model_response_1)


    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu}, 
                use_nucleus_sampling=False, 
                num_beams=5,
                max_new_tokens=512,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )
    model_response_2 = out[0]
    assistant_answer_records[str(img)]["assistant_2"] = model_response_2
    print("OPERA output:")
    print(model_response_2)

    # gpt-4v eval
    prompt = GPT_JUDGE_PROMPT.format(model_response_1, model_response_2)

    gpt_answer = get_gpt4v_answer(prompt, image_path)
    print(gpt_answer)
    gpt_answer_records[str(img)] = gpt_answer
    print(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" "))
    print(len(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")))
    try:
        hal_score_1, hal_score_2 = gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")
        det_score_1, det_score_2 = gpt_answer.split("Detailedness: ")[-1].split("\n")[0].split(" ")
    except:
        continue
    avg_hal_score_1 += int(hal_score_1)
    avg_hal_score_2 += int(hal_score_2)
    avg_det_score_1 += int(det_score_1)
    avg_det_score_2 += int(det_score_2)
    num_count += 1
    print("=========================================")

    # dump metric file
    with open(os.path.join(base_path + f"/{args.model}", 'answers.json'), "w") as f:
        json.dump(assistant_answer_records, f)

    # dump metric file
    with open(os.path.join(base_path + f"/{args.model}", 'records.json'), "w") as f:
        json.dump(gpt_answer_records, f)

avg_score = float(avg_hal_score_1) / num_count
avg_score = float(avg_hal_score_2) / num_count
avg_score = float(avg_det_score_1) / num_count
avg_score = float(avg_det_score_2) / num_count
print(f"The avg hal score for Assistant 1 and Assistent 2: {avg_hal_score_1}; {avg_hal_score_2}")
print(f"The avg det score for Assistant 1 and Assistent 2: {avg_det_score_1}; {avg_det_score_2}")
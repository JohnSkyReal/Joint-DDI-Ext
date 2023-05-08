import os
import json
import torch
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="DrugBank_test.xls")
parser.add_argument("--output_path", type=str, default="prediction_drugbank_opt-lora-drug-ft10-gpt3_form")
parser.add_argument("--model_path", type=str, default="facebook/opt-6.7b")
parser.add_argument("--lora_path", type=str, default="opt-lora-ft10-gpt3_form")
parser.add_argument("--test_batch_size", type=int, default=128)
args = parser.parse_args()

# 加载model与tokenizer
peft_model_id = args.lora_path
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.padding_side = "left"

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


# dump:写入json文件
def dump_json(path, data):
    with open(path, 'w', encoding='utf-8')as f:
        json.dump(data, f)


def batch_completion(all_pairs, save_path):
    batch_size = args.test_batch_size
    input_texts = []
    for pair in all_pairs:
        input_text = pair['prompt']
        input_texts.append(input_text)
    print('----example----\n', input_texts[0])
    # Split input_texts into smaller batches
    input_text_batches = [input_texts[i:i + batch_size] for i in range(0, len(input_texts), batch_size)]

    i = 0
    for input_text_batch in input_text_batches:
        batch = tokenizer(
            input_text_batch,
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].cuda()
        print(input_ids)
        # print(tokenizer.decode(input_ids))

        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=512)
            for s in output_tokens:
                print(i+1)
                response = tokenizer.decode(s, skip_special_tokens=True)
                print(response)
                print('-' * 20)
                all_pairs[i]['response'] = response
                json_path = save_path + r'/response_{}.json'.format(i+1)
                dump_json(json_path, all_pairs[i])
                i += 1


def auto_completion(input_text):
    batch = tokenizer(input_text, return_tensors="pt")
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=512)
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(response)
        print('-' * 20)
    return response


def extract_prompt(lines):
    all_pairs = []
    for line in lines:
        pair = {
            'prompt': line["input"],
            'label': line["output"],
            'response': ''
        }
        all_pairs.append(pair)
    return all_pairs


def main():
    save_path = args.output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = args.data_path
    # lines = readxls(path)
    lines = load_dataset("json", data_files=path)
    all_pairs = extract_prompt(lines["train"])
    print(len(all_pairs))
    batch_completion(all_pairs, save_path)
    # i = 1
    # for pair in all_pairs:
    #     print(i)
    #     prompt = pair['prompt']
    #     response = auto_completion(prompt)
    #     pair['response'] = response
    #     json_path = save_path + r'/response_{}.json'.format(i)
    #     dump_json(json_path, pair)
    #     i += 1
    #     # time.sleep(2)


if __name__ == '__main__':
    main()

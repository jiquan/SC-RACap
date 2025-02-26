import ast
import json
import logging
import os
import re
from typing import List, Optional, Tuple
from openai import OpenAI

from evaluate_clair import load_scracap_data

deepseek_client = OpenAI(
    api_key='',
    base_url="https://api.deepseek.com",
)

client = OpenAI(
    api_key="",
)

_REL_PROMPT = """\
### 
Instruction:
Perform the surgical relation triplet extraction task.
Give you the surgical image caption and pre-defined relation types, you need to extract possible relation triplets. 
Provide each relation triplet in the following format: (head entity, tail entity, relation type)
The results need to be returned in json format like: [[h1,t1,P1],[h2,t2,P2]...]
The pre-defined relation types: "{relation_types}".
###
Example Input:
"In this surgical image, the doctor is performing a delicate procedure involving the manipulation of kidney parenchyma."
Example Response:
```json
[
    ["doctor", "procedure", "P2"],
    ["procedure", "kidney parenchyma", "P5"]
]
###
Example Input:
"To the right, the monopolar curved scissors are actively engaged in cutting or dissecting the tissue."
Example Response:
```json
[
    ["monopolar curved scissors", "tissue", "P12"],
    ["monopolar curved scissors", "tissue", "P13"],
    ["monopolar curved scissors", "right", "P7"]
]
```
###
Input:
{document}
###
Response:
"""


def truncate_string(input_string):
    # 定义方向词列表
    direction_words = ['left', 'right', 'upper', 'lower', 'bottom', 'top', 'central']

    # 找到最后一个出现的方向词的索引
    last_index = -1
    last_word = None
    for word in direction_words:
        temp_index = input_string.rfind(word)
        if temp_index > last_index:
            last_index = temp_index
            last_word = word

    # 如果找到方向词，则截断字符串
    if last_index != -1:
        return input_string[:last_index].strip(' ') + ' ' + last_word

    # 如果没有找到方向词，返回原字符串
    return input_string


def llm_rte(
    candidates: List[str],
    max_tokens: int = 2048,
):
    # Compute the CLAIR score for a list of candidates and targets.

    candidate_statements = [f"- {c}\n" for c in candidates]

    with open('rel.json', 'r') as f:
        pre_rel = json.load(f)
        str_data = json.dumps(pre_rel, ensure_ascii=False)
    formatted_prompt = _REL_PROMPT.format(
        document="".join(candidates[0][0]),
        relation_types="".join(str_data),
    )

    messages = [{"role": "user", "content": formatted_prompt}]
    for _ in range(3):
        # Run the model
        logging.debug(f'RTE prompt: "{formatted_prompt}"')
        response = deepseek_client.chat.completions.create(
            model='deepseek-chat',
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.8,
            messages=messages
        )
        # response = client.chat.completions.create(
        #     model='gpt-4',
        #     max_tokens=max_tokens,
        #     temperature=0.0,
        #     top_p=0.8,
        #     messages=messages
        # )
        logging.debug(f'CLAIR response: "{response}"')

        # Parse the JSON response
        try:
            data = response.choices[0].message.content
            # 正则表达式，用来匹配元组
            result = ast.literal_eval(data.strip('` \njson'))
            for index, item in enumerate(result):
                if ':' in item[2]:
                    item[2] = item[2].split(':')[0]
                item[1] = truncate_string(item[1]).strip()
                result[index] = tuple(item)
            break
        except Exception as e:
            logging.warn(
                f"Could not parse response from LLM: {response}. Retrying"
            )
            continue
    else:
        logging.error("Could not parse response from CLAIR after 3 tries. Setting score to 0.")
        result = []

    return result


def load_git_data(gt_path='../gt/val.jsonl', pred_path='../pred/class_instrument.jsonl'):
    vid_gt = dict()
    vid_pred = dict()
    with open(gt_path, "r") as jsonl_file:
        for line in jsonl_file:
            try:
                # 解析JSON数据并将其添加到data列表中
                data_item = json.loads(line)
                if isinstance(data_item['text'], str):
                    # vid_gt[data_item['file_name']] = [data_item['text']]
                    vid_gt[data_item['file_name']] = [
                        data_item['text'].lower()]
                elif isinstance(data_item['text'], list):
                    # vid_gt[data_item['file_name']] = data_item['text']
                    vid_gt[data_item['file_name']] = [data_item['text'][0].lower()]
            except json.JSONDecodeError as e:
                # 处理JSON解析错误，如果发生错误的话
                print(f"Error parsing JSON: {str(e)}")
    with open(pred_path, "r") as jsonl_file:
        for line in jsonl_file:
            try:
                # 解析JSON数据并将其添加到data列表中
                data_item = json.loads(line)
                vid_pred[data_item['file_name']] = [data_item['text']]
            except json.JSONDecodeError as e:
                # 处理JSON解析错误，如果发生错误的话
                print(f"Error parsing JSON: {str(e)}")
    result = []
    for key in vid_gt.keys():
        result.append({
            "image_name": key,
            "prediction": vid_pred[key],
            "captions": vid_gt[key],
        })
    return result


def load_trancap_data(gt_path='../gt/val.jsonl', pred_path='../pred/best_predictions_epoch_26.json'):
    vid_gt = dict()
    vid_pred = dict()
    with open(gt_path, "r") as jsonl_file:
        for line in jsonl_file:
            try:
                # 解析JSON数据并将其添加到data列表中
                data_item = json.loads(line)
                if isinstance(data_item['text'], str):
                    # vid_gt[data_item['file_name']] = [data_item['text']]
                    vid_gt[data_item['file_name']] = [
                        data_item['text'].lower()]
                elif isinstance(data_item['text'], list):
                    # vid_gt[data_item['file_name']] = data_item['text']
                    vid_gt[data_item['file_name']] = [item.lower() for item in
                                                      data_item['text']]
            except json.JSONDecodeError as e:
                # 处理JSON解析错误，如果发生错误的话
                print(f"Error parsing JSON: {str(e)}")
    with open(pred_path, 'r',
              encoding='utf-8') as file:
        pred_data = json.load(file)
        for data_item in pred_data:
            vid_pred[data_item['image_id'] + '.png'] = [
                data_item['caption'].replace('\n', '').replace(' ,',
                                                               ',').replace(
                    ' .', '.')]
    result = []
    for key in vid_gt.keys():
        result.append({
            "image_name": key,
            "prediction": vid_pred[key],
            "captions": vid_gt[key],
        })
    return result


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)

    # 读取当前处理到的样本索引
    index_file = 'current_index.txt'
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            start_index = int(f.read().strip())
    else:
        start_index = 0

    # Example data processing
    # with open('../pred/coco_generated_captions_044.json') as f:
    #     data = json.load(f)

    data = load_git_data(pred_path='../pred/base.jsonl')
    # data = load_trancap_data(pred_path='../pred/best_predictions_epoch_26.json')
    # data = load_scracap_data(gt_path='../gt/val.jsonl',
    #                        pred_path='../pred/ablation.json')
    results_file = './new_prompt/rte_git3.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = []

    for i, sample in enumerate(data[start_index:], start=start_index):
        info = llm_rte([sample['prediction']])
        if not info:
            raise KeyError
        print(f"Processing sample {i+1}/{len(data)}: {sample['image_name']}")
        print(f"RTE: {info}")
        result = {
            "image": sample['image_name'],
            "prediction": sample['prediction'],
            "rte": info,
        }
        results.append(result)

        # 增量写入结果文件
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # 更新当前处理到的样本索引
        with open(index_file, 'w') as f:
            f.write(str(i + 1))

    logging.info("Processing completed.")
import json
import logging
import os
from typing import List, Optional, Tuple
from openai import OpenAI

deepseek_client = OpenAI(
    api_key='',
    base_url="https://api.deepseek.com",
)

client = OpenAI(
    api_key="",
)

_CLAIR_PROMPT = """\
You are trying to determine whether a candidate caption describes the same image as a reference caption. Please note:

You need to focus on the instruments, tissues, and actions described, as well as whether their positions are consistent.
You need to check whether the descriptions of the functions or operations of the instruments are consistent.
Even if the sequence of words and the form of description in the candidate caption are very similar to the reference caption, you should consider their descriptions inconsistent if the important information about instruments, tissues, etc., is different.
Candidate caption:
{candidate_statements}
Reference caption:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate caption is \
describing the same image as the reference caption? (JSON format, with a key "score", \
value between 0 and 100, and a key "reason" with a string value.)
"""

def clair(
    candidates: List[str],
    targets: List[str],
    max_tokens: int = 2048,
) -> Tuple[float, Optional[str]]:
    # Compute the CLAIR score for a list of candidates and targets.

    candidate_statements = [f"- {c}\n" for c in candidates]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIR_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )

    messages = [{"role": "user", "content": formatted_prompt}]
    score, reason = None, None
    for _ in range(3):
        # Run the model
        logging.debug(f'CLAIR prompt: "{formatted_prompt}"')
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
            json_data = json.loads(data.strip('` \njson'))
            score = float(json_data["score"])
            reason = json_data.get("reason", 'Unknown')
            break
        except (KeyError, ValueError):
            logging.warn(
                f"Could not parse response from CLAIR: {response}. Retrying"
            )
            continue
    else:
        logging.error("Could not parse response from CLAIR after 3 tries. Setting score to 0.")
        score = 0.0
        reason = None

    return score / 100, reason


def load_git_data(gt_path='gt/val.jsonl', pred_path='pred/class_instrument.jsonl'):
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


def load_qwen_data(gt_path='gt/val.jsonl', pred_path='pred/qwen2_vl_2B_lora.jsonl'):
    vid_gt = dict()
    vid_pred = dict()
    with open(gt_path, "r") as gt_file, open(pred_path, "r") as pred_file:
        for gt_line, pred_line in zip(gt_file, pred_file):
            try:
                # 解析gt文件中的JSON数据
                gt_item = json.loads(gt_line)
                pred_item = json.loads(pred_line)

                # 检查 gt_item['text'] 的类型并转换为小写
                if isinstance(gt_item['text'], str):
                    vid_gt[gt_item['file_name']] = [gt_item['text'].lower()]
                elif isinstance(gt_item['text'], list):
                    vid_gt[gt_item['file_name']] = [gt_item['text'][0].lower()]

                # 将预测结果添加到vid_pred
                vid_pred[gt_item['file_name']] = [pred_item['predict']]

            except json.JSONDecodeError as e:
                # 处理JSON解析错误
                print(f"Error parsing JSON: {str(e)}")

    result = []
    for key in vid_gt.keys():
        result.append({
            "image_name": key,
            "prediction": vid_pred[key],
            "captions": vid_gt[key],
        })
    return result


def load_trancap_data(gt_path='gt/val.jsonl', pred_path='pred/best_predictions_epoch_26.json'):
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


def load_scracap_data(gt_path='gt/val.jsonl', pred_path='pred/best.json'):
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
    with open(pred_path, 'r',
              encoding='utf-8') as file:
        pred_data = json.load(file)
        for data_item in pred_data:
            vid_pred[data_item['image_name']] = [data_item['prediction']]
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
    index_file = 'current_index_scracap.txt'
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            start_index = int(f.read().strip())
    else:
        start_index = 0

    # Example data processing
    # with open('pred/qwen2_vl_2B_lora.json') as f:
    #     data = json.load(f)

    # data = load_qwen_data(pred_path='pred/qwen2_vl_2B_lora.jsonl')
    # data = load_trancap_data(gt_path='gt/val.jsonl', pred_path='pred/best_predictions_epoch_26.json')
    data = load_scracap_data(gt_path='gt/val.jsonl', pred_path='pred/scracap.json')
    results_file = 'result/scracap.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
            total_score = sum(item['score'] for item in results)
    else:
        results = []
        total_score = 0.0

    for i, sample in enumerate(data[start_index:], start=start_index):
        score, reason = clair([sample['prediction']], [sample['captions']])
        print(f"Processing sample {i+1}/{len(data)}: {sample['image_name']}")
        print(f"Score: {score}, Reason: {reason}")
        result = {
            "image": sample['image_name'],
            "prediction": sample['prediction'],
            "captions": sample['captions'],
            "score": score,
            "reason": reason
        }
        results.append(result)
        total_score += score

        # 增量写入结果文件
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # 更新当前处理到的样本索引
        with open(index_file, 'w') as f:
            f.write(str(i + 1))

    average_score = total_score / len(data)
    print(f"Average Score: {average_score}")

    logging.info("Processing completed.")
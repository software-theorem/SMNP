import argparse
import csv
import json
import logging
import os
import re
import sys
import tokenize
from io import StringIO
import time
import csv

from util.remove_comments import remove_comments_and_docstrings
import openai
from numpy import mean
from model import GPT, Deepseek, deepseekcode

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer_name = RobertaTokenizer.from_pretrained(r"F:\fuwuqi\cot\codebert", local_files_only=True )

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def write_formatted_txt(results, language, eval_model):
    file_path = f'./test/{language}/{eval_model}-read.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as fw:
        for i, item in enumerate(results):
            formatted_str = f"-----------------样本：{i} -----------------\n"
            formatted_str += f"idx: {item['idx']}\n"
            formatted_str += f"code:\n{item['code']}\n"
            # 统一输出 reasons 字符串
            formatted_str += f"reasons:\n{item['reasons']}\n\n"

            formatted_str += "\n"
            fw.write(formatted_str)

def evaluate_all(input, len_names, model):
    # 构造 prompt
    prompt = (
            ) + input

    message = model.ask(prompt)
    TOTAL_FUNCTIONS = 

    # 先初始化
    method_names = ['N/A'] * TOTAL_FUNCTIONS
    c1_scores = [0] * TOTAL_FUNCTIONS
    c2_scores = [0] * TOTAL_FUNCTIONS

    # 提取函数名
    name_pattern = re.findall(r'[*#\s]*Function Name\s*(\d+)\s*:\s*([^\n*#]+)', message)

    for index_str, name in name_pattern:
        try:
            index = int(index_str) - 1  # 索引从0开始
            if 0 <= index < TOTAL_FUNCTIONS:
                method_names[index] = name.strip()
            else:
                logger.warning(f'Function index {index + 1} out of expected range')
        except ValueError:
            logger.error(f'Invalid function index: {index_str}')

    # 提取 C1/C2 分数（按顺序匹配）
    c_scores = re.findall(r'\*?\*?C1:\*?\*?\s*(\d+).*?\*?\*?C2:\*?\*?\s*(\d+)', message, re.DOTALL)

    for i, (c1, c2) in enumerate(c_scores):
        if i < TOTAL_FUNCTIONS:
            c1_scores[i] = int(c1)
            c2_scores[i] = int(c2)
        else:
            logger.warning(f'C1/C2 index {i + 1} out of expected range')

    return list(zip(method_names, c1_scores, c2_scores)), message

def evaluate(code, names, file_path, cnt=0, model=None):
    # 确保输出为 .jsonl
    if not file_path.endswith(".jsonl"):
        file_path = file_path.rsplit(".", 1)[0] + ".jsonl"

    mode = 'a' if cnt > 0 else 'w'

    # 准备 CSV 文件路径（同名 .csv）
    csv_path = file_path.replace(".jsonl", ".csv")
    write_csv_header = not os.path.exists(csv_path) or cnt == 0

    with open(file_path, mode, encoding="utf-8") as f_jsonl, \
         open(csv_path, mode, encoding="utf-8", newline='') as f_csv:

        csv_writer = csv.writer(f_csv)

        # 写入 CSV 表头
        if write_csv_header:
            csv_writer.writerow(['idx'] + [f'scores{j+1}' for j in range(len(names))])

        for i in range(len(code)):
            if i < cnt:
                continue

            # 构造 prompt 输入
            input = 'Code:\n' + code[i] + '\n'
            for j in range(len(names)):
                input += f'names {j}: {names[j][i]}\n'

            # 获取评分
            method_data, reasons = evaluate_all(input, len(names), model)
            method_names, scores_c1, scores_c2 = zip(*method_data)

            # 构建 JSONL 结果
            result = {
                "idx": i,
                "code": code[i],
                "reasons": reasons  # 如果没有 reasons 提取可以改成空串或注释掉
            }

            for j in range(len(method_names)):
                result[f"names[{j}]"] = method_names[j]
                result[f"scores_c1[{j}]"] = scores_c1[j]
                result[f"scores_c2[{j}]"] = scores_c2[j]

            # 写入 JSONL 文件
            f_jsonl.write(json.dumps(result, ensure_ascii=False) + '\n')

            # 写入 CSV 文件（列顺序：idx, c1_0, c2_0, c1_1, c2_1, ...）
            csv_row = [i]
            for j in range(len(method_names)):
                csv_row.extend([scores_c1[j], scores_c2[j]])
            csv_writer.writerow(csv_row)

            print(i)
def compare_human_eval_and_gpt_eval(language, eval_model='gpt-4',model=None):
    code = []
    gold = []
    idxs = []
    names = []
  
    # === 读取人工注释（gold） ===
    with open(r'', "r",
              encoding="utf-8") as fg:
        # for idx, line in enumerate(fg):
        #     if idx > 300:
        #         break
        for idx, line in enumerate(fg):
            if idx > max(target_idxs):  # 提前结束
                break
            line = line.strip()
            if idx not in target_idxs or not line:  # 不是目标行或空行则跳过
                continue
            try:
                js = json.loads(line.strip())
                name = js["gold_name"].replace("\n", " ").split('.')[-1].strip()
                raw_code = js["gold_code"]

                # 替换函数名，防止泄露
                code_without_name = raw_code.replace(name, " ")

                # 去除注释和 docstring
                try:
                    clean_code = remove_comments_and_docstrings(source=code_without_name, lang="java")
                except Exception as e:
                    print(f"[Warning] Failed to remove name for idx {idx}: {e}")
                    clean_code = code_without_name

                # 存储
                code.append(clean_code)
                gold.append(name)
                idxs.append(js["idx"])

            except Exception as e:
                print(f"[Error] Failed to process line {idx}: {e}")

    # 添加人工注释作为 names[0]
    names.append(gold)

    # === 读取模型生成的注释 ===
    # models = ['gpt-4', 'gpt-4-two', 'deepseek', 'deepseek-two', 'deepseekcode', 'deepseekcode-two']
    models = ['gpt-4', 'gpt-4-two']

    for llm in models:
        # 根据模型名判断一级目录名
        if llm.startswith("gpt-4"):
            group = "gpt-4"
        elif llm.startswith("deepseekcode"):
            group = "deepseekcode"
        elif llm.startswith("deepseek"):
            group = "deepseek"
        else:
            group = llm  # fallback

        # 构造当前模型的路径
        # pre_file_path = r''.format(language, group, llm)
        pre_file_path=r"".format(language, group, llm)
        output_map = {}
        try:
            with open(pre_file_path, "r", encoding="utf-8") as f:
                print(pre_file_path)
                for line in f:
                    js = json.loads(line.strip())
                    i = js["idx"]
                    if i in idxs:
                        try:
                            name = re.search(r"`([a-zA-Z0-9_]+)`", js["prompt2_answer"]).group(1)
                            output_map[i] = name
                        except Exception as e:
                            print(f"[Warning] Failed to parse name at idx {i}: {e}")
        except FileNotFoundError:
            print(f"[Error] File not found: {pre_file_path}")
            output_map = {}

        # 构造当前模型下的 name 列表（按 idxs 顺序）
        output = [output_map.get(i, "") for i in idxs]
        names.append(output)

    # === 检查结果 ===
    for i, names_list in enumerate(names):
        print(f"names[{i}] = {names_list}")

    # 可选：检查长度一致性
    assert all(len(c) == len(gold) for c in names), "Length mismatch in names!"

    evaluate(code=code, names=names, file_path=f'./test/{language}/{eval_model}.jsonl', cnt=0, model=model)

    file_path = f"./test/{language}/{eval_model}.jsonl"

    # 读取 JSONL 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line.strip()) for line in f]

    # 写入格式化 .txt 文件
    write_formatted_txt(results, language, eval_model)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--openai_key", default="", type=str)
    parser.add_argument("--deepseek_key", default="", type=str)
    parser.add_argument("--qwen_key", default="", type=str)
    parser.add_argument("--qwencode_key", default="", type=str)
    parser.add_argument("--max_new_tokens", default=300, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--log_filename", default='log-eval-llms.txt', type=str)
    parser.add_argument("--n_reasoning_paths", default=1, type=int)
    parser.add_argument("--frequency_penalty", default=0, type=int)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    args.logger = logging.getLogger(__name__)
    fh = logging.FileHandler(args.log_filename)
    args.logger.addHandler(fh)

    MODEL_NAME_OR_PATH = {'gpt-4': '',
                          'deepseek': '',
                          'deepseekcode': '',
'qwen': '',
'qwencode': '',
                          }
    args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]
    if args.model == 'gpt-4':
        model = GPT(args=args)
    elif args.model == 'deepseek':
        model = Deepseek(args=args)
    elif args.model == 'deepseekcode':
        model = deepseekcode(args=args)
    else:
        print('Model not found!')
        sys.exit(0)

    for language in ['java']:
        compare_human_eval_and_gpt_eval(language=language, eval_model=args.model, model=model)

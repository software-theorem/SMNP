import csv
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
# from model import gpt, StarChat, CodeLLAMA
from model import GPT, Deepseek, deepseekcode, qwencoder,qwen
from util.remove_comments import remove_comments_and_docstrings
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer_name = RobertaTokenizer.from_pretrained(r"F:\fuwuqi\cot\codebert")


def replace_first_func_name(raw_code, func_name):
    idx = raw_code.find(func_name)
    if idx == -1:
        return raw_code  # 函数名未找到，不替换
    return raw_code[:idx] + "XXXXX" + raw_code[idx + len(func_name):]
def generate_summaries_chain_of_thought(args, model, code, output_file, cnt=0):
    args.logger.info('chain of thought prompt...')
    prompt1 = \

    prompt2 = '''
                  '''

    # f = open(output_file, args.mode, encoding="utf-8", newline='')
    # writer = csv.writer(f)

    with open(args.summaries_file, 'w', encoding="utf-8") as f1, open(args.methodname_file, 'w',
                                                                      encoding="utf-8") as f2:
        for idx, c in tqdm(enumerate(code)):
            if idx < cnt:
                continue

            # 生成 prompt1 的回复
            comment = model.ask(input=prompt1.format(c))  # prompt1's answer

            if isinstance(comment, set):
                comment = list(comment)  # 确保 comment 是 list 格式

            # 保存 prompt1 结果到 summaries 文件
            json.dump({'idx': idx, 'prompt1_answer': comment}, f1, ensure_ascii=False)
            f1.write("\n")

            # 生成 prompt2 的回复，将 comment 作为输入
            message = model.ask(input=prompt2.format(comment), history=[(prompt1.format(c), comment)])

            if isinstance(message, set):
                message = list(message)

                # 保存 prompt2 结果到 methodname 文件
            json.dump({'idx': idx, 'prompt2_answer': message}, f2, ensure_ascii=False)
            f2.write("\n")

            print('current idx:', idx)


def write_ground_truth(gold, output_path):
    f = open(output_path, "w", encoding="utf-8", newline='')
    writer = csv.writer(f)
    cnt = 0
    for g in tqdm(gold):
        writer.writerow([cnt, g])
        cnt = cnt + 1
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", default="", type=str)
    parser.add_argument("--deepseek_key", default="", type=str)
    parser.add_argument("--qwen_key", default="", type=str)
    parser.add_argument("--qwencoder_key", default="", type=str)

    parser.add_argument("--data_file", default=r"", type=str)
    parser.add_argument("--language", default="", type=str)
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--cot", default="", type=str)  
    parser.add_argument("--count", default=0, type=int, help="continue from sample `count`")
    parser.add_argument("--n_reasoning_paths", default=1, type=int)
    parser.add_argument("--frequency_penalty", default=0, type=int)

    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--write_groundtruth", default=True, type=bool)
    parser.add_argument("--intent", default=False,
                        type=bool)  # True: use fewshot_example_intent, False: use fewshot_example_language_4
    parser.add_argument("--intent_type", default='', type=str)  # only used when intent is True

    parser.add_argument("--max_new_tokens", default=300, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--basic_prompt", default='Please generate an appropriate method name for the following function:\n', type=str)
    parser.add_argument("--log_filename", default='log-gpt-4.txt', type=str)

    args = parser.parse_args()

    # ouput directory
    if args.intent:
        dir = './result/{}/{}/{}/{}/'.format(args.intent_type, args.model, args.temperature, args.top_p)
        if os.path.exists(dir) == False:
            os.makedirs(dir)
    else:
        dir = './result/{}/{}/{}/'.format(args.language, args.model, args.cot)
        if os.path.exists(dir) == False:
            os.makedirs(dir)

    args.summaries_file=dir + "summaries.jsonl"
    args.methodname_file=dir + "methodname.jsonl"


    # logger
    log_file_path = os.path.join(os.path.join(dir, args.log_filename))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到标准输出，避免红色
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8")  # 记录到文件
        ]
    )

    args.logger = logging.getLogger(__name__)
    args.logger.info("Training/evaluation parameters %s", args)
    args.logger.info("\n")

    codes = []
    gold = []

    with open(args.data_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 50:  # 达到500行后停止读取
                break
            line = line.strip()
            if not line:  # 跳过空行
                continue

            js = json.loads(line)

            # 处理函数名：移除换行，取最后部分，清理空格
            func_name = js['func_name'].replace('\n', ' ').split('.')[-1].strip()

            # 处理代码：替换换行和函数名
            raw_code = js["code"]
            code_without_name = replace_first_func_name(raw_code, func_name)
            try:
                clean_code = remove_comments_and_docstrings(source=code_without_name, lang="python")
            except Exception as e:
                print(f"[Warning] Failed to remove comments for idx {idx}: {e}")
                clean_code = code_without_name  # Fall back to original code if comment removal fails
            codes.append(clean_code)

            # 处理目标名称：小写化并移除末尾数字
            gold_name = func_name.lower().rstrip('0123456789')
            gold_tokens = tokenizer_name.tokenize(gold_name)
            gold_str = ' '.join(gold_tokens).replace('\n', '')

            # 去除首尾空格并处理多余空格
            gold_str = ' '.join(gold_str.strip().split())

            # 最终将 gold 转换为没有空格的字符串
            result = ''.join(gold_str.split())  # 这里将 gold 字符串的多余空格去掉
            gold.append(result)  # 把最终的结果追加到 gold 列表中

        # load model
        MODEL_NAME_OR_PATH = {'gpt-4': '',
                              'deepseek': '',
                              'deepseekcode': '',
                              'qwen': '',
                              'qwencode': ''
                              }
        args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]
        if args.model == 'gpt-4':
            model = GPT(args=args)
        elif args.model == 'deepseek':
            model = Deepseek(args=args)
        elif args.model == 'deepseekcode':
            model = deepseekcode(args=args)
        elif args.model == 'qwencode':
            model = qwencoder(args=args)
        elif args.model == 'qwen':
            model = qwen(args=args)
        else:
            print('Model not found!')
            sys.exit(0)

        # write ground truth
        if args.write_groundtruth:
            if args.intent:
                write_ground_truth(gold, './result/{}/groundtruth.csv'.format(args.intent_type))
            else:
                write_ground_truth(gold, './result/{}/{}/{}/groundtruth.csv'.format(args.language, args.model, args.cot))

        if args.count > 0:
            args.mode = 'a'
        else:
            args.mode = 'w'

        generate_summaries_chain_of_thought(args, model, codes, args.count)

if __name__ == '__main__':
        main()
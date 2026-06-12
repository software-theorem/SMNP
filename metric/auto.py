# -*- coding: utf-8 -*-
"""
Method Name Evaluation (直接在代码里改配置即可运行)

功能：
- 读取 .jsonl（每行一个样本 dict）
- 自动抓取候选键（默认为：任意以 *_name_one / *_name_two 结尾，且不是 gold_name）
- 评估指标：Accuracy（不区分大小写）、基于“去重 token”的 Precision/Recall/F1、BLEU-4（含平滑）、ROUGE-L（F1）
- 输出：逐样本逐候选明细表、候选键/模型汇总表（按 candidate_key 分组求均值）
- 导出 CSV 与 Excel

使用方式：
1) 仅修改“CONFIG（需要你改）”的路径等，然后直接运行本文件。
2) 如果要改变候选键识别规则，改 KEY_SUFFIXES 或 REGEX_INCLUDE / REGEX_EXCLUDE 即可。
"""

# =======================
# CONFIG（需要你改）
# =======================
JSONL_PATH = r"F:\fuwuqi\newcot\extra_mn\java_mn.jsonl"      # 你的 .jsonl 文件路径
OUT_CSV_DETAILED = r"F:\fuwuqi\newcot\cal_metric\auto/results_detailed.csv"
OUT_CSV_SUMMARY  = r"F:\fuwuqi\newcot\cal_metric\auto/results_summary.csv"
KEY_SUFFIXES = ("name_one", "name_two")  # 候选键后缀规则
ENABLE_PREFIX_SUMMARY = True             # True: 把 *_one/_two 合并汇总为同一个前缀
# =======================

import json, math, re, os
import pandas as pd
from itertools import chain
from typing import List, Dict, Tuple

# -------------------------------
# 工具
# -------------------------------
def split_camel_case(s: str) -> List[str]:
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Za-z])(\d)', r'\1 \2', s)
    s = re.sub(r'(\d)([A-Za-z])', r'\1 \2', s)
    return s.split()

def name_tokens(name: str) -> List[str]:
    if name is None: return []
    s = re.sub(r'[^A-Za-z0-9]+', ' ', name.strip())
    cc = split_camel_case(s)
    toks = list(chain.from_iterable(t.lower().split() for t in cc))
    return [t for t in toks if t]

def precision_recall_f1(ref_tokens: List[str], cand_tokens: List[str]) -> Tuple[float,float,float]:
    ref_set, cand_set = set(ref_tokens), set(cand_tokens)
    if not cand_set and not ref_set: return 1,1,1
    if not cand_set: return 0,0,0
    inter = len(ref_set & cand_set)
    p = inter/len(cand_set) if cand_set else 0
    r = inter/len(ref_set) if ref_set else 0
    f1 = (2*p*r/(p+r)) if (p+r)>0 else 0
    return p,r,f1

def ngrams(tokens: List[str], n:int): return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_score(ref_tokens: List[str], cand_tokens: List[str], max_n:int=4)->float:
    if not cand_tokens: return 0
    weights=[1/max_n]*max_n; precisions=[]
    for n in range(1,max_n+1):
        ref_counts={}
        for g in ngrams(ref_tokens,n): ref_counts[g]=ref_counts.get(g,0)+1
        cand_counts={}
        for g in ngrams(cand_tokens,n): cand_counts[g]=cand_counts.get(g,0)+1
        match=sum(min(c,ref_counts.get(g,0)) for g,c in cand_counts.items())
        precisions.append((match+1)/(len(cand_counts)+1))
    log_p=sum(w*math.log(p) for w,p in zip(weights,precisions))
    bp=1.0 if len(cand_tokens)>len(ref_tokens) else math.exp(1-len(ref_tokens)/len(cand_tokens)) if cand_tokens else 0
    return bp*math.exp(log_p)

def lcs_length(x,y):
    m,n=len(x),len(y)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1]=dp[i][j]+1 if x[i]==y[j] else max(dp[i][j+1],dp[i+1][j])
    return dp[m][n]

def rouge_l_f1(ref_tokens,cand_tokens):
    if not ref_tokens and not cand_tokens: return 1
    if not ref_tokens or not cand_tokens: return 0
    l=lcs_length(ref_tokens,cand_tokens)
    r=l/len(ref_tokens); p=l/len(cand_tokens)
    return 2*p*r/(p+r) if (p+r)>0 else 0

def accuracy(ref,cand): return 1.0 if (ref or "").lower()==(cand or "").lower() else 0.0

# -------------------------------
# 主流程
# -------------------------------
def is_candidate_key(key:str)->bool:
    kl=key.lower()
    if kl=="gold_name": return False
    return any(kl.endswith(suf) for suf in KEY_SUFFIXES)

def normalize_prefix(key:str)->str:
    for suf in ("_name_one","_name_two"):
        if key.endswith(suf): return key[:-len(suf)]+"_name"
    return key

def eval_one_pair(ref,cand)->Dict[str,float]:
    ref_toks, cand_toks=name_tokens(ref), name_tokens(cand)
    p,r,f1=precision_recall_f1(ref_toks,cand_toks)
    return dict(Precision=p,Recall=r,F1=f1,
                BLEU4=bleu_score(ref_toks,cand_toks),
                ROUGE_L_F1=rouge_l_f1(ref_toks,cand_toks),
                Accuracy=accuracy(ref,cand))

def evaluate_jsonl(path:str)->pd.DataFrame:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            s=json.loads(line); ref=s.get("gold_name","")
            for k,v in s.items():
                if is_candidate_key(k):
                    m=eval_one_pair(ref,v)
                    rows.append(dict(sample=s.get("sample",""),idx=s.get("idx",""),
                                     gold_name=ref,candidate_key=k,candidate_name=v,**m))
    df=pd.DataFrame(rows)
    return df.sort_values(by=["idx","F1","BLEU4","ROUGE_L_F1"],ascending=[True,False,False,False]).reset_index(drop=True)

def summarize_by_key(df:pd.DataFrame)->pd.DataFrame:
    df2=df.copy()
    df2["key_prefix"]=df2["candidate_key"].apply(normalize_prefix if ENABLE_PREFIX_SUMMARY else lambda x:x)
    agg=["Precision","Recall","F1","BLEU4","ROUGE_L_F1","Accuracy"]
    summary=df2.groupby("key_prefix",as_index=False)[agg].mean()
    summary["Count"]=df2.groupby("key_prefix")["candidate_name"].count().values
    return summary.sort_values(by=["F1","BLEU4","ROUGE_L_F1"],ascending=False)

def save_outputs(detailed,summary,csv_detailed,csv_summary):
    os.makedirs(os.path.dirname(csv_detailed),exist_ok=True)
    os.makedirs(os.path.dirname(csv_summary),exist_ok=True)
    detailed.to_csv(csv_detailed,index=False,encoding="utf-8-sig")
    summary.to_csv(csv_summary,index=False,encoding="utf-8-sig")

# -------------------------------
# 执行
# -------------------------------
if __name__=="__main__":
    df_detail=evaluate_jsonl(JSONL_PATH)
    df_summary=summarize_by_key(df_detail)

    print("\n=== 明细（前 20 行） ===")
    print(df_detail.head(20))
    print("\n=== 汇总 ===")
    print(df_summary)

    save_outputs(df_detail,df_summary,OUT_CSV_DETAILED,OUT_CSV_SUMMARY)
    print(f"\n已保存:\n - 明细: {OUT_CSV_DETAILED}\n - 汇总: {OUT_CSV_SUMMARY}")
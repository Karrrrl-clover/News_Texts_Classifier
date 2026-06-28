"""
data_pre - 数据预处理

顺序：① 正则表达式剔除无效字符 - ② 中文分词 - ③ 去掉停用词

Author: 骆昊
Version: 0.0.1
"""
import logging
import re

import jieba
import pandas as pd

from src.config import Config

jieba.setLogLevel(logging.ERROR)

# 清洗文本的正则表达式
SPACE_PATTERN = re.compile(r'[\s\u3000]+')
BRACE_PATTERN = re.compile(r'\[.*?\]|\(.*?\)|【.*?】|（.*?）|「.*?」')
ATUSR_PATTERN = re.compile(r'@\w+')
TOPIC_PATTERN = re.compile(r'#.*?#')
MYURL_PATTERN = re.compile(r'https?://[^\s\u4e00-\u9fa5]+')
CHENN_PATTERN = re.compile(r'[^a-zA-Z0-9\s\u4e00-\u9fa5:,.?!;：，。？！；]+')
NOENN_PATTERN = re.compile(r'[^\s\u4e00-\u9fa5：，。？！；]+')

# 加载停用词表
with open(Config.stopwords_file, encoding='utf-8') as file_object:
    STOP_WORDS = set(file_object.read().splitlines())


def clean_raw_text(text: str, allow_eng_num: bool=True) -> str:
    """清理原始文本内容"""
    text = re.sub(SPACE_PATTERN, ' ', text)
    text = re.sub(BRACE_PATTERN, '', text)
    text = re.sub(ATUSR_PATTERN, '', text)
    text = re.sub(TOPIC_PATTERN, '', text)
    text = re.sub(MYURL_PATTERN, '', text)

    if allow_eng_num:
        text = re.sub(CHENN_PATTERN, '', text)
    else:
        text = re.sub(NOENN_PATTERN, '', text)

    text = re.sub(SPACE_PATTERN, ' ', text)
    return text.strip()


def cut_zh_words(text: str) -> str:
    """中文句子的清理和分词"""
    text = clean_raw_text(text)
    words = jieba.lcut(text)
    results = [word for word in words
               if word.strip() and word not in STOP_WORDS]

    if len(results) > 1:
        return ' '.join(results)

    return ''


def rf_clean_data(read_load,save_load):
    """程序入口"""
    df = pd.read_csv(read_load, sep='\t', names=['text', 'label'])
    df['text'] = df.text.map(cut_zh_words)
    df.query('text != ""').to_csv(save_load, index=False,header=False)



def ft_clean_data(read_load,save_load):
    df = pd.read_csv(read_load, sep='\t', names=['text', 'label'])
    df['text'] = df.text.map(cut_zh_words)
    df = df.query('text != ""')
    df['label'] = df.label.map(lambda x: f'__label__{x}')
    df['result'] = df.label + ' ' + df.text
    df['result'].to_csv(save_load, index=False, header=False)

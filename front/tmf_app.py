"""
tmf_app - Streamlit 用户界面

Author: KClover
Version: 0.0.1
"""
import sys
import pandas as pd
from pathlib import Path
import streamlit as st
import requests
from loguru import logger
from sympy import false

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8080"

logger.remove()
logger.add(sys.stderr, level='INFO')
logger.add(
    BASE_DIR / 'logs/api_service.log',
    rotation='100 MB',   # 何时关闭当前日志文件并创建新文件
    retention='7 days',  # 旧日志文件的回收与清理策略
    serialize=True,      # 是否将日志转换为 JSON 字符串输出
    level='INFO',        # 日志处理器接收的最低日志级别
    enqueue=True,        # 日志打印非阻塞
    catch=True,          # 防止日志写入失败导致应用崩溃
    compression='zip',   # 当日志触发切分时旧的日志文件会被自动压缩
)


def get_class_label(text, model):
    """获取标签类别"""
    try:
        resp = requests.get(
            url=BASE_URL + f"/api/{model}/predict",
            headers={'Content-Type': 'application/json'},
            json={'text':text},
        )
        result = resp.json()
        print(result)
        if result['code'] == 0:
            return result['label']
        return result['message']
    except Exception as e:
        logger.error(str(e))
        logger.critical(str(e))
        return ''



st.title('文本分类专家')
label2path = {'RandomForest': 'v1', 'Fasttext': 'v2', 'Bert_zh_Chinese': 'v3', 'Deepseek': 'v4'}
select_model = st.radio('请选择使用推理的模型',['RandomForest','Fasttext','Bert_zh_Chinese','Deepseek'])
content = st.text_input(label = '',placeholder='请输入文本内容')
ok_button = st.button('确定')

upload_file = st.file_uploader('请上传文件')
predict = []
if upload_file is not None:
    df = pd.read_csv(upload_file, sep='\t', names=['text', 'label'])
    for sentence in df.text.values:
        predict.append(get_class_label(sentence,model=label2path[select_model]))
    df['result'] = pd.DataFrame({'result': predict})
    st.dataframe(df)

if ok_button and content.strip() and select_model:
    class_label = get_class_label(content,model=label2path[select_model])
    st.write(f'**分类结果**：{class_label}')
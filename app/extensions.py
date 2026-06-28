"""
extensions - 扩展功能

Author: 骆昊
Version: 0.0.1
"""
import joblib
import torch
import fasttext
from src.config import Config
from modelscope import AutoTokenizer,AutoModelForSequenceClassification
from app.config import BaseConfig

class TextClassifierExtension:
    """文本分类器扩展"""

    def __init__(self):
        self.class_labels = None

        self.text_clf_model1 = None             #随机森林模型

        self.text_clf_model2 = None             #festtext模型

        self.text_clf_model3 = None             #bert微调模型
        self.tokenizer = None
        self.device= torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

    def init_app(self, model_path: str):
        print('===== [extensions] 正在加载文本分类模型 =====')
        if not self.text_clf_model1:
            self.text_clf_model1 = joblib.load(BaseConfig.MODEL_v1_PATH)
            self.class_labels = open(Config.class_file, encoding='utf-8').read().strip().splitlines()

        if not self.text_clf_model2:
            self.text_clf_model2 = fasttext.load_model(str(BaseConfig.MODEL_v2_PATH))
            self.class_labels = open(Config.class_file, encoding='utf-8').read().strip().splitlines()

        if not self.text_clf_model3:
            self.tokenizer = AutoTokenizer.from_pretrained(BaseConfig.MODEL_v3_PATH)
            self.text_clf_model3 = AutoModelForSequenceClassification(BaseConfig.MODEL_v3_PATH)
            self.class_labels = open(Config.class_file, encoding='utf-8').read().strip().splitlines()


        print('===== [extensions] 文本分类模型加载完成 =====')


thy_extension = TextClassifierExtension()

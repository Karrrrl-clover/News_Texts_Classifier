"""
model_eval - 模型评估

Author: 骆昊
Version: 0.0.1
"""
import time
import joblib
import pandas as pd
import torch
import numpy as np
import fasttext
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.config import Config
from modelscope import AutoTokenizer,AutoModelForSequenceClassification

def rf_evaluate_model(read_file,model_file):
    df = pd.read_csv(read_file)
    pl = joblib.load(model_file)  # type: Pipeline

    total_accuracy = 0.0
    total_duration = 0.0
    for _ in range(100):
        temp = df.sample(n=10)
        X_test, y_test = temp.text.values, temp.label.values
        start = time.perf_counter()
        y_pred = pl.predict(X_test)
        end = time.perf_counter()
        total_accuracy += accuracy_score(y_test, y_pred)
        total_duration += end - start

    print(f'Accuracy: {total_accuracy / 100:.2%}')
    print(f'Duration: {total_duration / 100:.3f}')


def ft_evaluate_model(read_file,model_file):
    df = pd.read_csv(Config.test_pre_file)
    clf = fasttext.load_model(str(Config.ftz_model_file))

    total_accuracy = 0.0
    total_duration = 0.0
    for _ in range(100):
        temp = df.sample(n=100)
        X_test, y_test = temp.text.values, temp.label.values
        start = time.perf_counter()
        labels, _ = clf.predict(X_test.tolist())
        end = time.perf_counter()
        y_pred = [int(label[-1]) for label in np.ravel(labels)]
        total_accuracy += accuracy_score(y_test, y_pred)
        total_duration += end - start

    print(f'Accuracy: {total_accuracy / 100:.2%}')
    print(f'Duration: {total_duration / 100:.3f}')




def bert_evaluate_model(read_file,model_file):
    df = pd.read_csv(read_file, sep='\t', names=['text', 'label'])
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = AutoModelForSequenceClassification.from_pretrained(model_file)

    accuracy = 0.0
    for _ in range(1000):
        temp = df.sample(n=10)
        inputs = tokenizer(temp.text.tolist(), return_tensors='pt', truncation=True, padding=True, max_length=32)

        with torch.inference_mode():
            y_hats = model(**inputs)
            y_pred = y_hats.logits.argmax(dim=-1)

            accuracy += accuracy_score(temp.label.values, y_pred.numpy())

    print(f'模型的准确率为：{accuracy/1000:.4f}')

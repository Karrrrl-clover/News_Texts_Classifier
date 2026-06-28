"""
model_train - 训练模型

Author: 骆昊
Version: 0.0.1
"""
import joblib
import torch
import pandas as pd
import fasttext
import torch.optim as optim
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset,DataLoader
from modelscope import AutoTokenizer,AutoModelForSequenceClassification
from src.config import Config
# from skl2onnx import to_onnx



def rf_train_model():
    """模型训练"""
    # 加载数据
    df = pd.read_csv(Config.train_pre_file)
    X_train, y_train = df.text.values, df.label.values
    df = pd.read_csv(Config.valid_pre_file)
    X_valid, y_valid = df.text.values, df.label.values

    # 随机森林模型参数（控制预剪枝）
    rf_params = {
        'n_estimators': 128,
        'min_samples_split': 16,
        'max_features': 'log2',
        'n_jobs': -1
    }

    # 构造流水线
    pl = Pipeline(steps=[
        # TfidfVectorizer ---> 将分词后的中文文本处理成稀疏向量（矩阵）- SparseMatrix
        ('vec', TfidfVectorizer(ngram_range=(1, 2), min_df=0.0001, max_df=0.99)),
        # SelectKBest ---> 特征选择 ---> 方差分析（最大化组间方差最小化组内方差）
        ('sel', SelectKBest(k=8192)),
        # RandomForestClassifier ---> 集成学习（Bagging）
        ('clf', RandomForestClassifier(**rf_params)),
    ])
    # 喂入数据
    pl.fit(X_train, y_train)
    # 验证效果
    y_pred = pl.predict(X_valid)

    # 模型评估 ---> 分类模型 ---> Accuracy / Precision / Recall / F1 Score / AUC
    print(confusion_matrix(y_valid, y_pred))
    print(classification_report(y_valid, y_pred))

    # 保存模型（序列化）
    joblib.dump(pl, Config.pkl_model_file, compress=3)

    # # 将模型保存为 ONNX - Open Neural Network eXchange
    # # 需要安装 skl2onnx - conda install skl2onnx -c conda-forge
    # initial_type = [('text_input', ['str', None, 1])]
    # onnx_pipeline = to_onnx(pl, initial_types=initial_type)
    # with open(Config.onnx_model_file, 'wb') as file_obj:
    #     file_obj.write(onnx_pipeline.SerializeToString())



def fasttest_train_model():
    """训练FestText模型"""
    clf = fasttext.train_supervised(
        input=str(Config.fasttext_trian_file),
        autotuneValidationFile=str(Config.fasttext_test_file),
        autotuneMetric='f1',
        autotuneModelSize='50M',
        autotuneDuration=300
    )

    clf.save_model(str(Config.ftz_model_file))


class TMFDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_len=32):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        doc, label = self.corpus[index]
        inputs = self.tokenizer(
            doc,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_len
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
def bert_trian_model():
    """Bert预训练模型微调"""
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    corpus = []
    with open(Config.train_raw_file, encoding='utf-8') as file_obj:
        while line := file_obj.readline():
            line = line.strip()
            if line:
                doc, label = line.split('\t', maxsplit=1)
                corpus.append((doc, int(label)))

    model_name = 'google-bert/bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TMFDataset(corpus, tokenizer, max_len=32)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            loss = loss_func(outputs.logits, batch['label'].to(device))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch[{epoch:>2d}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}')
    model.save_model_pretrained(str(Config.bert_model_file))
    tokenizer.save_model_pretrained(str(Config.bert_model_file))

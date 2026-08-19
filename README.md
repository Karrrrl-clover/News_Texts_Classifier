# 项目说明
基于传统机器学习FastText，RandomForest，和Bert模型，大模型API设计的一个多模型文本分类系统

V1：基于RandomForest
V2：基于FastText
V3：基于Bert模型
V4：直接使用通用大模型API+prompts

## 注意事项

### 克隆项目

```bash
git clone https://gitee.com/jackfrued/tmf_v1.git
```

### 还原环境

Pip:

```bash
pip install -r requirements.txt
```

### 测试用例

```bash
pytest tests/ -v
pytest tests/ -v -m api
```

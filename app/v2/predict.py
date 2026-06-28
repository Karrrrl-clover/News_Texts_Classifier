"""
项目:my_tmf
文件:predict.py
IDE:PyCharm
时间:2026/6/28 17:55
作者:Kclover

描述:
    TODO:
"""
from flask import Blueprint, request
from app.extensions import thy_extension
from src.data_pre import cut_zh_words

# 创建蓝图对象
model_bp_v2 = Blueprint('models', __name__, url_prefix='/api/v2')


# @model_bp.before_request
# def before_request():
#     """每个请求进来时把启动时加载好的模型单例安全地绑定到当前请求的 g 对象上"""
#     g.model = thy_extension.text_clf_model
#     g.class_labels = thy_extension.class_labels


@model_bp_v2.route('/predict', methods=['GET', 'POST'])
def text_clf_predict():
    """文本分类预测"""
    # silent=True - 如果解析 JSON 失败不抛出异常而是返回 None
    # force=True - 强制将消息体解析为 JSON 格式
    json_string = request.get_json(silent=True, force=True)
    if not json_string:
        return {'code': -20, 'message': '请求体不符合 JSON 格式规范'}

    text = json_string.get('text', '')
    if not text:
        return {'code': -10, 'message': '请提供要分类的文本内容'}

    text = cut_zh_words(text)
    y_pred,_= thy_extension.text_clf_model.predict([text])
    label_index = int(y_pred[0][0][9:])
    return {'code': 0, 'message': 'OK', 'label': thy_extension.class_labels[y_pred[0]]}
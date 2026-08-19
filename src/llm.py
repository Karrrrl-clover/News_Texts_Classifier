"""
项目:my_tmf
文件:llm.py
IDE:PyCharm
时间:2026/6/28 18:12
作者:KClover

描述:
    TODO:
"""
import json
import requests


def classify_text(text_to_classify: str):
    system_prompt = '''
    你是一个文本分类专家，根据用户输入，从以下10个类别中选择唯一一个：
    【finance, realty, stocks, education, science, society, politics, sports, game, entertainment】

    规则：
    1. 仅能选择上述类别之一。
    2. 可参考示例推理。
    3. 若输入不属于任何类别或模糊，回答“不认识（不在预设分类之中）”。
    4. 只输出类别名称，不附加任何解释。

    示例：1
    用户输入：近几年来，日元持续贬值，已经严重影响到了日本民众的生活 
    回答： finance
    
    示例：2
    用户输入：今年中高考圆满结束，各位广大考生妙笔生花，蟾宫折桂 
    回答： education
    
    示例：3
    文本：著名歌手周杰伦演唱会现场，热闹非凡，观众情绪高涨 
    回答： entertainment
    
    示例：4
    用户输入：昨天世界杯，法国对阵挪威，上演帽子戏法，最终法国4:1挪威 
    回答： sports
    
    示例：5
    用户输入：你好，早上好 
    回答：Unknown
    '''

    # API接口的地址
    url = 'https://api.deepseek.com/chat/completions'
    # 构造HTTP请求头
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer sk-e8693ab838fa4e7cbda6b42024538b1a'
    }
    # 构造HTTP请求消息体
    data = {
        'model': 'deepseek-v4-flash',
        'max_tokens': 256,
        'temperature': 0.2,
        'stream': False,
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': text_to_classify
            }
        ]
    }
    # 向API接口发起POST请求
    resp = requests.post(
        url=url,
        headers=headers,
        data=json.dumps(data)
    )
    if resp.status_code == 200:
        # 解析JSON数据获取API接口响应内容
        return resp.json()['choices'][0]['message']['content']

    else:
        return resp.status_code

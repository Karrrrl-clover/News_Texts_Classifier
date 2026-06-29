"""
test_api - API 接口测试

测试用例 - test_* - 单元测试（测试最小执行单元 --- 函数）

conda install pytest -c conda-forge

Author: 骆昊
Version: 0.0.1
"""
import pytest
import requests

BASE_URL = "http://127.0.0.1:8080"


@pytest.mark.smoke
def test_base_url():
    """测试基路径"""
    resp = requests.get(BASE_URL)
    assert resp.status_code == 200


@pytest.mark.api
def test_predict_valid():
    """测试 /predict 接口"""
    resp = requests.post(
        url=BASE_URL + '/api/v3/predict',
        headers={
            'Content-Type': 'application/json',
        },
        json={
            'text': '湖北省黄冈市09届高三年级期末考试试题',
        },
    )

    assert resp.status_code == 200
    result = resp.json()  # type: dict
    assert 'label' in result
    assert result.get('code') == 0 and result.get('message') == 'OK'


@pytest.mark.api
def test_predict_invalid():
    """测试 /predict 接口"""
    resp = requests.post(
        url=BASE_URL + '/api/v3/predict',
        headers={
            'Content-Type': 'application/json',
        },
        json={},
    )
    assert resp.status_code == 200
    result = resp.json()  # type: dict
    assert 'label' not in result
    assert result.get('code') == -20 and result.get('message') != 'OK'

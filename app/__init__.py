"""
Flask 应用工厂与核心生命周期管理模块

本模块负责整个 Flask 应用程序的初始化、配置加载、第三方扩展组件绑定以及业务蓝图的动态注册。
采用工厂模式（Application Factory Pattern）设计，以支持多环境隔离部署并有效规避循环导入问题。

主要职责包括:
1. 环境配置分发: 依据环境变量 `FLASK_ENV` 动态加载 `app.config` 映射类。
2. 扩展组件装配: 初始化并注入数据库 (SQLAlchemy)、缓存 (Redis) 等基础中间件。
3. 路由蓝图注册: 统一挂载各版本 API 业务流（如预测推理模块）。
4. 全局拦截兜底: 注册全局异常捕获器 (Error Handlers)，确保所有错误输出标准化为 JSON 响应。

使用示例:
    >>> from app import create_app
    >>>
    >>> app = create_app()

Author: KClover
Version: 0.0.1
"""
import os
from flask import request
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException
from app.config import config_map

from app.v1.predict import model_bp_v1
from app.v2.predict import model_bp_v2
from app.v3.predict import model_bp_v3
from app.v4.predict import model_bp_v4

from app.extensions import thy_extension


def create_app() -> Flask:
    """应用工厂函数"""
    app = Flask(__name__)

    # 动态加载环境配置
    env = os.getenv('FLASK_ENV', 'production')
    app.config.from_object(config_map[env]())

    # print(app.config)

    # 将组件动态绑定到当前 app
    # db.init_app(app)
    # redis_client.init_app(app)

    thy_extension.init_app(rf = app.config['MODEL_V1_PATH'],ft = app.config['MODEL_V2_PATH'],bert = app.config['MODEL_V3_PATH'])

    # 注册业务蓝图
    app.register_blueprint(model_bp_v1)
    app.register_blueprint(model_bp_v2)
    app.register_blueprint(model_bp_v3)
    app.register_blueprint(model_bp_v4)

    # 注册钩子函数
    register_handlers(app)

    return app


def register_handlers(app: Flask):
    """注册拦截钩子函数"""

    @app.before_request
    def check_auth():
        white_list = []


        if request.endpoint not in white_list:
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'code': 401, 'msg': '请提供身份认证信息'}), 401
        return None

    @app.after_request
    def add_cors_headers(response):
        # 动态修改 HTTP 头，解决跨域问题
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        # 拦截标准 HTTP 错误（如：4xx 错误）
        if isinstance(e, HTTPException):
            return jsonify({'code': e.code, 'message': e.description}), e.code

        # 拦截业务层未捕获的异常
        return jsonify({'code': 500, 'message': '服务器维护中请稍后尝试'}), 500

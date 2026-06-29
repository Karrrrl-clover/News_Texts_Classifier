"""
config - 全局配置项

Author: 骆昊
Version: 0.0.1
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class BaseConfig:
    """基础配置类（所有环境共享的公共配置）"""

    # 核心安全密匙（生产环境必须在 .env 中覆盖它）
    # 可以通过 openssl rand -hex 32
    SECRET_KEY = os.getenv('SECRET_KEY', '82ccc7debfad504e2e026c0c14e8721e17f9c4d884bfb94e32247a0cc3abf5e2')

    # 模型持久化文件路径
    MODEL_V1_PATH = BASE_DIR / 'models' / 'text-clf-model.pkl'
    MODEL_V2_PATH = BASE_DIR / 'models' / 'text-clf-model.ftz'
    MODEL_V3_PATH = BASE_DIR / 'models' / 'bert_models'



class DevelopmentConfig(BaseConfig):
    """开发环境配置"""

    DEBUG = True

    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{BASE_DIR / "app_dev.db"}'
    )


class TestingConfig(BaseConfig):
    """测试环境配置"""
    pass



class ProductionConfig(BaseConfig):
    """生产环境配置"""

    DEBUG = False

    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

    # 生产环境中间件配置
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'max_overflow': 20
    }



# 不同环境对应的配置类
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

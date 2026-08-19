"""
run_pipeline - 离线训练流水线

Author: KClover
Version: 0.0.1
"""
from src.config import Config
from src.data_pre import rf_clean_data, ft_clean_data
from src.model_eval import rf_evaluate_model,ft_evaluate_model,bert_evaluate_model
from src.model_train import rf_train_model, fasttest_train_model, bert_trian_model


def main():
    """离线训练流水线程序入口"""
    print("\n=== [Step 1/3] 开始执行数据清洗和准备工作 ===")
    rf_clean_data(Config.train_raw_file,Config.train_pre_file)      #randomforest训练集
    rf_clean_data(Config.test_raw_file,Config.train_pre_file)       #randomforest测试集
    rf_clean_data(Config.valid_raw_file,Config.train_pre_file)      #randomforest预测集

    ft_clean_data(Config.train_raw_file,Config.fasttext_trian_file) #fastext训练集
    ft_clean_data(Config.test_raw_file,Config.fasttext_test_file)   #fasttext测试集


    print("\n=== [Step 2/3] 开始训练和导出文本分类模型 ===")
    rf_train_model()
    fasttest_train_model()
    bert_trian_model()

    print("\n=== [Step 3/3] 启动服务之前对模型进行评估 ===")
    rf_evaluate_model(Config.valid_pre_file,Config.pkl_model_file)

    ft_evaluate_model(Config.valid_raw_file,Config.ftz_model_file)

    bert_evaluate_model(Config.valid_raw_file,Config.bert_models)


if __name__ == "__main__":
    main()

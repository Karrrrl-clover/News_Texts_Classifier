"""
data_eda - 探索性数据分析

pip list --format=freeze > requirements.txt
pip install -r requirements.txt

Author: KClover
Version: 0.0.1
"""
import pandas as pd
import matplotlib.pyplot as plt

from config import Config

df = pd.read_csv(Config.train_raw_file, sep='\t', names=['text', 'label'])
df.info()

print(df.label.value_counts())
print(df.label.value_counts(normalize=True))

df['length'] = df.text.map(len)
print(f'Min text length: {df.length.min()}')
print(f'Max text length: {df.length.max()}')
print(f'Mean text length: {df.length.mean():.2f}')
print(f'Std text length: {df.length.std():.4f}')

df.length.plot(
    kind='box',
    showmeans=True,
)
plt.show()

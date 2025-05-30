import numpy as np
import pandas as pd
import tensorflow as tf
from new_transformer_feature_extractor import TransformerFeatureExtractor
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
print("加载预处理数据...")
X = np.load('processed_data/X_test_smote.npy')
y = np.load('processed_data/y_test_smote.npy')

# 2. 重塑数据为序列形式 (samples, sequence_length, features)
sequence_length = 10
n_features = X.shape[1] // sequence_length
X = X.reshape(-1, sequence_length, n_features)
print(f"样本数据形状: {X.shape}")

# 3. 只分析T2DM阳性样本
positive_indices = np.where(y == 1)[0]
X_pos = X[positive_indices]
print(f"T2DM阳性样本数量: {len(positive_indices)}")

# 4. 加载全局特征统计信息
print("加载全局特征统计信息...")
with open('results/feature_stats.json', 'r') as f:
    feature_stats = json.load(f)

# 5. 定义特征列（确保与X_test_smote.npy中的特征顺序一致）
feature_cols = [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
    'heart_rate', 'QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50'
]

# 6. 加载Transformer模型和注意力
print("加载Transformer模型...")
extractor = TransformerFeatureExtractor(input_dim=n_features, sequence_length=sequence_length)
model = extractor.build()
model.load_weights('new_transformer_feature_extractor_weights.weights.h5')
attn_scores = extractor.get_attention_scores(X_pos)  # shape: [N, head, seq, seq]
print(f"注意力分数形状: {attn_scores.shape}")

# 7. 分析每个样本的attention和异常特征
results = []
for i, (x_seq, attn) in enumerate(zip(X_pos, attn_scores)):
    # 计算每个特征的attention均值（平均所有头和时间步）
    feature_attn = attn.mean(axis=(0, 1))  # shape: [seq]
    top_idx = np.argsort(feature_attn)[-3:]  # 取attention最高的3个特征
    for idx in top_idx:
        value = x_seq[:, idx].mean()
        stats = feature_stats[feature_cols[idx]]
        is_abnormal = (value > stats['mean'] + stats['threshold']) or (value < stats['mean'] - stats['threshold'])
        results.append({
            'sample': i,
            'feature': feature_cols[idx],
            'value': value,
            'is_abnormal': is_abnormal,
            'attn_score': feature_attn[idx]
        })

# 8. 汇总统计
df = pd.DataFrame(results)
summary = df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False)
print("\n特征异常比例（按attention排序）:")
print(summary)

# 9. 保存汇总结果
os.makedirs('results', exist_ok=True)
summary.to_csv('results/attention_abnormal_summary.csv')

# 10. 可视化attention与异常特征的重叠
plt.figure(figsize=(10, 6))
sns.barplot(x=summary.index, y=summary.values)
plt.title('Proportion of Abnormal Features (High Attention)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/attention_abnormal_overlap.png')
plt.close()

# 11. 绘制attention热力图
plt.figure(figsize=(max(8, attn_scores.shape[2]), max(6, attn_scores.shape[2])))
heatmap_data = attn_scores.mean(axis=(0, 1))  # shape: (seq, seq)
N = heatmap_data.shape[0]
labels = feature_cols[:N]  # 只取有数据的特征名
sns.heatmap(heatmap_data, cmap='viridis', xticklabels=labels, yticklabels=labels, vmin=heatmap_data.min(), vmax=heatmap_data.max())
plt.title('Attention Heatmap')
plt.tight_layout()
plt.savefig('results/attention_heatmap.png')
plt.close()

# 12. 绘制attention分数与特征值的散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='attn_score', y='value', hue='is_abnormal', data=df)
plt.title('Attention Score vs. Feature Value')
plt.tight_layout()
plt.savefig('results/attention_scatter.png')
plt.close()

# 13. 绘制正常与异常样本的特征值箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(x='feature', y='value', hue='is_abnormal', data=df)
plt.title('Feature Values: Normal vs. Abnormal')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/feature_boxplot.png')
plt.close()

print("分析完成！结果已保存到 results 目录。")
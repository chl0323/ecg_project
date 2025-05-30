import numpy as np
import pandas as pd
from new_transformer_feature_extractor import TransformerFeatureExtractor
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# 初始化结果列表
results = []

# 补充特征列和序列长度定义
feature_cols = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
     'heart_rate','QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50']
sequence_length = 10

# 加载处理好的数据
print("加载预处理数据...")
X_train = np.load('processed_data/X_train_smote.npy')
y_train = np.load('processed_data/y_train_smote.npy')
X_val = np.load('processed_data/X_val_smote.npy')
y_val = np.load('processed_data/y_val_smote.npy')
X_test = np.load('processed_data/X_test_smote.npy')
y_test = np.load('processed_data/y_test_smote.npy')

# 重塑数据为序列形式 (samples, sequence_length, features)
n_features = X_train.shape[1] // sequence_length
X_train = X_train.reshape(-1, sequence_length, n_features)
X_val = X_val.reshape(-1, sequence_length, n_features)
X_test = X_test.reshape(-1, sequence_length, n_features)

print(f"训练数据形状: {X_train.shape}")
print(f"验证数据形状: {X_val.shape}")
print(f"测试数据形状: {X_test.shape}")

# 读取history.csv，按多指标排序选最优epoch
print("选择最优模型...")
history = pd.read_csv('history.csv')
history['epoch'] = history.index + 1
sorted_history = history.sort_values(
    by=['val_auc', 'val_loss', 'val_accuracy', 'epoch'],
    ascending=[False, True, False, True]
)
best_row = sorted_history.iloc[0]
best_epoch = int(best_row['epoch'])
best_weights_path = f'checkpoints/model_epoch_{best_epoch:02d}.weights.h5'
print(f"选择最优epoch: {best_epoch}, 权重文件: {best_weights_path}")

# 用最佳权重加载完整模型（含分类头）
print("加载最优模型...")
full_model = tf.keras.models.load_model('transformer_full_model.keras', compile=False)

# ========== 评估最优模型在验证集和测试集上的性能 ==========
print("评估模型性能...")
val_pred = full_model.predict(X_val, batch_size=128)
test_pred = full_model.predict(X_test, batch_size=128)

val_pred_label = (val_pred > 0.5).astype(int)
test_pred_label = (test_pred > 0.5).astype(int)

# 计算各项指标
val_metrics = {
    'Accuracy': accuracy_score(y_val, val_pred_label),
    'Recall': recall_score(y_val, val_pred_label),
    'F1 Score': f1_score(y_val, val_pred_label),
    'AUC': roc_auc_score(y_val, val_pred)
}

test_metrics = {
    'Accuracy': accuracy_score(y_test, test_pred_label),
    'Recall': recall_score(y_test, test_pred_label),
    'F1 Score': f1_score(y_test, test_pred_label),
    'AUC': roc_auc_score(y_test, test_pred)
}

print('--- 最优模型在验证集上的性能 ---')
for metric, value in val_metrics.items():
    print(f'{metric}: {value:.4f}')

print('--- 最优模型在测试集上的性能 ---')
for metric, value in test_metrics.items():
    print(f'{metric}: {value:.4f}')

# ========== 创建模型性能可视化 ==========
print("创建模型性能可视化...")
os.makedirs('results/model_performance', exist_ok=True)

# 1. 验证集和测试集性能对比条形图
metrics_df = pd.DataFrame({
    'Metric': list(val_metrics.keys()) * 2,
    'Value': list(val_metrics.values()) + list(test_metrics.values()),
    'Dataset': ['Validation'] * 4 + ['Test'] * 4
})

fig = px.bar(metrics_df, 
             x='Metric', 
             y='Value', 
             color='Dataset',
             barmode='group',
             title='Model Performance Comparison: Validation vs Test',
             labels={'Value': 'Score', 'Metric': 'Performance Metric'},
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(template='plotly_white')
pio.write_html(fig, 'results/model_performance/metrics_comparison.html')

# 2. 性能指标雷达图
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=list(val_metrics.values()),
    theta=list(val_metrics.keys()),
    fill='toself',
    name='Validation'
))
fig.add_trace(go.Scatterpolar(
    r=list(test_metrics.values()),
    theta=list(test_metrics.keys()),
    fill='toself',
    name='Test'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Model Performance Radar Chart'
)
pio.write_html(fig, 'results/model_performance/performance_radar.html')

# 3. ROC曲线
from sklearn.metrics import roc_curve, auc
fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
roc_auc_val = auc(fpr_val, tpr_val)
roc_auc_test = auc(fpr_test, tpr_test)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr_val, y=tpr_val,
    name=f'Validation ROC (AUC = {roc_auc_val:.3f})',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=fpr_test, y=tpr_test,
    name=f'Test ROC (AUC = {roc_auc_test:.3f})',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    name='Random',
    mode='lines',
    line=dict(dash='dash')
))
fig.update_layout(
    title='ROC Curves',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=700
)
pio.write_html(fig, 'results/model_performance/roc_curves.html')

# 4. 混淆矩阵热力图
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(12, 5))

# 验证集混淆矩阵
plt.subplot(1, 2, 1)
cm_val = confusion_matrix(y_val, val_pred_label)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 测试集混淆矩阵
plt.subplot(1, 2, 2)
cm_test = confusion_matrix(y_test, test_pred_label)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('results/model_performance/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 预测概率分布图
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=val_pred[y_val == 1],
    name='Validation Positive',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=val_pred[y_val == 0],
    name='Validation Negative',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=test_pred[y_test == 1],
    name='Test Positive',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=test_pred[y_test == 0],
    name='Test Negative',
    opacity=0.7,
    histnorm='probability'
))
fig.update_layout(
    title='Prediction Probability Distribution',
    xaxis_title='Prediction Probability',
    yaxis_title='Density',
    barmode='overlay'
)
pio.write_html(fig, 'results/model_performance/prediction_distribution.html')

# ========== 创建更多PNG格式可视化 ==========
print("创建PNG格式可视化...")

# 将结果转换为DataFrame
df = pd.DataFrame(results)
print(f"分析结果数据形状: {df.shape}")

# 1. 模型性能指标对比图
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, list(val_metrics.values()), width, label='Validation', color='#3498db')
plt.bar(x + width/2, list(test_metrics.values()), width, label='Test', color='#e74c3c')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Metrics Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/model_performance/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 预测概率密度图
plt.figure(figsize=(12, 6))
sns.kdeplot(data=val_pred[y_val == 1], label='Validation Positive', color='#2ecc71')
sns.kdeplot(data=val_pred[y_val == 0], label='Validation Negative', color='#e74c3c')
sns.kdeplot(data=test_pred[y_test == 1], label='Test Positive', color='#27ae60')
sns.kdeplot(data=test_pred[y_test == 0], label='Test Negative', color='#c0392b')
plt.title('Prediction Probability Density')
plt.xlabel('Prediction Probability')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/model_performance/prediction_density.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 特征重要性条形图
plt.figure(figsize=(12, 6))
feature_importance = df.groupby('feature')['attn_score'].mean().sort_values(ascending=False)
sns.barplot(x=feature_importance.index, y=feature_importance.values, palette='viridis')
plt.title('Feature Importance Based on Attention Scores')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Average Attention Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/model_performance/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 异常特征分布图
plt.figure(figsize=(15, 8))
abnormal_counts = df[df['is_abnormal']].groupby('feature').size()
normal_counts = df[~df['is_abnormal']].groupby('feature').size()
x = np.arange(len(feature_cols))
width = 0.35

plt.bar(x - width/2, [normal_counts.get(f, 0) for f in feature_cols], width, label='Normal', color='#3498db')
plt.bar(x + width/2, [abnormal_counts.get(f, 0) for f in feature_cols], width, label='Abnormal', color='#e74c3c')

plt.xlabel('Features')
plt.ylabel('Count')
plt.title('Normal vs Abnormal Feature Distribution')
plt.xticks(x, feature_cols, rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/model_performance/abnormal_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 注意力分数热力图
plt.figure(figsize=(12, 10))
heatmap_data = attn_scores.mean(axis=(0, 1))
sns.heatmap(heatmap_data, 
            cmap='YlOrRd',
            xticklabels=feature_cols,
            yticklabels=feature_cols,
            annot=True,
            fmt='.2f',
            square=True)
plt.title('Feature Attention Heatmap')
plt.tight_layout()
plt.savefig('results/model_performance/attention_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 特征值箱线图
plt.figure(figsize=(15, 8))
sns.boxplot(x='feature', y='value', hue='is_abnormal', data=df, palette='Set2')
plt.title('Feature Values Distribution')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Value')
plt.legend(title='Status')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/model_performance/feature_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 预测概率与特征值散点图
plt.figure(figsize=(12, 6))
plt.scatter(df['value'], df['attn_score'], 
           c=df['is_abnormal'].astype(int), 
           cmap='viridis',
           alpha=0.6)
plt.colorbar(label='Is Abnormal')
plt.title('Feature Values vs Attention Scores')
plt.xlabel('Feature Value')
plt.ylabel('Attention Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/model_performance/feature_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. 模型性能指标雷达图
plt.figure(figsize=(10, 10))
metrics = list(val_metrics.keys())
values_val = list(val_metrics.values())
values_test = list(test_metrics.values())

# 计算角度
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
values_val = np.concatenate((values_val, [values_val[0]]))
values_test = np.concatenate((values_test, [values_test[0]]))
angles = np.concatenate((angles, [angles[0]]))

ax = plt.subplot(111, polar=True)
ax.plot(angles, values_val, 'o-', linewidth=2, label='Validation')
ax.fill(angles, values_val, alpha=0.25)
ax.plot(angles, values_test, 'o-', linewidth=2, label='Test')
ax.fill(angles, values_test, alpha=0.25)

ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
ax.set_ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
plt.title('Model Performance Radar Chart')
plt.tight_layout()
plt.savefig('results/model_performance/performance_radar.png', dpi=300, bbox_inches='tight')
plt.close()

print("PNG格式可视化完成！所有图表已保存到 results/model_performance 目录")

# ========== 用最佳权重提取全部特征（只做特征保存，不做分类评估） ==========
print("提取特征...")
os.makedirs('processed_data', exist_ok=True)
extractor = TransformerFeatureExtractor(input_dim=n_features, sequence_length=sequence_length)
feature_model = extractor.build()
feature_model.load_weights(best_weights_path)

# 加载全局特征统计信息
print("加载全局特征统计信息...")
with open('results/feature_stats.json', 'r') as f:
    feature_stats = json.load(f)

# 获取阳性样本
positive_indices = np.where(y_test == 1)[0]
X_pos = X_test[positive_indices]
print(f"T2DM阳性样本数量: {len(positive_indices)}")

# 初始化结果列表
results = []

try:
    # 获取注意力分数
    attn_scores = extractor.get_attention_scores(X_pos)
    print(f"注意力分数形状: {attn_scores.shape}")
    print(f"特征列数量: {len(feature_cols)}")
    
    # 检查特征统计信息
    print("检查特征统计信息...")
    for feature in feature_cols:
        if feature not in feature_stats:
            print(f"警告: 特征 {feature} 在统计信息中不存在!")
            print(f"可用的特征: {list(feature_stats.keys())}")
            raise ValueError(f"特征 {feature} 在统计信息中不存在")
    
    # 分析每个样本的attention和异常特征
    print(f"开始分析 {len(X_pos)} 个阳性样本...")
    
    for i, (x_seq, attn) in enumerate(zip(X_pos, attn_scores)):
        # 计算每个特征的attention均值（平均所有头和时间步）
        feature_attn = attn.mean(axis=(0, 1))  # shape: [seq]
        
        # 确保特征数量匹配
        if len(feature_attn) != len(feature_cols):
            print(f"警告: 样本 {i} 的特征数量不匹配! feature_attn: {len(feature_attn)}, feature_cols: {len(feature_cols)}")
            continue
            
        # 获取top 3特征的索引
        top_idx = np.argsort(feature_attn)[-3:]
        
        for idx in top_idx:
            if idx >= len(feature_cols):
                continue
                
            feature_name = feature_cols[idx]
            value = float(x_seq[:, idx].mean())  # 确保值是标量
            stats = feature_stats[feature_name]
            
            is_abnormal = (value > stats['mean'] + stats['threshold']) or (value < stats['mean'] - stats['threshold'])
            attn_score = float(feature_attn[idx])  # 确保值是标量
            
            results.append({
                'sample': int(i),  # 确保是整数
                'feature': str(feature_name),  # 确保是字符串
                'value': value,
                'is_abnormal': bool(is_abnormal),  # 确保是布尔值
                'attn_score': attn_score
            })
    
    print(f"分析完成，收集到 {len(results)} 条结果")
    if len(results) == 0:
        raise ValueError("没有收集到任何结果数据，无法创建可视化")
    
    # 保存特征
    embeddings = feature_model.predict(X_test, batch_size=128)
    np.save('processed_data/ecg_embeddings.npy', embeddings)
    np.save('processed_data/ecg_labels.npy', y_test)
    print("特征提取完成，已保存到 processed_data 目录")
    
    # ========== 创建高级可视化 ==========
    print("创建高级可视化...")
    os.makedirs('results/visualizations', exist_ok=True)
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(results)
    print(f"DataFrame形状: {df.shape}")
    print("DataFrame列名:", df.columns.tolist())
    print("DataFrame前几行:")
    print(df.head())
    
    if df.empty:
        raise ValueError("DataFrame为空，无法创建可视化")
    
    # 确保所需的列都存在
    required_columns = ['feature', 'value', 'is_abnormal', 'attn_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必要的列: {missing_columns}")

    # 1. 交互式条形图
    fig = px.bar(df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False), 
                 title='Proportion of Abnormal Features (High Attention)',
                 labels={'value': 'Abnormal Proportion', 'index': 'Feature'},
                 color='value',
                 color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Proportion of Abnormal Values",
        template="plotly_white"
    )
    pio.write_html(fig, 'results/visualizations/attention_abnormal_interactive.html')
    
    # 2. 高级热力图
    plt.figure(figsize=(12, 10))
    heatmap_data = attn_scores.mean(axis=(0, 1))
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["#2c3e50", "#3498db", "#e74c3c"])
    sns.heatmap(heatmap_data, 
                cmap=custom_cmap,
                xticklabels=feature_cols,
                yticklabels=feature_cols,
                annot=True,
                fmt='.2f',
                square=True)
    plt.title('Attention Heatmap with Custom Colormap', pad=20)
    plt.tight_layout()
    plt.savefig('results/visualizations/attention_heatmap_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 小提琴图 + 箱线图组合
    plt.figure(figsize=(15, 8))
    sns.violinplot(x='feature', y='value', hue='is_abnormal', data=df,
                  split=True, inner='box', palette='Set2')
    plt.title('Feature Value Distribution: Normal vs Abnormal', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_violin_box.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 交互式散点图矩阵
    fig = px.scatter_matrix(df,
                           dimensions=['value', 'attn_score'],
                           color='is_abnormal',
                           title='Feature Value vs Attention Score Matrix',
                           labels={'value': 'Feature Value', 'attn_score': 'Attention Score'},
                           color_discrete_sequence=px.colors.qualitative.Set2)
    pio.write_html(fig, 'results/visualizations/feature_matrix_interactive.html')
    
    # 5. 3D散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=df['value'],
        y=df['attn_score'],
        z=df['sample'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['is_abnormal'].astype(int),
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df['feature']
    )])
    fig.update_layout(
        title='3D Feature Space Visualization',
        scene=dict(
            xaxis_title='Feature Value',
            yaxis_title='Attention Score',
            zaxis_title='Sample Index'
        )
    )
    pio.write_html(fig, 'results/visualizations/3d_feature_space.html')
    
    # 6. 雷达图
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False).values,
        theta=df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False).index,
        fill='toself',
        name='Abnormal Proportion'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Feature Abnormal Proportion Radar Chart'
    )
    pio.write_html(fig, 'results/visualizations/feature_radar.html')
    
    # 7. 特征重要性条形图
    plt.figure(figsize=(12, 6))
    feature_importance = df.groupby('feature')['attn_score'].mean().sort_values(ascending=False)
    sns.barplot(x=feature_importance.index, y=feature_importance.values, palette='viridis')
    plt.title('Feature Importance Based on Attention Scores', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 交互式时序图
    fig = go.Figure()
    for feature in feature_cols:
        feature_data = df[df['feature'] == feature]
        fig.add_trace(go.Scatter(
            x=feature_data['sample'],
            y=feature_data['value'],
            mode='lines+markers',
            name=feature,
            hovertemplate='Sample: %{x}<br>Value: %{y:.2f}<br>Feature: ' + feature
        ))
    fig.update_layout(
        title='Feature Values Over Samples',
        xaxis_title='Sample Index',
        yaxis_title='Feature Value',
        hovermode='closest'
    )
    pio.write_html(fig, 'results/visualizations/feature_timeline.html')
    
    print("可视化完成！所有图表已保存到 results/visualizations 目录")
    
except Exception as e:
    print(f"Error during feature extraction or visualization: {str(e)}")
    if 'embeddings' in locals():
        np.save('interrupted_embeddings.npy', embeddings)
        print("Embeddings state saved. You can resume extraction later.")
    raise  # Re-raise the exception to see the full traceback

# 确保检查点目录存在（如需后续训练）
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import os

# 定义基础特征列
def get_base_feature_cols():
    return [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
    'anchor_age', 'gender', 'ecg_sequence', 'heart_rate']

# 定义需要计算的特征
def calculate_additional_features(df):
    """计算额外的特征"""
    print("[特征计算] 开始计算额外特征...")
    
    # 计算QTc相关比率
    df['QTc_RR_ratio'] = df['QTc_Interval'] / df['RR_Interval']
    df['QT_RR_ratio'] = df['QT_Interval'] / df['RR_Interval']
    
    # 计算波峰比率
    df['P_R_ratio'] = df['P_Wave_Peak'] / df['R_Wave_Peak']
    df['T_R_ratio'] = df['T_Wave_Peak'] / df['R_Wave_Peak']
    
    # 计算心率变异性指标
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id]
        rr_intervals = subject_data['RR_Interval'].dropna()
        if len(rr_intervals) > 1:
            # RMSSD
            df.loc[df['subject_id'] == subject_id, 'HRV_RMSSD'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            # pNN50
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            df.loc[df['subject_id'] == subject_id, 'HRV_pNN50'] = (nn50 / (len(rr_intervals) - 1)) * 100
    
    print("[特征计算] 完成！")
    return df

# 获取所有特征列
def get_all_feature_cols():
    base_cols = get_base_feature_cols()
    additional_cols = ['QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio', 'HRV_RMSSD', 'HRV_pNN50']
    return base_cols + additional_cols

feature_cols = get_all_feature_cols()

sequence_length = 10

def redistribute_subject_ids(df, sequence_length=10):
    """
    重新分配subject_id，将每个subject_id的ECG数据按sequence_length分组
    如果剩余数据不足sequence_length，则丢弃
    """
    print("[数据重分配] 开始重新分配subject_id...")
    new_df = pd.DataFrame()
    new_subject_id = 19998592  # 从19998592开始
    
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id].copy()
        n_sequences = len(subject_data) // sequence_length
        
        # 处理完整的序列
        for i in range(n_sequences):
            start_idx = i * sequence_length
            end_idx = (i + 1) * sequence_length
            sequence_data = subject_data.iloc[start_idx:end_idx].copy()
            sequence_data['subject_id'] = new_subject_id
            sequence_data['ecg_sequence'] = range(1, sequence_length + 1)
            new_df = pd.concat([new_df, sequence_data])
            new_subject_id += 1
    
    print(f"[数据重分配] 完成！原始subject_id数: {len(df['subject_id'].unique())}, 新subject_id数: {new_subject_id-19998592}")
    return new_df

def prepare_sequence_data_by_subject(data, sequence_length=10):
    X_sequences, y_sequences = [], []
    for subject_id in data['subject_id'].unique():
        subject_data = data[data['subject_id'] == subject_id]
        subject_features = subject_data[feature_cols].values
        subject_labels = subject_data['target'].values
        # 由于已经按sequence_length分组，直接使用整个序列
        X_sequences.append(subject_features)
        y_sequences.append(subject_labels[0])  # 使用序列的第一个标签
    return np.array(X_sequences), np.array(y_sequences)

def groupwise_rus_concat(df, group_col='gender', sequence_length=10, random_state=42):
    """
    按分组变量分组后分别做RUS再拼接，返回全局平衡数据和分组标签
    """
    all_X, all_y, all_group = [], [], []
    for group_value in df[group_col].unique():
        group_df = df[df[group_col] == group_value]
        X_seq, y_seq = prepare_sequence_data_by_subject(group_df, sequence_length)
        if len(X_seq) == 0:
            continue
        X_seq_2d = X_seq.reshape(X_seq.shape[0], -1)
        rus = RandomUnderSampler(random_state=random_state)
        X_bal, y_bal = rus.fit_resample(X_seq_2d, y_seq)
        all_X.append(X_bal)
        all_y.append(y_bal)
        all_group.append(np.full(len(y_bal), group_value))
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    group_final = np.concatenate(all_group, axis=0)
    return X_final, y_final, group_final

def load_and_preprocess_data(test_size=0.2, val_size=0.1, random_state=42):
    """
    加载数据并进行预处理，包括序列生成、数据平衡和数据集划分
    """
    print("[数据预处理] 开始...")
    df = pd.read_excel('/Users/pursuing/Downloads/project/second_data_processing_forms/data_binary2.xlsx')
    
    # 打印数据信息
    print("\n[数据信息]")
    print("列名:", df.columns.tolist())
    print("数据形状:", df.shape)
    print("数据类型:\n", df.dtypes)
    
    # 检查subject_id列是否存在
    if 'subject_id' not in df.columns:
        print("\n[错误] 未找到subject_id列！")
        print("可用的列名:", df.columns.tolist())
        return None
    
    # 计算额外特征
    df = calculate_additional_features(df)
    
    # 重新分配subject_id
    df = redistribute_subject_ids(df, sequence_length)
    
    # 打印重分配后的数据信息
    print("\n[重分配后数据信息]")
    print("列名:", df.columns.tolist())
    print("数据形状:", df.shape)
    print("数据类型:\n", df.dtypes)
    
    X_seq, y_seq = prepare_sequence_data_by_subject(df, sequence_length)
    X_seq_2d = X_seq.reshape(X_seq.shape[0], -1)
    
    smote = SMOTE(random_state=random_state)
    X_balanced_smote, y_balanced_smote = smote.fit_resample(X_seq_2d, y_seq)
    
    rus = RandomUnderSampler(random_state=random_state)
    X_balanced_rus, y_balanced_rus = rus.fit_resample(X_seq_2d, y_seq)
    
    class_weights = {0: 1, 1: len(y_seq[y_seq == 0]) / len(y_seq[y_seq == 1])}
    
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/X_balanced_smote.npy', X_balanced_smote)
    np.save('processed_data/y_balanced_smote.npy', y_balanced_smote)
    np.save('processed_data/X_balanced_rus.npy', X_balanced_rus)
    np.save('processed_data/y_balanced_rus.npy', y_balanced_rus)
    
    # SMOTE划分
    X_temp, X_test_smote, y_temp, y_test_smote = train_test_split(
        X_balanced_smote, y_balanced_smote, test_size=test_size, random_state=random_state, stratify=y_balanced_smote)
    val_ratio = val_size / (1 - test_size)
    X_train_smote, X_val_smote, y_train_smote, y_val_smote = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp)
    
    # RUS划分
    X_temp, X_test_rus, y_temp, y_test_rus = train_test_split(
        X_balanced_rus, y_balanced_rus, test_size=test_size, random_state=random_state, stratify=y_balanced_rus)
    X_train_rus, X_val_rus, y_train_rus, y_val_rus = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp)
    
    np.save('processed_data/X_train_smote.npy', X_train_smote)
    np.save('processed_data/y_train_smote.npy', y_train_smote)
    np.save('processed_data/X_val_smote.npy', X_val_smote)
    np.save('processed_data/y_val_smote.npy', y_val_smote)
    np.save('processed_data/X_test_smote.npy', X_test_smote)
    np.save('processed_data/y_test_smote.npy', y_test_smote)
    np.save('processed_data/X_train_rus.npy', X_train_rus)
    np.save('processed_data/y_train_rus.npy', y_train_rus)
    np.save('processed_data/X_val_rus.npy', X_val_rus)
    np.save('processed_data/y_val_rus.npy', y_val_rus)
    np.save('processed_data/X_test_rus.npy', X_test_rus)
    np.save('processed_data/y_test_rus.npy', y_test_rus)
    
    print(f"[数据预处理] 完成！SMOTE训练集: {X_train_smote.shape}, 验证集: {X_val_smote.shape}, 测试集: {X_test_smote.shape}")
    print(f"[数据预处理] 完成！RUS训练集: {X_train_rus.shape}, 验证集: {X_val_rus.shape}, 测试集: {X_test_rus.shape}")
    
    # 分组后分别做RUS再拼接
    print("[分组平衡] 按gender分组分别做RUS...")
    X_group_rus, y_group_rus, group_labels = groupwise_rus_concat(df, group_col='gender', sequence_length=sequence_length, random_state=random_state)
    np.save('processed_data/X_groupwise_rus.npy', X_group_rus)
    np.save('processed_data/y_groupwise_rus.npy', y_group_rus)
    np.save('processed_data/group_labels_rus.npy', group_labels)
    print(f"[分组平衡] 完成！拼接后样本数: {X_group_rus.shape[0]}，分组标签样本数: {group_labels.shape[0]}")
    
    return {
        'smote': {
            'train': (X_train_smote, y_train_smote),
            'val': (X_val_smote, y_val_smote),
            'test': (X_test_smote, y_test_smote),
            'full': (X_balanced_smote, y_balanced_smote)
        },
        'rus': {
            'train': (X_train_rus, y_train_rus),
            'val': (X_val_rus, y_val_rus),
            'test': (X_test_rus, y_test_rus),
            'full': (X_balanced_rus, y_balanced_rus)
        },
        'groupwise_rus': {
            'X': X_group_rus,
            'y': y_group_rus,
            'group': group_labels
        },
        'class_weights': class_weights
    }

if __name__ == "__main__":
    data_dict = load_and_preprocess_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    print("所有数据集已保存到 'processed_data' 目录。")
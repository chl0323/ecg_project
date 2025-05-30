import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set English font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Read original data
df = pd.read_excel('data_binary2.xlsx')

# Print column names
print("\n=== Data Columns ===")
print(df.columns.tolist())

# Get all feature columns (excluding target column)
features = [col for col in df.columns if col != 'target']

# 1. Basic Statistics
print("\n=== Basic Statistics ===")
print(df[features].describe())

# 2. Outlier Detection (using 3 standard deviations as threshold)
print("\n=== Outlier Detection ===")
abnormal_stats = []
feature_stats = {}  # Store statistics for each feature

for feature in features:
    mean = df[feature].mean()
    std = df[feature].std()
    threshold = 3 * std
    abnormal_count = len(df[abs(df[feature] - mean) > threshold])
    abnormal_ratio = abnormal_count/len(df)*100
    
    print(f"\n{feature} Outlier Statistics:")
    print(f"Mean: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Outlier Count (>3σ): {abnormal_count}")
    print(f"Outlier Ratio: {abnormal_ratio:.2f}%")
    
    abnormal_stats.append({
        'Feature': feature,
        'Mean': mean,
        'Std': std,
        'AbnormalCount': abnormal_count,
        'AbnormalRatio': abnormal_ratio
    })
    
    # Store feature statistics
    feature_stats[feature] = {
        'mean': float(mean),  # Convert to Python native type
        'std': float(std),
        'threshold': float(threshold),
        'abnormal_count': int(abnormal_count),
        'abnormal_ratio': float(abnormal_ratio)
    }

# Save feature statistics to JSON file
with open('feature_stats.json', 'w') as f:
    json.dump(feature_stats, f, indent=4)
print("\nFeature statistics saved to feature_stats.json")

# Convert abnormal statistics to DataFrame and sort
abnormal_df = pd.DataFrame(abnormal_stats)
abnormal_df = abnormal_df.sort_values('AbnormalRatio', ascending=False)
print("\n=== Outlier Ratio Ranking ===")
print(abnormal_df[['Feature', 'AbnormalCount', 'AbnormalRatio']])

# 3. Visualization
# 3.1 Outlier ratio bar plot
plt.figure(figsize=(15, 6))
sns.barplot(data=abnormal_df, x='Feature', y='AbnormalRatio')
plt.xticks(rotation=45, ha='right')
plt.title('Outlier Ratio by Feature')
plt.tight_layout()
plt.savefig('abnormal_ratios.png')
plt.close()

# 3.2 Feature distribution plots (4 features per row)
n_features = len(features)
n_rows = (n_features + 3) // 4  # Round up
plt.figure(figsize=(20, 5*n_rows))

for i, feature in enumerate(features, 1):
    plt.subplot(n_rows, 4, i)
    sns.histplot(data=df, x=feature, bins=50)
    plt.title(f'{feature} Distribution')
    plt.axvline(df[feature].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(df[feature].mean() + 3*df[feature].std(), color='g', linestyle='--', label='3σ')
    plt.axvline(df[feature].mean() - 3*df[feature].std(), color='g', linestyle='--')
    plt.legend()

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# 4. Correlation Analysis
print("\n=== Correlation Analysis ===")
correlation = df[features].corr()
print(correlation)

# 5. Relationship with Target Variable
print("\n=== Relationship with Target Variable ===")
for feature in features:
    print(f"\n{feature} Statistics by Target Value:")
    print(df.groupby('target')[feature].describe())

# 6. Save Detailed Analysis Results
with open('feature_analysis.txt', 'w') as f:
    f.write("=== Feature Analysis Results ===\n\n")
    f.write("1. Data Columns:\n")
    f.write(str(df.columns.tolist()) + "\n\n")
    f.write("2. Basic Statistics:\n")
    f.write(str(df[features].describe()) + "\n\n")
    f.write("3. Outlier Statistics:\n")
    f.write(str(abnormal_df) + "\n\n")
    f.write("4. Correlation Analysis:\n")
    f.write(str(correlation) + "\n\n")
    f.write("5. Relationship with Target Variable:\n")
    for feature in features:
        f.write(f"\n{feature} Statistics by Target Value:\n")
        f.write(str(df.groupby('target')[feature].describe()) + "\n") 
import tensorflow as tf
from keras import layers, Model

class TransformerFeatureExtractor:
    def __init__(self, input_dim, sequence_length, num_heads=4, ff_dim=128, num_layers=2, dropout=0.1):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_layers = []  # 保存每层attention层对象

    def build(self):
        inputs = layers.Input(shape=(self.sequence_length, self.input_dim))
        x = inputs
        self.attention_layers = []
        for _ in range(self.num_layers):
            attn_layer = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.ff_dim // self.num_heads
            )
            attn = attn_layer(x, x, return_attention_scores=False)
            self.attention_layers.append(attn_layer)
            x = layers.LayerNormalization()(x + attn)
            x = layers.BatchNormalization()(x)
            ffn = layers.Dense(self.ff_dim, activation='relu')(x)
            ffn = layers.Dense(self.input_dim)(ffn)
            x = layers.LayerNormalization()(x + ffn)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
        embedding = layers.GlobalAveragePooling1D()(x)
        return Model(inputs, embedding)

    def get_attention_scores(self, x_input, layer_idx=0):
        """
        获取指定层的attention scores。
        x_input: shape (batch, seq_len, input_dim)
        layer_idx: 指定第几层的attention（从0开始）
        返回: (batch, num_heads, seq_len, seq_len)
        """
        if not isinstance(x_input, tf.Tensor):
            x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
        
        # 确保输入数据形状正确
        if len(x_input.shape) != 3:
            raise ValueError(f"输入数据应该是3维的 (batch_size, sequence_length, features), 但得到的是 {x_input.shape}")
        
        if x_input.shape[1] != self.sequence_length:
            raise ValueError(f"序列长度应该是 {self.sequence_length}, 但得到的是 {x_input.shape[1]}")
        
        if x_input.shape[2] != self.input_dim:
            raise ValueError(f"特征维度应该是 {self.input_dim}, 但得到的是 {x_input.shape[2]}")
        
        attn_layer = self.attention_layers[layer_idx]
        _, attn_scores = attn_layer(x_input, x_input, return_attention_scores=True)
        return attn_scores.numpy()
# Archivo: kan_bert_model.py

from typing import Union, Tuple, List, Any


import tensorflow as tf
import os

from bert_model import BertModel, Embedding, AttentionBlock, Pooler, load_weights_from_checkpoint
from kan_layer import DenseKAN


class KANOutput(tf.keras.layers.Layer):
    def __init__(self, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        hidden_states, input_tensor = inputs
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class KANEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.attention = AttentionBlock(hidden_size, num_heads, dropout_rate, name='attention')
        self.kan_output = KANOutput(dropout_rate, name='output')

    def call(self, inputs: tf.Tensor, shared_ffn_layer: tf.keras.layers.Layer) -> tf.Tensor:
        attention_output = self.attention(inputs)
        ffn_output = shared_ffn_layer(attention_output) 
        layer_output = self.kan_output([ffn_output, attention_output])
        return layer_output

class KANEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            KANEncoderLayer(hidden_size, num_heads, dropout_rate, name=f'layer_{i}')
            for i in range(num_layers)
        ]
    
    def call(self, inputs: tf.Tensor, *, shared_ffn_layer: tf.keras.layers.Layer) -> tf.Tensor:
        hidden_states = inputs
        for layer in self.layers:
            hidden_states = layer(hidden_states, shared_ffn_layer=shared_ffn_layer)
        return hidden_states

class KanBertModel(tf.keras.Model):
    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = Embedding(config['vocab_size'], config['max_position_embeddings'], config['type_vocab_size'], config['hidden_size'], config['dropout_rate'], name='embeddings')
        self.pooler = Pooler(config['hidden_size'], name='pooler')
        
        kan_intermediate_size = config.get('kan_intermediate_size', 64) # Permitir override
        kan_grid_size = config.get('kan_grid_size', 3)
        
        self.shared_kan_ffn = tf.keras.Sequential([
            DenseKAN(kan_intermediate_size, grid_size=kan_grid_size, spline_order=3, name='ffn_intermediate_kan', track_inputs = False),
            DenseKAN(config['hidden_size'], grid_size=kan_grid_size, spline_order=3, name='ffn_output_kan', track_inputs = False)
        ], name='shared_kan_ffn')
        
        self.shared_kan_ffn.build(input_shape=(None, None, config['hidden_size']))

        self.encoder = KANEncoder(
            config['num_layers'], config['hidden_size'], config['num_heads'],
            config['dropout_rate'], name='encoder'
        )

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], return_sequence: bool = False) -> Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        input_ids, segment_ids = inputs
        embedding_output = self.embeddings([input_ids, segment_ids])
        sequence_output = self.encoder(embedding_output, shared_ffn_layer=self.shared_kan_ffn)
        pooled_output = self.pooler(sequence_output)
        return (pooled_output, sequence_output) if return_sequence else pooled_output
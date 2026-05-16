import tensorflow as tf
import numpy as np

class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_position_embeddings, type_vocab_size, hidden_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = tf.keras.layers.Embedding(vocab_size, hidden_size, name='word_embeddings')
        self.position_embeddings = tf.keras.layers.Embedding(max_position_embeddings, hidden_size, name='position_embeddings')
        self.token_type_embeddings = tf.keras.layers.Embedding(type_vocab_size, hidden_size, name='token_type_embeddings')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        input_ids, segment_ids = inputs
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        token_embeddings, position_embeddings, segment_embeddings = self.word_embeddings(input_ids), self.position_embeddings(position_ids), self.token_type_embeddings(segment_ids)
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert hidden_size % num_heads == 0
        self.num_heads, self.head_size = num_heads, hidden_size // num_heads
        self.query, self.key, self.value = [tf.keras.layers.Dense(hidden_size, name=n) for n in ['query', 'key', 'value']]

    def split_heads(self, x, batch_size):
        return tf.transpose(tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size)), perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query_layer, key_layer, value_layer = self.query(inputs), self.key(inputs), self.value(inputs)
        query_heads, key_heads, value_heads = [self.split_heads(t, batch_size) for t in [query_layer, key_layer, value_layer]]
        attention_scores = tf.matmul(query_heads, key_heads, transpose_b=True) / tf.math.sqrt(float(self.head_size))
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        context_layer = tf.matmul(attention_probs, value_heads)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        return tf.reshape(context_layer, (batch_size, -1, self.num_heads * self.head_size))

class AttentionOutput(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hidden_size, name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        hidden_states, input_tensor = inputs
        hidden_states = self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)
        return hidden_states

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.self = SelfAttention(hidden_size, num_heads, name='self')
        self.attention_output = AttentionOutput(hidden_size, dropout_rate, name='output')

    def call(self, inputs):
        self_output = self.self(inputs)
        return self.attention_output([self_output, inputs])

class BertIntermediate(tf.keras.layers.Layer):
    def __init__(self, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(intermediate_size, activation='gelu', name='dense')

    def call(self, hidden_states):
        return self.dense(hidden_states)

class BertOutput(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hidden_size, name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        hidden_states, input_tensor = inputs
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.attention = AttentionBlock(hidden_size, num_heads, dropout_rate, name='attention')
        self.intermediate = BertIntermediate(intermediate_size, name='intermediate')
        self.bert_output = BertOutput(hidden_size, dropout_rate, name='output')

    def call(self, inputs):
        attention_output = self.attention(inputs)
        intermediate_output = self.intermediate(attention_output)
        return self.bert_output([intermediate_output, attention_output])

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.layer = [EncoderLayer(hidden_size, num_heads, intermediate_size, dropout_rate, name=f'layer_{i}') for i in range(num_layers)]
    
    def call(self, inputs):
        x = inputs
        for layer in self.layer: x = layer(x)
        return x

class Pooler(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hidden_size, activation='tanh', name='dense')
    
    def call(self, inputs):
        return self.dense(inputs[:, 0])

class BertModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = Embedding(config['vocab_size'], config['max_position_embeddings'], config['type_vocab_size'], config['hidden_size'], config['dropout_rate'], name='embeddings')
        self.encoder = Encoder(config['num_layers'], config['hidden_size'], config['num_heads'], config['intermediate_size'], config['dropout_rate'], name='encoder')
        self.pooler = Pooler(config['hidden_size'], name='pooler')

    def call(self, inputs, return_sequence=False):
        input_ids, segment_ids = inputs
        embedding_output = self.embeddings([input_ids, segment_ids])
        sequence_output = self.encoder(embedding_output)
        pooled_output = self.pooler(sequence_output)
        return (pooled_output, sequence_output) if return_sequence else pooled_output

def load_weights_from_checkpoint(model, checkpoint_path, exclude_strings=None):
    """
    Carga los pesos de un checkpoint de TF1 en un modelo Keras,
    manejando dinámicamente los prefijos de nombres de modelo.

    Args:
        model: El modelo Keras en el que cargar los pesos.
        checkpoint_path: Ruta al checkpoint de TF1.
        exclude_strings: Una lista de strings. Si el path de un peso
                         contiene alguna de estas strings, no se cargará.
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    ckpt_vars = {name: name for name, _ in tf.train.list_variables(checkpoint_path)}
    
    skipped_weights, loaded_weights_count, excluded_by_user = [], 0, []

    for weight in model.weights:
        path = weight.path.split(':')[0]
        
        if exclude_strings and any(ex_str in path for ex_str in exclude_strings):
            excluded_by_user.append(path)
            continue


        relative_path = path.replace(model.name + '/', '', 1)

        if 'embeddings/' in relative_path and relative_path.endswith('/embeddings'):
            relative_path = relative_path[:-11]

        ckpt_name = 'bert/' + relative_path
        
        if ckpt_name in ckpt_vars:
            try:
                weight.assign(reader.get_tensor(ckpt_name))
                loaded_weights_count += 1
            except Exception as e:
                print(f"Error al asignar el peso {path} como {ckpt_name}: {e}")
        else:
            if 'kan' not in path:
                skipped_weights.append(path)

    print(f"\nCargados {loaded_weights_count} de {len(model.weights)} pesos del modelo.")
    if excluded_by_user:
        print("Pesos excluidos por el usuario (para entrenamiento de FFN):")
        for name in excluded_by_user: print(f" - {name}")
    if skipped_weights:
        print("Pesos no encontrados en checkpoint (normal para KAN o clasificador):")
        for name in skipped_weights: print(f" - {name}")
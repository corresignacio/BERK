from collections import defaultdict
import tensorflow as tf

def analyze_model_parameters(model: tf.keras.Model) -> dict:
    """
    Clasifica y cuenta los parámetros de un modelo BERT o KAN-BERT por módulo,
    con una lógica de clasificación robusta.
    
    Args:
        model: Un modelo Keras construido.
        
    Returns:
        Un diccionario con el conteo de parámetros por módulo.
    """
    param_counts = defaultdict(lambda: {'trainable': 0, 'non_trainable': 0})
    
    for var in model.weights:
        path = var.path.lower()
        count = tf.size(var).numpy()
        
        module = "Otros" # Default
        
        # ---- LÓGICA DE CLASIFICACIÓN CORREGIDA Y SIMPLIFICADA ----
        # El orden de las condiciones es importante.
        
        if "embeddings" in path:
            module = "Embeddings"
        elif "pooler" in path:
            module = "Pooler"
        elif "layernorm" in path:
            module = "Normalización (LN)"
        elif "attention" in path:
            module = "Atención (sin LN)"
        # Si no es ninguno de los anteriores, y está en el encoder o en la ffn, es FFN.
        # Esto captura 'intermediate/dense' y 'output/dense' en BERT
        # y 'shared_kan_ffn' en KAN-BERT.
        elif "encoder" in path or "ffn" in path:
            module = "FFN (pura)"
            
        if var.trainable:
            param_counts[module]['trainable'] += count
        else:
            param_counts[module]['non_trainable'] += count
            
    return dict(param_counts)


def print_parameter_analysis(analysis: dict, model_name: str):
    """Imprime una tabla formateada del análisis de parámetros."""
    print(f"\n--- Análisis Detallado de Parámetros para: {model_name} ---")
    print("-" * 85)
    print(f"{'Módulo':<25} | {'Entrenables':>15} | {'No Entrenables':>15} | {'Total':>15}")
    print("-" * 85)
    
    total_trainable = 0
    total_non_trainable = 0
    
    modules = ["Embeddings", "Atención (sin LN)", "FFN (pura)", "Normalización (LN)", "Pooler", "Otros"]
    for module in modules:
        if module in analysis:
            trainable = analysis[module]['trainable']
            non_trainable = analysis[module]['non_trainable']
            total = trainable + non_trainable
            print(f"{module:<25} | {trainable:>15,} | {non_trainable:>15,} | {total:>15,}")
            total_trainable += trainable
            total_non_trainable += non_trainable
            
    print("-" * 85)
    grand_total = total_trainable + total_non_trainable
    print(f"{'TOTAL':<25} | {total_trainable:>15,} | {total_non_trainable:>15,} | {grand_total:>15,}")
    print("-" * 85)
    return total_trainable, total_non_trainable
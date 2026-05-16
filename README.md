# BERK: Bidirectional Encoder Representations from Kanformers

Este repositorio contiene el código y los experimentos desarrollados para la tesis sobre la arquitectura **BERK**, una variante híbrida del modelo BERT que integra Redes de Kolmogorov-Arnold (KAN) con el objetivo de explorar la eficiencia y la reducción de redundancia en los módulos de Feed-Forward Networks (FFN).

## Descripción del Proyecto

Con el surgimiento de la arquitectura Transformer, el procesamiento del lenguaje natural (NLP) ha avanzado significativamente. Sin embargo, diversos estudios sugieren que modelos como BERT podrían presentar una sobre-parametrización considerable, especialmente en sus capas densas (FFN).

**BERK** propone reemplazar los módulos FFN tradicionales por capas **Kolmogorov-Arnold (KAN)** compartidas. Esta reformulación permitió observar:
- **Reducción en el conteo de parámetros:** Una disminución del **81,3%** en los parámetros específicos del módulo FFN.
- **Desempeño comparable:** Los experimentos iniciales sugieren rendimientos cercanos a la versión *Tiny* de BERT en diversas tareas de clasificación del benchmark **GLUE**, manteniendo una capacidad competitiva con una fracción de los parámetros.

### Comparativa de Resultados (Validación GLUE)

El siguiente cuadro resume el desempeño obtenido en los subconjuntos de validación frente al modelo BERT tradicional:

| Arquitectura | # Parámetros FFN | SST-2 | QQP | QNLI | MNLI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BERK** | 49,296 | **82.7%** | 83.4% | 78.2% | 68.7% |
| BERT | 263,424 | 81.0% | **85.1%** | **78.3%** | **70.1%** |

*Nota: Los resultados indican que BERK podría ofrecer un equilibrio favorable entre eficiencia y precisión para entornos con recursos limitados.*

#### Infraestructura y Hardware
Los entrenamientos fueron realizados en instancias de pago de Google Colab dotadas con placas de video NVIDIA A100 de 80 GB de Video-RAM. 


---

## Estructura del Código

El proyecto está organizado en dos carpetas principales: `models/` (implementación del motor) y `experimentos/` (validación y pruebas).

### 📂 `models/`
Contiene la implementación *from scratch* en **Keras/TensorFlow** de los modelos y capas:

- **`bert_model.py`**: Implementación base de BERT. Incluye la lógica para cargar pesos oficiales de Google (TinyBERT) desde checkpoints de TensorFlow 1.x.
- **`kan_layer.py`**: Implementación de la capa **DenseKAN**. Utiliza B-splines como funciones de activación en las aristas, permitiendo una mayor expresividad con menos parámetros.
- **`kan_bert.py`**: El modelo híbrido **BERK**. Reemplaza los bloques `BertIntermediate` y `BertOutput` por un módulo KAN FFN compartido entre todos los encoders.
- **`utils.py`**: Herramientas para el análisis y conteo detallado de parámetros, permitiendo comparar la carga computacional entre BERT y BERK.

### 📂 `experimentos/`
Notebooks de Jupyter con el flujo de trabajo de la investigación:

- **`KAN_layer.ipynb`**: Pruebas funcionales de la capa KAN comparándola con capas MLP tradicionales (Dense).
- **`MyBERT_vs_HFBERT.ipynb`**: Validación de nuestra implementación. Compara las representaciones vectoriales (embeddings) generadas por nuestro código frente a las de Hugging Face para asegurar la paridad numérica.
- **`BERT.ipynb`**: Demostración de la carga de pesos pre-entrenados y el funcionamiento básico del encoder.
- **`BERT_vs_KANBERT.ipynb`**: **Notebook principal de la tesis**. Contiene el proceso de Fine-tuning en los datasets de GLUE, la comparación de curvas de aprendizaje y la obtención de los resultados finales presentados en la tesis.

---

## Uso y Ejecución

### Requisitos
- Python 3.8+
- TensorFlow 2.x
- Hugging Face `transformers` (solo para tokenización y validación cruzada)
- NumPy, Matplotlib, Pandas

### Ejecución Local
1. **Preparación de Pesos**: Asegurarse de contar con la carpeta `BERT_tiny` que contenga los pesos oficiales de Google en el formato original de TensorFlow checkpoint.
2. **Notebooks**: La forma más directa de replicar los experimentos consiste en abrir las notebooks en la carpeta `experimentos/`. Se sugiere comenzar por `BERT_vs_KANBERT.ipynb` para observar el entrenamiento completo.
3. **Integración de Modelos**:
   ```python
   from models.kan_bert import KanBertModel
   
   config = {
       "vocab_size": 30522,
       "hidden_size": 128,
       "num_layers": 2,
       "num_heads": 2,
       "kan_intermediate_size": 16,
       # ... otros parámetros
   }
   model = KanBertModel(config)
   ```

## Referencias
- *Attention is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al., 2018)
- *KAN: Kolmogorov-Arnold Networks* (Liu et al., 2024)
- *One Wide Feedforward is All You Need* (Belcak & Wattenhofer, 2024)

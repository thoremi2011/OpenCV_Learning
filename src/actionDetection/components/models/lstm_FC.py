import tensorflow as tf
import keras
from keras import layers, Model
from keras.layers import Dense, LSTM, Input, Dropout
from typing import Tuple

class Attention(tf.keras.layers.Layer):
    """
    Capa de atención que aprende a ponderar la importancia de cada paso temporal
    """
    def __init__(self, units: int):
        """
        Args:
            units: Dimensionalidad del espacio de atención
        """
        super(Attention, self).__init__()
        self.W = Dense(units)  # Capa que aprende qué características son importantes
        self.V = Dense(1)      # Capa que convierte las características en un score

    def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calcula los pesos de atención y el vector de contexto
        Args:
            hidden_states: Estados ocultos del LSTM, shape (batch_size, seq_len, hidden_size)
        Returns:
            context_vector: Vector de contexto ponderado, shape (batch_size, hidden_size)
            attention_weights: Pesos de atención, shape (batch_size, seq_len, 1)
        """
        # 1. Calcular un score para cada estado oculto
        # shape: (batch_size, seq_len, units)
        score = tf.nn.tanh(self.W(hidden_states))
        
        # 2. Convertir scores a pesos mediante softmax
        # shape: (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # 3. Crear vector de contexto como suma ponderada de estados ocultos
        # shape: (batch_size, hidden_size)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

def create_lstm_attention_model(
    sequence_length: int,
    num_features: int,
    num_classes: int,
    lstm_units: list = [256, 128],  # Lista con unidades para cada capa LSTM
    attention_units: int = 128,
    dropout_rate: float = 0.3
) -> Model:
    """
    Crea un modelo LSTM con mecanismo de atención
    Args:
        sequence_length: Longitud de las secuencias de entrada
        num_features: Número de características por frame
        num_classes: Número de clases a predecir
        lstm_units: Lista con número de unidades para cada capa LSTM
        attention_units: Unidades en la capa de atención
        dropout_rate: Tasa de dropout
    Returns:
        Modelo compilado
    """
    # 1. Definir la entrada
    inputs = Input(shape=(sequence_length, num_features))
    
    # 2. Capas LSTM apiladas
    x = inputs
    for i, units in enumerate(lstm_units[:-1]):
        x = LSTM(units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
    
    # Última capa LSTM (debe devolver secuencias para la atención)
    lstm_out = LSTM(lstm_units[-1], return_sequences=True)(x)
    lstm_drop = Dropout(dropout_rate)(lstm_out)
    
    # 3. Capa de atención
    attention = Attention(attention_units)
    context_vector, attention_weights = attention(lstm_drop)
    
    # 4. Dropout final
    context_drop = Dropout(dropout_rate)(context_vector)
    
    # 5. Capa Dense final para clasificación
    outputs = Dense(num_classes, activation='softmax')(context_drop)
    
    # 6. Crear y compilar modelo
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

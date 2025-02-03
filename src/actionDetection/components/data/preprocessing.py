import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple

class FeatureExtractor:
    def __init__(self, frame_size: Tuple[int, int] = (224, 224)):
        """
        Inicializa el extractor de características
        Args:
            frame_size: Tamaño al que se redimensionarán los frames (height, width)
        """
        self.frame_size = frame_size
        
        # Cargar EfficientNetB0 pre-entrenado
        self.model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            pooling='avg'  # Esto nos dará un vector de 1280 dimensiones por frame (embedding)
        )

    def preprocess_frame(self, frame):
        """
        Preprocesa un frame para EfficientNetB0
        """
        # Redimensionar
        frame = tf.image.resize(frame, self.frame_size)
        # Normalizar usando estadísticas de ImageNet
        frame = tf.keras.applications.efficientnet.preprocess_input(frame)
        return frame

    def extract_features(self, video_path: str, save_path: Path) -> None:
        """
        Extrae embeddings de cada frame del video y los guarda
        Args:
            video_path: Ruta al archivo de video
            save_path: Ruta donde guardar el archivo .npy con los embeddings
        """
        cap = cv2.VideoCapture(video_path)
        features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocesar frame
            frame = self.preprocess_frame(frame)
            
            # Extraer embedding (añadir dimensión de batch)
            frame_features = self.model(tf.expand_dims(frame, 0))
            features.append(frame_features[0])  # Shape: (1280,)
            
        cap.release()
        
        # Convertir lista de embeddings a array y guardar
        # Shape final: (num_frames, 1280)
        features = tf.stack(features).numpy()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), features)

    def process_video_directory(self, input_dir: Path, output_dir: Path):
        """
        Procesa todos los videos en un directorio
        Args:
            input_dir: Directorio con los videos
            output_dir: Directorio donde guardar las características
        """
        video_extensions = ['.mp4', '.avi', '.mov']
        
        for video_path in input_dir.rglob("*"):
            if video_path.suffix.lower() in video_extensions:
                # Mantener la estructura de carpetas
                relative_path = video_path.relative_to(input_dir)
                save_path = output_dir / relative_path.parent / f"{video_path.stem}_features.npy"
                
                try:
                    print(f"Procesando {video_path}...")
                    self.extract_features(str(video_path), save_path)
                except Exception as e:
                    print(f"Error procesando {video_path}: {str(e)}")


class SequenceGenerator:
    def __init__(self, sequence_length: int = 16, stride: int = 8):
        """
        Inicializa el generador de secuencias
        Args:
            sequence_length: Longitud de cada secuencia de frames
            stride: Número de frames de desplazamiento entre secuencias
        """
        self.sequence_length = sequence_length
        self.stride = stride

    def get_sequences(self, features_path: Path) -> np.ndarray:
        """
        Crea secuencias a partir de embeddings extraídos
        Args:
            features_path: Ruta al archivo .npy con los embeddings
        Returns:
            Array con las secuencias de embeddings
            Shape: (num_sequences, sequence_length, 1280)
        """
        # Cargar embeddings
        features = np.load(str(features_path))  # Shape: (num_frames, 1280)
        total_frames = len(features)
        
        sequences = []
        for start_idx in range(0, total_frames, self.stride):
            end_idx = start_idx + self.sequence_length
            
            if end_idx <= total_frames:
                # Secuencia completa
                sequence = features[start_idx:end_idx]
            else:
                # Padding para la última secuencia si es necesario
                sequence = np.zeros((self.sequence_length, 1280))
                remaining = total_frames - start_idx
                sequence[:remaining] = features[start_idx:]
            
            sequences.append(sequence)
        
        return np.array(sequences)  # Shape: (num_sequences, sequence_length, 1280)

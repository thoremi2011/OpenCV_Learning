import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional
from .preprocessing import SequenceGenerator

class VideoDataset(tf.keras.utils.Sequence):
    def __init__(self, 
                 root_dir: str,
                 batch_size: int = 32,
                 sequence_length: int = 16,
                 stride: int = 8,
                 shuffle: bool = True):
        """
        Dataset para cargar secuencias de características
        Args:
            root_dir: Directorio raíz con los archivos de características
            batch_size: Tamaño del batch
            sequence_length: Longitud de cada secuencia
            stride: Desplazamiento entre secuencias
            shuffle: Si True, mezcla los datos en cada época
        """
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.sequence_generator = SequenceGenerator(sequence_length, stride)
        self.shuffle = shuffle
        
        # Mapear clases a índices
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Obtener lista de archivos de características
        self.samples = self._get_samples()
        self.indexes = np.arange(len(self.samples))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _get_samples(self) -> List[Dict]:
        """
        Obtiene lista de archivos de características y sus etiquetas
        """
        samples = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for features_path in class_dir.glob("*_features.npy"):
                    samples.append({
                        'path': features_path,
                        'label': self.class_to_idx[class_dir.name]
                    })
        return samples

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indexes]

        batch_x = []
        batch_y = []

        for sample in batch_samples:
            path = sample['path']
            label = sample['label']

            # Obtener todas las secuencias posibles
            sequences = self.sequence_generator.get_sequences(path)
            # Seleccionar una secuencia aleatoria
            sequence_idx = np.random.randint(len(sequences))
            
            batch_x.append(sequences[sequence_idx])
            batch_y.append(label)

        return np.array(batch_x), tf.keras.utils.to_categorical(batch_y, num_classes=len(self.classes))

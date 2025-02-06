import tensorflow as tf
import mlflow
import os

class CustomCallback(tf.keras.callbacks.Callback):
    """Callback para guardado de checkpoints y logging con MLflow"""
    def __init__(self, patience=5, checkpoint_dir='artifacts/models', max_checkpoints=3):
        super(CustomCallback, self).__init__()
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.best_val_loss = float('inf')
        self.wait = 0
        self.best_weights = None
        self.checkpoint_files = []
        
    def on_epoch_end(self, epoch, logs=None):
        self._save_checkpoint(epoch, logs)
        self._log_to_mlflow(epoch, logs)
    
    def _save_checkpoint(self, epoch, logs):
        val_loss = logs.get('val_loss')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
            
            # Crear nuevo checkpoint
            checkpoint_path = f'{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.h5'
            self.model.save(checkpoint_path)
            self.checkpoint_files.append(checkpoint_path)
            
            # Mantener solo los últimos max_checkpoints
            if len(self.checkpoint_files) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_files.pop(0)
                try:
                    os.remove(old_checkpoint)
                except:
                    pass
                    
            print(f"\nGuardado checkpoint del epoch {epoch+1} (mejor val_loss: {val_loss:.4f})")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nNo hay mejora en {self.patience} epochs")
    
    def _log_to_mlflow(self, epoch, logs):
        mlflow.log_metrics({
            "train_loss": logs['loss'],
            "train_accuracy": logs['accuracy'],
            "val_loss": logs['val_loss'],
            "val_accuracy": logs['val_accuracy']
        }, step=epoch) 
import os
import mlflow
import mlflow.keras
from datetime import datetime
import tensorflow as tf
from ..models.callbacks import CustomCallback
from ...utils.logger import logger

class ModelTrainer:
    def __init__(self, model_params, training_params):
        self.model_params = model_params
        self.training_params = training_params
        self.setup_mlflow()
        
    def setup_mlflow(self):
        try:
            # Asegurar que la ruta es relativa a la raíz del proyecto
            mlflow_path = os.path.join("artifacts", "mlruns")
            # Crear el directorio si no existe
            os.makedirs(mlflow_path, exist_ok=True)
            # Configurar MLflow con la ruta correcta
            mlflow.set_tracking_uri(f"file:{mlflow_path}")
            mlflow.set_experiment(self.training_params['experiment_name'])
            logger.info(f"MLflow configurado correctamente en {mlflow_path}")
        except Exception as e:
            logger.error(f"Error configurando MLflow: {str(e)}")
            raise
    
    def train(self, model, train_dataset, val_dataset, test_dataset):
        """Ejecuta el entrenamiento, evalúa y guarda el modelo."""
        logger.info("Iniciando entrenamiento del modelo")
        try:
            # Iniciar un run en MLflow
            with mlflow.start_run(run_name=f"lstm_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                logger.info("Registrando parámetros en MLflow")
                mlflow.log_params(self.model_params)
                
                # 1. Configurar callbacks
                try:
                    callbacks = [
                        CustomCallback(
                            patience=self.training_params.get('patience', 7),
                            checkpoint_dir=self.training_params.get('checkpoint_dir', 'artifacts/models')
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            f"{self.training_params['checkpoint_dir']}/best_model.h5",
                            monitor='val_accuracy',
                            save_best_only=True,
                            verbose=1
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6,
                            verbose=1
                        )
                    ]
                    logger.info("Callbacks configurados correctamente")
                except Exception as e:
                    logger.error(f"Error configurando callbacks: {str(e)}")
                    raise
                
                # 2. Entrenar modelo
                try:
                    logger.info("Iniciando entrenamiento...")
                    history = model.fit(
                        train_dataset,
                        epochs=self.training_params['epochs'],
                        validation_data=val_dataset,
                        callbacks=callbacks,
                        verbose=1
                    )
                except KeyboardInterrupt:
                    logger.info("Entrenamiento interrumpido manualmente")
                    # Restaurar mejores pesos si existen
                    if callbacks[0].best_weights is not None:
                        model.set_weights(callbacks[0].best_weights)
                        logger.info("Se restauraron los mejores pesos hallados")
                    return None, None, None
                
                # 3. Evaluar modelo en test
                logger.info("Evaluando modelo en conjunto de test")
                test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
                
                # Loguear métricas en MLflow
                mlflow.log_metrics({
                    "test_accuracy": test_acc,
                    "test_loss": test_loss
                })
                
                logger.info(f"Test accuracy: {test_acc:.4f}")
                logger.info(f"Test loss: {test_loss:.4f}")
                
                # 4. Guardar modelo final en MLflow
                logger.info("Guardando modelo final en MLflow")
                mlflow.keras.log_model(model, "model")
                
                return history, test_loss, test_acc
                
        except Exception as e:
            logger.error(f"Error en el proceso de entrenamiento: {str(e)}")
            raise
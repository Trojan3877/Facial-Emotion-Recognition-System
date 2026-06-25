import os
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

def build_baseline_cnn(input_shape=(48, 48, 1), classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(classes, activation='softmax')
    ])
    return model

def run_tracked_training(epochs=10, batch_size=64, lr=0.001):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Facial_Emotion_Recognition_Architecture")
    
    # Initialize auto-logging capabilities for automatic epoch intercept captures
    mlflow.keras.autolog()
    
    with mlflow.start_run() as run:
        mlflow.log_param("optimizer_type", "Adam")
        mlflow.log_param("learning_rate", lr)
        
        # Mocking empty array allocations for syntax visualization
        # Replace with your local directory loader matrices (e.g., Fer2013)
        X_train = tf.random.normal((500, 48, 48, 1))
        y_train = tf.one_hot(tf.random.uniform((500,), maxval=7, dtype=tf.int32), 7)
        
        model = build_baseline_cnn()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        
        # Structural asset registry persistence
        mlflow.keras.log_model(model, artifact_path="emotion_model_registry")
        print(f"Compilation Complete. Run ID Registered: {run.info.run_id}")

if __name__ == "__main__":
    run_tracked_training()

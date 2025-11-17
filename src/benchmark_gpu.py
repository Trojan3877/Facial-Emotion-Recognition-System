import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Detect GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("🚀 Running on GPU.")
else:
    print("⚠ Running on CPU.")

# Load model
model = load_model("emotion_model_final.h5")

# Dummy input (48x48 grayscale)
sample = np.random.rand(1, 48, 48, 1).astype("float32")

# Benchmark
num_runs = 200
start = time.time()

for _ in range(num_runs):
    _ = model.predict(sample)

end = time.time()

total_time = end - start
avg_latency = total_time / num_runs
fps = 1.0 / avg_latency

print(f"Total runs: {num_runs}")
print(f"Avg latency per inference: {avg_latency * 1000:.3f} ms")
print(f"Throughput (FPS): {fps:.2f}")

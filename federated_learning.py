import glob
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

# Load local data
def load_local_data():
    data = {}
    for file in glob.glob("data/user_*_data.csv"):
        user_id = file.split("_")[1]
        df = pd.read_csv(file)
        data[user_id] = df
    return data

# Preprocess data for federated learning
def preprocess_data(data):
    client_data = []
    for user, df in data.items():
        features = df[["heart_rate", "sleep_duration", "activity_level"]].values
        labels = df["label"].values.reshape(-1, 1)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(10)
        client_data.append(dataset)
    return client_data

# Define the Keras model
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Wrap the Keras model for TFF
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Main function to train the federated model
def main():
    local_data = load_local_data()
    federated_data = preprocess_data(local_data)

    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    state = iterative_process.initialize()

    NUM_ROUNDS = 10
    for round_num in range(1, NUM_ROUNDS + 1):
        state, metrics = iterative_process.next(state, federated_data)
        print(f"Round {round_num}, Metrics: {metrics}")

if __name__ == "__main__":
    main()

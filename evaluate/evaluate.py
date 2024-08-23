import tensorflow as tf
from data_loader import load_data, create_generators, custom_generator, dataset_from_generator
from model import build_model

def evaluate_model():
    df = load_data()
    train_df = df.sample(frac=1, random_state=0).sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index)

    _, test_generator = create_generators(train_df, test_df)

    output_signature = (
        tf.TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    )

    test_dataset = dataset_from_generator(lambda: custom_generator(test_generator), output_signature)

    model = tf.keras.models.load_model('age_gender_model.h5')
    model.summary()

    test_steps = len(test_df) // 32
    results = model.evaluate(test_dataset, steps=test_steps)
    print(f"Test Loss: {results[0]}, Test Age MSE: {results[1]}, Test Gender Accuracy: {results[2]}")

if __name__ == "__main__":
    evaluate_model()
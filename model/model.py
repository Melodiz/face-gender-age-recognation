import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate
from tensorflow.keras.models import Model

class AgeGenderModel:
    def __init__(self, input_shape=(200, 200, 3)):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        vggnet = VGG16(include_top=False, input_shape=self.input_shape)
        vggnet.trainable = False

        output = vggnet.layers[-1].output
        flatten = Flatten()(output)

        dense1 = Dense(512, activation='relu')(flatten)
        dense2 = Dense(512, activation='relu')(flatten)

        dense3 = Dense(512, activation='relu')(dense1)
        dense4 = Dense(512, activation='relu')(dense2)

        output1 = Dense(1, activation='linear', name='age')(dense3)
        output2 = Dense(1, activation='sigmoid', name='gender')(dense4)

        model = Model(inputs=vggnet.input, outputs=[output1, output2])
        return model

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, train_data, validation_data, epochs, batch_size, callbacks=None):
        history = self.model.fit(train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        return history

    def evaluate_model(self, test_data):
        results = self.model.evaluate(test_data)
        return results

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

# Example usage:
# model = AgeGenderModel()
# model.compile_model(optimizer='adam', loss={'age': 'mse', 'gender': 'binary_crossentropy'}, metrics={'age': 'mae', 'gender': 'accuracy'})
# history = model.train_model(train_data, validation_data, epochs=10, batch_size=32, callbacks=[...])
# results = model.evaluate_model(test_data)
# model.save_model('age_gender_model.h5')
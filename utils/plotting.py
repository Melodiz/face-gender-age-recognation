import matplotlib.pyplot as plt

def plot_statistics(history):
    plt.ion()  # Turn on interactive mode
    plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(history.history['age_mse'])
    plt.plot(history.history['val_age_mse'])
    plt.title('Age MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(history.history['gender_accuracy'], label='train accuracy')
    plt.plot(history.history['val_gender_accuracy'], label='val accuracy')
    plt.title('Gender Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
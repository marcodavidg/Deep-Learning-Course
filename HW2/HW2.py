import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical

def plot_feature_maps(model, input_image, number_layers, section1 = True):
    print("len ==", input_image.shape)
    """Plot the different feature maps of the model. Iterates through all the convolutional layers, plotting their outputs."""
    if section1:
        # Print the image used
        plt.imshow(input_image.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

        for layer in range(number_layers):
            output_layer = model.layers[layer * 2].output
            feature_map_model = tf.keras.models.Model(inputs=model.inputs, outputs=output_layer)
            feature_maps = feature_map_model.predict(input_image)
            for filter in range(feature_maps.shape[-1]):
                plt.subplot(4, 2 ** (2 + layer) / 4, filter + 1)
                image_array = feature_maps[:, :, :, filter]
                image = image_array.reshape((feature_maps.shape[1], feature_maps.shape[2]))
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.show()
    else:
        plt.imshow(input_image.reshape(32, 32, 3))
        plt.axis('off')
        plt.show()
        for layer in range(number_layers):
            output_layer = model.layers[layer * 2].output
            feature_map_model = tf.keras.models.Model(inputs=model.inputs, outputs=output_layer)
            feature_maps = feature_map_model.predict(input_image)
            for filter in range(8):
                plt.subplot(4, 2, filter + 1)
                image_array = feature_maps[:, :, :, filter]
                image = image_array.reshape((feature_maps.shape[1], feature_maps.shape[2]))
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.show()


def section1():  # SECTION 1 OF THE ASSIGNMENT

    # -------- LOAD DATA --------
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap='gray')
    #     # The CIFAR labels happen to be arrays,
    #     # which is why you need the extra index
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    # -------- BUILD MODEL --------
    model = models.Sequential()
    model.add(layers.Conv2D(4, (5, 5), activation='relu', kernel_regularizer='l2', strides=(1, 1), padding="same", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (5, 5), activation='relu', kernel_regularizer='l2', strides=(1, 1), padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_regularizer='l2', strides=(1, 1), padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_regularizer='l2'))
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, kernel_regularizer='l2'))

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # -------- TRAIN THE MODEL --------
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # -------- PLOT INCORRECT RESULTS --------
    predictions = model.predict(test_images)
    classes = np.argmax(predictions, axis=1) # Get final prediction of the model
    comparison = classes - test_labels
    incorrect = 5
    correct = 0
    for i, sample in enumerate(test_images):
        if comparison[i] == 0:
            if  correct < 5:
                correct += 1
                plt.subplot(2, 5, correct)
                plt.imshow(sample, cmap='gray')
                plt.axis('off')
                plt.title("Correctly predicted label: " + str(classes[i]))
        else:
            incorrect += 1
            plt.subplot(2, 5, incorrect)
            plt.imshow(sample, cmap='gray')
            plt.axis('off')
            plt.title("True label: " + str(test_labels[i]) + " Predicted: " + str(classes[i]))
            if incorrect == 10:
                break
    plt.show()

    # -------- PLOT WEIGHTS AND BIASES --------
    weights_conv1 = model.layers[0].weights[0].numpy()
    bias_conv1 = model.layers[0].weights[1].numpy()
    weights_conv2 = model.layers[2].weights[0].numpy()
    bias_conv2 = model.layers[2].weights[1].numpy()
    weights_conv3 = model.layers[4].weights[0].numpy()
    bias_conv3 = model.layers[4].weights[1].numpy()
    weights_fc1 = model.layers[6].weights[0].numpy()
    bias_fc1 = model.layers[6].weights[1].numpy()
    weights_fc2 = model.layers[7].weights[0].numpy()
    bias_fc2 = model.layers[7].weights[1].numpy()

    plt.subplot(2, 2, 1)
    plt.hist(weights_conv1.flatten(), density=False, bins=50)
    plt.ylabel('First Conv Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 2)
    plt.hist(bias_conv1.flatten(), density=False, bins=50)
    plt.ylabel('First Conv Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 3)
    plt.hist(weights_conv2.flatten(), density=False, bins=50)
    plt.ylabel('Second Conv Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 4)
    plt.hist(bias_conv2.flatten(), density=False, bins=50)
    plt.ylabel('Second Conv Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])
    plt.show()

    plt.subplot(2, 2, 1)
    plt.hist(weights_conv3.flatten(), density=False, bins=50)
    plt.ylabel('Third Conv Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 2)
    plt.hist(bias_conv3.flatten(), density=False, bins=50)
    plt.ylabel('Third Conv Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 3)
    plt.hist(weights_fc1.flatten(), density=False, bins=50)
    plt.ylabel('First FC Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 4)
    plt.hist(bias_fc1.flatten(), density=False, bins=50)
    plt.ylabel('First FC Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])
    plt.show()

    plt.subplot(2, 2, 1)
    plt.hist(weights_fc2.flatten(), density=False, bins=50)
    plt.ylabel('Output FC Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 2)
    plt.hist(bias_fc2.flatten(), density=False, bins=50)
    plt.ylabel('Output FC Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])
    plt.show()

    # -------- PLOT ACCURACY AND LOSS --------
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.title('Model Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Cross Entropy')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_loss, test_acc)

    # -------- PLOT FEATURE MAPS --------
    input_image = test_images[4:4 + 1, :, :, :]
    plot_feature_maps(model, input_image, 3)


def section2():  # SECTION 2 OF THE ASSIGNMENT

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize the arrays of the images
    train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0



    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer='l2', strides=(1, 1), padding="same", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer='l2', strides=(1, 1), padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2', strides=(1, 1), padding="same"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_regularizer='l2'))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer='l2'))
    model.add(layers.Dense(10, kernel_regularizer='l2'))

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # -------- TRAIN THE MODEL --------
    history = model.fit(train_images, train_labels, epochs=15, batch_size=64,
                        validation_data=(test_images, test_labels))

    # -------- PLOT INCORRECT RESULTS --------
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    predictions = model.predict(test_images)
    incorrect = 5
    correct = 0
    for i, sample in enumerate(test_images):
        predicted = np.argmax(predictions[i], axis=0)
        if  test_labels[i][0] == predicted:
            if correct < 5:
                correct += 1
                plt.subplot(2, 5, correct)
                plt.imshow(sample, cmap='gray')
                plt.axis('off')
                plt.title("Correctly predicted label: " + class_names[predicted])
        else:
            incorrect += 1
            plt.subplot(2, 5, incorrect)
            plt.imshow(sample, cmap='gray')
            plt.axis('off')
            plt.title("True label: " + class_names[test_labels[i][0]] + " Predicted: " + class_names[predicted])
            if incorrect == 10:
                break
    plt.show()

    # -------- PLOT WEIGHTS AND BIASES --------
    weights_conv1 = model.layers[0].weights[0].numpy()
    bias_conv1 = model.layers[0].weights[1].numpy()
    weights_conv2 = model.layers[2].weights[0].numpy()
    bias_conv2 = model.layers[2].weights[1].numpy()
    weights_conv3 = model.layers[4].weights[0].numpy()
    bias_conv3 = model.layers[4].weights[1].numpy()
    weights_fc1 = model.layers[6].weights[0].numpy()
    bias_fc1 = model.layers[6].weights[1].numpy()
    weights_fc2 = model.layers[7].weights[0].numpy()
    bias_fc2 = model.layers[7].weights[1].numpy()
    weights_fc3 = model.layers[8].weights[0].numpy()
    bias_fc3 = model.layers[8].weights[1].numpy()

    plt.subplot(2, 2, 1)
    plt.hist(weights_conv1.flatten(), density=False, bins=50)
    plt.ylabel('First Conv Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 2)
    plt.hist(bias_conv1.flatten(), density=False, bins=50)
    plt.ylabel('First Conv Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 3)
    plt.hist(weights_conv2.flatten(), density=False, bins=50)
    plt.ylabel('Second Conv Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 4)
    plt.hist(bias_conv2.flatten(), density=False, bins=50)
    plt.ylabel('Second Conv Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])
    plt.show()

    plt.subplot(2, 2, 1)
    plt.hist(weights_conv3.flatten(), density=False, bins=50)
    plt.ylabel('Third Conv Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 2)
    plt.hist(bias_conv3.flatten(), density=False, bins=50)
    plt.ylabel('Third Conv Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 3)
    plt.hist(weights_fc1.flatten(), density=False, bins=50)
    plt.ylabel('First FC Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 4)
    plt.hist(bias_fc1.flatten(), density=False, bins=50)
    plt.ylabel('First FC Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])
    plt.show()

    plt.subplot(2, 2, 1)
    plt.hist(weights_fc2.flatten(), density=False, bins=50)
    plt.ylabel('Second FC Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])
    plt.subplot(2, 2, 2)
    plt.hist(bias_fc2.flatten(), density=False, bins=50)
    plt.ylabel('Second FC Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 3)
    plt.hist(weights_fc3.flatten(), density=False, bins=50)
    plt.ylabel('Output FC Layer Weights')
    plt.xlabel('Data')
    plt.xlim([-1, 1])

    plt.subplot(2, 2, 4)
    plt.hist(bias_fc3.flatten(), density=False, bins=50)
    plt.ylabel('Output FC Layer Biases')
    plt.xlabel('Data')
    plt.xlim([-1, 1])
    plt.show()
    #
    # -------- PLOT ACCURACY AND LOSS --------
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Cross Entropy')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.show()
    #
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_loss, test_acc)

    # # -------- PLOT FEATURE MAPS --------
    print("----------------", test_images.shape)
    input_image = test_images[4:5, :, :, :]
    plot_feature_maps(model, input_image, 3, section1 = False)


# section1()
section2()

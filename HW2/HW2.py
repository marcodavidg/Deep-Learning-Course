import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
import theano
def section1(): # Section 1 of the assignment
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    class_names = ['zero','one','two','three','four','five','six','seven','eight','nine']
    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap='gray')
    #     # The CIFAR labels happen to be arrays, 
    #     # which is why you need the extra index
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    model = models.Sequential()
    model.add(layers.Conv2D(4, (3, 3), activation='relu', strides=(1, 1), input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', strides=(1, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, kernel_regularizer='l2', activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    history = model.fit(train_images, train_labels, epochs=1, 
                        validation_data=(test_images, test_labels))




    # -------- Plot incorrect results --------
    predictions = model.predict(test_images)
    classes = np.argmax(predictions, axis = 1)
    
    comparison = classes - test_labels
    mala = None
    print(classes)
    incorrect = 0
    for i, sample in enumerate(test_images):
        if comparison[i] == 0:
            next
        else:
            incorrect += 1
            plt.subplot(2, 5, incorrect)
            plt.imshow(sample)
            plt.axis('off')
            plt.title("True label: " + str(test_labels[i]) + " Predicted: " + str(classes[i]))
            mala = i
            if incorrect == 10:
                break
    
    plt.show()


    # -------- Plot weights and biases --------
    # weights_layer1 = model.layers[0].weights[0].numpy()
    # bias_layer1 = model.layers[0].weights[1].numpy()
    # weights_layer2 = model.layers[2].weights[0].numpy()
    # bias_layer2 = model.layers[2].weights[1].numpy()
    # weights_layer3 = model.layers[4].weights[0].numpy()
    # bias_layer3 = model.layers[4].weights[1].numpy()
    # weights_layer4 = model.layers[6].weights[0].numpy()
    # bias_layer4 = model.layers[6].weights[1].numpy()
    # weights_layer4 = model.layers[7].weights[0].numpy()
    # bias_layer4 = model.layers[7].weights[1].numpy()


    # plt.subplot(2, 3, 1)
    # plt.hist(weights_layer1.flatten(), density=False, bins=50)
    # plt.ylabel('First Conv Layer Weights')
    # plt.xlabel('Data')

    # plt.subplot(2, 3, 2)
    # plt.hist(bias_layer1.flatten(), density=False, bins=50)
    # plt.ylabel('First Conv Layer Biases')
    # plt.xlabel('Data')

    # plt.subplot(2, 3, 4)
    # plt.hist(weights_layer2.flatten(), density=False, bins=50)
    # plt.ylabel('Second Conv Layer Weights')
    # plt.xlabel('Data')

    # plt.subplot(2, 3, 5)
    # plt.hist(bias_layer2.flatten(), density=False, bins=50)
    # plt.ylabel('Second Conv Layer Biases')
    # plt.xlabel('Data')

    # plt.subplot(2, 3, 3)
    # plt.hist(weights_layer3.flatten(), density=False, bins=50)
    # plt.ylabel('Third Conv Layer Weights')
    # plt.xlabel('Data')

    # plt.subplot(2, 3, 6)
    # plt.hist(bias_layer3.flatten(), density=False, bins=50)
    # plt.ylabel('Third Conv Layer Biases')
    # plt.xlabel('Data')
    # plt.show()


    # # -------- Plot accuracy and loss --------
    # plt.subplot(2, 1, 1)
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')

    # plt.subplot(2, 1, 2)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.ylabel('Cross Entropy')
    # plt.ylim([0,1.0])
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')

    # plt.show()

    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    # print(test_loss, test_acc)


    # -------- Plot different feature maps --------
    output_layer = model.layers[0].output

    input_image = test_images[incorrect:incorrect+1,:,:,:]
    print(input_image.shape)


    feature_map_model = tf.keras.models.Model(inputs=model.inputs, outputs=output_layer)
    feature_maps = feature_map_model.predict(input_image)
    print(f"The shape of the 0 is =======>> {feature_maps.shape}")
    for x in range(feature_maps.shape[-1]):
        plt.subplot(2, 2, x + 1)
        image_array = feature_maps[:,:,:,x]
        image = image_array.reshape((26,26))
        plt.imshow(image, cmap='gray')
    plt.show()

    output_layer = model.layers[2].output
    feature_map_model = tf.keras.models.Model(inputs=model.inputs, outputs=output_layer)
    feature_maps = feature_map_model.predict(input_image)
    print(f"The shape of the 0 is =======>> {feature_maps.shape}")
    for x in range(feature_maps.shape[-1]):
        plt.subplot(4, 2, x + 1)
        image_array = feature_maps[:,:,:,x]
        image = image_array.reshape((feature_maps.shape[1],feature_maps.shape[2]))
        plt.imshow(image, cmap='gray')
    plt.show()

    output_layer = model.layers[4].output
    feature_map_model = tf.keras.models.Model(inputs=model.inputs, outputs=output_layer)
    feature_maps = feature_map_model.predict(input_image)
    print(f"The shape of the 0 is =======>> {feature_maps.shape}")
    for x in range(feature_maps.shape[-1]):
        plt.subplot(4, 4, x + 1)
        image_array = feature_maps[:,:,:,x]
        image = image_array.reshape((feature_maps.shape[1],feature_maps.shape[2]))
        plt.imshow(image, cmap='gray')
    plt.show()



def section2(): # Section 2 of the assignment

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i])
    #     # The CIFAR labels happen to be arrays, 
    #     # which is why you need the extra index
    #     plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))



    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


    print(test_acc)

section1()
# section2()
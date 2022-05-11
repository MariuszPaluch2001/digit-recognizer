import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import argparse
import tensorflow

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tensorflow.cast(image, tensorflow.float32) / 255., label

def train_model(num_epochs : int):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )


    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tensorflow.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tensorflow.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tensorflow.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tensorflow.data.AUTOTUNE)

    model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
    tensorflow.keras.layers.Dense(units = 128, activation='relu'),
    tensorflow.keras.layers.Dense(units = 128, activation='relu'),
    tensorflow.keras.layers.Dense(units = 10, activation=tensorflow.nn.softmax)
    ])

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(0.001),
        loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tensorflow.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
    )

    model.save('hw_digit_recognizer.model')

    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse bool")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=False)
    parser.add_argument('--epochs', default=3, type=int, nargs='?')


    args = parser.parse_args()


    if args.train:
        model = train_model(args.epochs)
    else:
        model = tensorflow.keras.models.load_model('hw_digit_recognizer.model')

    for i in range(1, 10):
        img = cv2.imread(f'digits/{i}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Number on image is probably : {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
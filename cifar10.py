import numpy as np
import pandas as pd
import random

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import sys
sys.path.append("/Users/gabrielepadovani/Desktop/Università/mia/")

from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data


NUM_CLASSES = 10
WIDTH = 28
HEIGHT = 28
CHANNELS = 1

SHADOW_DATASET_SIZE = 1000
ATTACK_TRAIN_DATASET_SIZE = 100
ATTACK_TEST_DATASET_SIZE = 1000

NUMBER_OF_OUTLIERS = int(SHADOW_DATASET_SIZE * 0.1)

FLAGS = flags.FLAGS
flags.DEFINE_integer("target_epochs", 1, "Number of epochs to train target and shadow models.")
flags.DEFINE_integer("shadow_epochs", 2, "Number of epochs to train target and shadow models.")
flags.DEFINE_integer("attack_epochs", 1, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 10, "Number of MODELS, NOT EPOCHS to train attack models.")


def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # for i, y in enumerate(y_train): 
    #     if random.randint(0, 10) > 9: 
    #         y_train[i] = (y +5 ) % NUM_CLASSES

    # for i, y in enumerate(y_test): 
    #     if random.randint(0, 10) > 9: 
    #         y_test[i] = (y +5 ) % NUM_CLASSES

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(32, activation="relu"))
    # model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    # model.add(layers.Dense(15, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

RESULTS = [] 
def demo(argv):
    global RESULTS
    del argv  # Unused.
    
    runs = 10
    for _ in range(runs): 

        (X_train, y_train), (X_test, y_test) = get_data()

        # Train the target model.
        print("Training the target model...")
        target_model = target_model_fn()
        target_model.fit(
            X_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
        )

        # Train the shadow models.
        smb = ShadowModelBundle(
            target_model_fn,
            shadow_dataset_size=SHADOW_DATASET_SIZE,
            num_models=FLAGS.num_shadows,
        )

        # We assume that attacker's data were not seen in target's training.
        attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
            X_test, y_test, test_size=0.1
        )

        print("Training the shadow models...")
        X_shadow, y_shadow = smb.fit_transform(
            attacker_X_train,
            attacker_y_train,
            fit_kwargs=dict(
                epochs=FLAGS.shadow_epochs,
                verbose=True,
                validation_data=(attacker_X_test, attacker_y_test),
            ),
        )

        # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
        amb = AttackModelBundle(attack_model_fn, num_classes=2)

        # Fit the attack models.
        print("Training the attack models...")
        indices = list(range(len(X_shadow)))
        random.shuffle(indices)
        X_shadow = X_shadow[indices]
        y_shadow = y_shadow[indices]

        X_shadow = X_shadow[:ATTACK_TRAIN_DATASET_SIZE]
        y_shadow = y_shadow[:ATTACK_TRAIN_DATASET_SIZE]
        
        amb.fit(X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=False))

        # Prepare examples that were in the training, and out of the training.
        print("START TESTING")
        data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
        data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

        print("prepare_attack_data")
        # Compile them into the expected format for the AttackModelBundle.
        attack_test_data, real_membership_labels = prepare_attack_data(
            target_model, data_in, data_out
        )

        indices = list(range(len(attack_test_data)))
        random.shuffle(indices)
        attack_test_data = attack_test_data[indices]
        real_membership_labels = real_membership_labels[indices]

        # Compute the attack accuracy.
        print("predict")
        attack_guesses = amb.predict(attack_test_data, real_membership_labels)
        attack_accuracy = np.mean(attack_guesses == real_membership_labels)

        f1 = f1_score(attack_guesses, real_membership_labels)

        print("ACC", attack_accuracy)
        print("F1", f1)

        RESULTS.append({"acc": attack_accuracy, "f1": f1})

    pd.DataFrame(RESULTS, index=range(len(RESULTS))).to_csv("results.csv")

if __name__ == "__main__":
    app.run(demo)





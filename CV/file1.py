import random

from model import Model
import emnist
import numpy as np


def main():
    model = Model()

    images_train, labels_train = emnist.extract_training_samples('balanced')
    images_test, labels_test = emnist.extract_test_samples('balanced')
    a = np.random.randint(0, labels_train.shape[0])
    b = np.random.randint(0, labels_test.shape[0])
    print(f"train pred : {model.predict(images_train[a])}, "
          f"true : {model.dict_labels[labels_train[a]]}")
    print(f"test pred : {model.predict(images_test[b])}, "
          f"true : {model.dict_labels[labels_test[b]]}")


if __name__ =="__main__":

    main()
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import csv


# Reads the training and testing data from a CSV File
def read_file():
    with open('diabetes .csv') as csv_file:
        diabetes_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        next(diabetes_reader, None)  # skip the headers
        np_array = np.array(list(diabetes_reader), dtype=np.float32)
        return np_array


# Splits up the training and testing data

def split_array(np_array):
    np.random.shuffle(np_array)
    test, train = np.split(np_array, [1], axis=0)
    x_test, y_test = np.split(test, [-1], axis=1)
    x_train, y_train = np.split(train, [-1], axis=1)
    return x_train, y_train, x_test, y_test


# main execution function
def run():
    np_array = read_file()
    x_train, y_train, x_test, y_test = split_array(np_array)

    print('x_train SHAPE:', x_train.shape)

    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=500, batch_size=16)

    loss, accuracy = model.evaluate(x_train, y_train)
    print('Accuracy:', round(100 * accuracy, 2), ', % Loss:', round(100 * loss, 2), '%')

    for _ in range(10):
        prediction = np.round(model.predict(x_test))
        print('expected:', y_test, 'prediction:', prediction)

    print("Model has learned based on the input")
    print("Please enter the number of pregnancies you've had ", end='>> ')
    pregnancies = np.float32(input())
    print("Please enter your glucose level ", end='>> ')
    glucose = np.float32(input())
    print("Please enter your current blood pressure ", end='>> ')
    blood_pressure = np.float32(input())
    print("Please enter your current skin thickness ", end='>> ')
    skin_thickness = np.float32(input())
    print("Please enter your current insulin level ", end='>> ')
    insulin_level = np.float32(input())
    print("Please enter your current BMI ", end='>> ')
    bmi = np.float32(input())
    print("Please enter your current Diabetes Pedigree Function ", end='>> ')
    diabetes_pedigree_function = np.float32(input())
    print("Please enter your current age ", end='>> ')
    age = np.float32(input())

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin_level, bmi, diabetes_pedigree_function, age]])
    print(pregnancies)
    prediction = np.round(model.predict(input_data))

    if prediction == 1:
        print("The AI believes you have diabetes.Go get checked!!!! ")
    else:
        print("The AI believes you don't have diabetes")


run()

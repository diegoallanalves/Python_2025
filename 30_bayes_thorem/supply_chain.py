import keras
import pandas as pd
import scikitlearn

class supplychain
    def prediction():

        input_data =  pd.read_csv('C:\\Users\\Diego Alves\\Desktop\\Data_sets\\ChargingStations_GB.csv',40)
        x, y, x_test, y_test = scikitlearn.split(input_data)
        model = keras.Sequentia();
        model.add(Dense(units=64, activation='relu', input_dim=100))
        model.add(Dense(units=10, activation='softmax'))
        model.add(Dense(units=64, activation='relu', input_dim=100))
        model.add(Dense(units=10, activation='softmax'))

        model.compile(loss='categorical',
                      optimizer='sgd',
                      metrics=['accuracy'])
        model.fit(x, y, epochs=5, batch_size=32)

        classes = model.predict(x_test, batch_size=128)

        print(classes)

import tensorflow.keras as keras
from preproccessing import *
from melody_generator import *

# model variables
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 30  # 40 to 100
BATCH_SIZE = 128


# Base LSTM with functional API of keras
def build_LSTM_base_model(output_units, num_units, loss, learning_rate):
    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.3)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train_LSTM_base_model(save_dir, output_units, seq_len, single_file, map_path, num_units=NUM_UNITS, loss=LOSS,
                          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    # generate train seq
    x_train, y_train = generate_train_seq(seq_len, single_file, map_path)
    # build the model of network
    model = build_LSTM_base_model(output_units, num_units, loss, learning_rate)
    print(x_train.shape)
    # train model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # save the model
    model.save(save_dir)

    return history

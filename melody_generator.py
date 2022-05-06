import tensorflow.keras as keras
import json
import numpy as np
import music21
from preproccessing import *
import matplotlib.pyplot as plt


class Melody_Gen:

    def __init__(self, model_path, seq_len, mapp_path):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(mapp_path, "r") as fp:
            self.map = json.load(fp)

        self.start_symbols = ["/"] * seq_len

    def generate_melody(self, seed, num_steps, max_sequence_length, temp):
        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self.start_symbols + seed

        # map seed to int
        seed = [self.map[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit the seed
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot = keras.utils.to_categorical(seed, num_classes=len(self.map))
            # size of array is [1, max_sequence_length, num of symb in map]
            onehot = onehot[np.newaxis, ...]

            # make a prediction of note
            probab = self.model.predict(onehot)[0]

            note_int = self.sample_with_temp(probab, temp)

            # update seed
            seed.append(note_int)

            # map int to original value
            note_symbol = [k for k, v in self.map.items() if v == note_int][0]

            # check if generator thinks that melody is end
            if note_symbol == '/':
                break

            # update the melody
            melody.append(note_symbol)

        return melody

    def sample_with_temp(self, probab, temp):
        # the temp stands for:
        # if temp-> infinity the probabilities become similar and we literally chosen random one output
        # if temp-> 0 the hightest probability becomes one
        # if temp = 1 temp not occure influence on probab
        if (temp == 0):
            temp = 0.05
        predictions = np.log(probab) / temp
        # literally the softmax
        probab = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probab))
        index = np.random.choice(choices, p=probab)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.midi"):
        # create a music21 stream
        stream = music21.stream.Stream()

        # parse all the symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest or at the end of melody list

            if symbol != "-" or i + 1 == len(melody):

                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    # handle rest
                    if start_symbol == "R":
                        music21_event = music21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        music21_event = music21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(music21_event)

                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a duration sighn
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)

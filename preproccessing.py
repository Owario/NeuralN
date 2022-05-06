import os
import music21
import json
import numpy as np
import tensorflow.keras as keras


# return songs converted in music21 Score
def load_songs_from_base_dataset(dataset_path):
    songs = []

    #  go through all the files in dataset and load them with music21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                # music21 library can convert kern, MIDI, MusicXml in special music21 type Score afterwards in kern, MIDI etc
                song = music21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs


# check if chosen song acceptable with note duration, returns true or false
def if_song_have_acceptable_duration(song, acceptable_duration):
    # flat convert all Score data in list

    for note_ in song.flat.notesAndRests:
        # print(note_, note_.duration)
        if note_.duration.quarterLength not in acceptable_duration:
            return False

    return True


# encode chosen song to the corresponding time series, returns encoded song
def encode_song_to_time_series_representation(song, time_step=0.25):
    #  example of encoding:
    #  note = 50, duration = 1.5 -> convert to minimal duration type (0.25) [50, "-", "-", "-", "-", "-"]
    encoded_song = []

    for note_ in song.flat.notesAndRests:
        # notes part
        if isinstance(note_, music21.note.Note):
            curr_symbol = note_.pitch.midi  # 50
        # rests part
        elif isinstance(note_, music21.note.Rest):
            curr_symbol = "R"
        # after getting notes and rests we should convert it to the chosen type: time series representation
        steps = int(note_.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(curr_symbol)
            else:
                encoded_song.append("-")

    # convert song to a str
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song


# transpose all notes of song to the one chosen key for major and minor, return converted song
def transpose_to_the_one_key(song, pitch_major, pitch_minor):
    #  getting key from the original song
    parts = song.getElementsByClass(music21.stream.Part)
    measure_of_part0 = parts[0].getElementsByClass(music21.stream.Measure)
    key = measure_of_part0[0][4]

    #  if key is not marked, trying to estimate key using music21 library
    if not isinstance(key, music21.key.Key):
        key = song.analyze("key")

    #  get the interval for transposition
    if key.mode == "major":
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch(pitch_major))
    if key.mode == "minor":
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch(pitch_minor))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song


# find all songs in chosen path convert it to int and saves it in diffirent files
def preprocess(dataset_path, save_dir, major_pitch, minor_pitch, accept_durations):
    pass
    # load the folk songs (using music21)
    print("Load songs")
    songs = load_songs_from_base_dataset(dataset_path)
    print(f"Loaded {len(songs)} songs")

    iteration = 0
    for song in songs:

        # filter out songs that have non-acceptable durations
        if not if_song_have_acceptable_duration(song, accept_durations):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose_to_the_one_key(song, major_pitch, minor_pitch)

        # encode songs with music time series representation (2-step is notes with they durations)
        encoded_song = encode_song_to_time_series_representation(song)

        # save songs to text file
        path_for_save = os.path.join(save_dir, str(iteration))
        with open(path_for_save, "w") as file_path:
            file_path.write(encoded_song)

        iteration += 1

    print(f"Data succesfully converted and saved")


def load_file_data(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(data_path, single_file_dataset_path, seq_length):
    songs = ""
    song_delimiter = "/ " * seq_length
    # load encoded songs and add delimiters fo single file dataset
    for path, subdir, files in os.walk(data_path):
        for i, file in enumerate(files):
            file_path = os.path.join(path, file)
            song = load_file_data(file_path)
            songs = songs + song + " " + song_delimiter
            # if i != len(files) - 1:
            #     songs = songs + song + " " + song_delimiter
            # else:
            #     songs = songs + song + " "

    songs = songs[:-1]

    with open(single_file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


# create map for loaded songs and save it
def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


# returns songs converted in int using existing mapping file
def convert_songs_to_integer(songs, map_path):
    int_songs = []

    # load the mappings
    with open(map_path, "r") as fp:
        map = json.load(fp)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(map[symbol])

    return int_songs


# create train seq certain length by using created dataset file which is consisting of int samples
def generate_train_seq(sequence_length, single_file, map_path):
    # load songs and convert them to integer
    songs = load_file_data(single_file)
    int_songs = convert_songs_to_integer(songs, map_path)

    # generate training sequences
    y_train = []
    x_train = []
    num_of_seq = len(int_songs) - sequence_length
    for i in range(num_of_seq):
        x_train.append(int_songs[i:i + sequence_length])
        y_train.append(int_songs[i + sequence_length])
        # delete bad samples (/////)

    # one-hot encoding
    map_size = len(set(int_songs))
    x_train = keras.utils.to_categorical(x_train, num_classes=map_size)
    y_train = np.array(y_train)

    return x_train, y_train


def create_single_file_dataset(data_path, single_file_dataset_path, seq_length):
    songs = ""
    song_delimiter = "/ " * seq_length
    # load encoded songs and add delimiters fo single file dataset
    for path, subdir, files in os.walk(data_path):
        for i, file in enumerate(files):
            file_path = os.path.join(path, file)
            song = load_file_data(file_path)
            songs = songs + song + " " + song_delimiter
            # if i != len(files) - 1:
            #     songs = songs + song + " " + song_delimiter
            # else:
            #     songs = songs + song + " "

    songs = songs[:-1]

    with open(single_file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

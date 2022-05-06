import tensorflow.keras as keras
from preproccessing import *
from melody_generator import *
from models import *
import matplotlib.pyplot as plt

# for CUDA calc
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# model variables
SAVE_MODEL_PATH = "model.h5"

# preproccessing variables
DATASET_PATH = "deutschl"
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
PITCH_TO_CONVERT_FOR_MAJOR = "C"
PITCH_TO_CONVERT_FOR_MINOR = "A"
SAVE_DIR = "dataset"
MAPP_PATH = "map.json"
SINGLE_FILE_DATASET_DIR = "dataset file"
SEQUENCE_LENGTH = 64  # 4 4 signature music sample


def convert_krn_dataset_to_txt_dataset(dataset_path=DATASET_PATH, save_dir=SAVE_DIR,
                                       major_pitch=PITCH_TO_CONVERT_FOR_MAJOR, minor_pitch=PITCH_TO_CONVERT_FOR_MINOR,
                                       accept_durations=ACCEPTABLE_DURATIONS):
    # convert krn songs stored in dataset_path to int songs with txt type and saves them in save_dir
    preprocess(dataset_path, save_dir, major_pitch, minor_pitch, accept_durations)


def create_one_dataset_file(dataset_path=SAVE_DIR, single_file_dataset_file=SINGLE_FILE_DATASET_DIR,
                            seq_length=SEQUENCE_LENGTH, mapp_path=MAPP_PATH):
    # create one dataset file
    songs = create_single_file_dataset(dataset_path, single_file_dataset_file, seq_length)
    # create corresponding mapping file
    create_mapping(songs, mapp_path)


def create_dataset_of_int(dataset_path="deutschl_part1"):
    save_dir = dataset_path + "_converted"
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    print(f"songs will saved in {save_dir}")
    convert_krn_dataset_to_txt_dataset(dataset_path, save_dir)


def calculate_len_of_map(mapp_path=MAPP_PATH):
    with open(MAPP_PATH, "r") as fp:
        map = json.load(fp)
    print(f"songs in dataset consist of {len(map)} different notes")
    return len(map)


def plot_hist(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train acc")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy during train")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error val")

    plt.show()



if __name__ == "__main__":
    dataset_path = "deutschl_part1"
    save_dir = dataset_path + "_converted"
    # create_dataset_of_int(dataset_path)
    # create_one_dataset_file(save_dir)
    output_units = calculate_len_of_map()

    history = train_LSTM_base_model(SAVE_MODEL_PATH, output_units, SEQUENCE_LENGTH, SINGLE_FILE_DATASET_DIR, MAPP_PATH)

    # plot accuracy and error
    plot_hist(history)

    # MelodyGenerator = Melody_Gen(SAVE_MODEL_PATH, SEQUENCE_LENGTH, MAPP_PATH)
    #
    #
    # seed = "55"
    # melody = MelodyGenerator.generate_melody(seed, 1000, SEQUENCE_LENGTH, 0)
    # print(melody)
    # MelodyGenerator.save_melody(melody)

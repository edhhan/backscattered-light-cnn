import json
import os
import platform

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


def to_json(nb_photon):

    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, nb_photon)
    files = os.listdir(data_path)

    for file in files:
        splited_file = file.split(".")
        if len(file.split(".")) > 2:
            os.rename(r'data/' + file,
                      r'data/' + splited_file[0] + "." + splited_file[1] + "." + splited_file[2])
        else:
            filename = file.split(".txt")[0]
            os.rename(r'data/' + file, r'data/' + filename + ".json")


def load_data(nb_photon):
    """
    Function that reads multiple .json files
    :return: the actual return is a .npy saved file in the main repos
    """

    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, nb_photon)
    files = os.listdir(data_path)
    data = []

    for file in files:

        try:
            file_path = os.path.join(data_path, file)

            with open(file_path) as json_data:
                example_dict = json.load(json_data)  # creates a python dictionary

            input_intensity = example_dict.get("detected")
            input_flatten = np.array(input_intensity, dtype=np.float)

            # Correct version for 3 labels
            thickness = float(example_dict.get("epaisseur"))
            if 8 <= thickness < 12:
                label = np.array([1, 0, 0])
            elif 12 <= thickness < 16:
                label = np.array([0, 1, 0])
            elif 16 <= thickness <= 20:
                label = np.array([0, 0, 1])


            # Add noise
            signal_shot_poisson, signal_temp = gen_noise(input_flatten, float(nb_photon), 1100, 1225)

            data.append([signal_temp, label])

        except Exception as e:
            print(file, str(e))

    random.shuffle(data)
    np.save("data" + "_" + nb_photon + ".npy", data)


def reformat(data_intensity):
    """
    Utily function that reshapes the .txt files into the proper 2D matrix-image
        -Note : the .txt files are saved as a flatten 2D-array
    """
    side_length = 35
    data_reshape = np.zeros((35, 35))
    for i in range(0, 35):
        data_reshape[i][0:(side_length - 1)] = data_intensity[i * (side_length - 1):(i + 1) * (side_length - 1)]

    return data_reshape.reshape(35, 35)


class CustomDataSetLoaderCNN(Dataset):
    """
    Special class to help us utilize the DataLoader function provided by PyTorch library for CNN
    """

    def __init__(self, data, input_size=35, label_size=3):

        self.X_data = []
        self.Y_data = []
        counter = 0  # counts the number of valid examples

        for i, iterable_data in enumerate(data):
            try:

                # Split input
                self.X_data.append(torch.from_numpy(iterable_data[0] * 1000).view(-1, input_size, input_size))
                self.Y_data.append(torch.from_numpy(iterable_data[1]).view(-1, 1, label_size))
                counter += 1

            except Exception as e:
                print(i, str(e))

        self.len = counter

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return self.len


def preprocess(nb_photon, batch_size):
    """
    Prepares the data before training
    """

    # Load data from JSON file if necessary
    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, "data")
    os.chdir(data_dir)

    # If data is already unzipped and ready to be used, then load directly
    try:
        load_data(nb_photon)
    # Unzip and generate a .npy data file
    except Exception as e:
        print(e, "unzipping data :" + nb_photon)
        if platform.system() == "Windows":
            os.system("tar -xf " + nb_photon + ".zip")
            load_data(nb_photon)  # Note current directory is data
        else:
            os.system("unzip " + nb_photon + ".zip")
            load_data(nb_photon)  # Note current directory is data

    # Import pre-loaded data from .npy file (in data folder)
    data_load = np.load("data" + "_" + nb_photon + ".npy", allow_pickle=True)
    os.chdir(working_dir)

    # Separating data into validation and training sets
    validation_percentage = 0.20  # we reserve 20% of our data for validation
    validation_size = int(np.size(data_load, 0) * validation_percentage)
    data_train = data_load[:-validation_size]
    data_validation = data_load[-validation_size:]

    input_size = int(np.sqrt(len(data_load[0][0])))
    data_train_custom = CustomDataSetLoaderCNN(data_train, input_size)
    data_validation_custom = CustomDataSetLoaderCNN(data_validation, input_size)
    data_accuracy_custom = CustomDataSetLoaderCNN(data_validation, input_size)

    # Custom Dataloader objects based from the PyTorch library
    train_loader = DataLoader(dataset=data_train_custom, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=data_validation_custom, batch_size=batch_size, shuffle=True)
    accuracy_loader = DataLoader(dataset=data_accuracy_custom, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, accuracy_loader


def gen_noise(signal, nb_photons, wavelength, dim):
    """
    Utily function that generate shotnoise and thermal noise on the 2d images for given physical parameters
    """
    eta = 0.9
    c = 3e8
    h = 6.626e-34
    q = 1.602*10**(-19)
    kb = 1.380649*10**(-23)
    temperature = 273  # in Kelvin
    resistance = 1e6  # in Ohms

    nu = c/(wavelength * 1e-9)
    delta_f = c/(20 * 1e-9)

    source_power = (nb_photons / 5 * 1e-9) * (h * nu)
    sigma_shot = np.sqrt(((eta*q**2)/(h*nu))*source_power*delta_f)
    sigma_temp = np.sqrt(((4*kb*temperature*delta_f)/resistance))

    shot_noise_poisson = np.random.poisson(sigma_shot, dim)
    temp_noise = np.random.normal(0, sigma_temp, dim)

    signal_shot_poisson = signal + shot_noise_poisson
    signal_temp = signal + temp_noise + shot_noise_poisson

    return signal_shot_poisson, signal_temp

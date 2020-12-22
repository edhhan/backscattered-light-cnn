import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random



def clean_data(nb_photon):
    path = "C:/Users/Edward/Desktop" + "\\" + nb_photon
    files = os.listdir(path)

    for file in files:

        if nb_photon == "1e3":
            if 9000 <= os.stat(path + "\\" + file).st_size <= 12000:
                pass
            else:
                os.remove(path + "\\" + file)

        else:
            if 20000 <= os.stat(path + "\\" + file).st_size <= 30000:
                pass
            else:
                os.remove(path + "\\" + file)


def to_json(nb_photon):
    path = "C:/Users/Edward/Desktop" + "\\" + nb_photon  # TODO : change for a different user
    files = os.listdir(path)

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
    data_path = "C:/Users/Edward/Desktop" + "\\" + nb_photon
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

            # Old version with 5 labels
            # label = example_dict.get("labels")
            # label = np.array(label)

            # Add noise
            signal_shot_poisson, signal_temp = gen_noise(input_flatten, float(nb_photon), 1100, 1225)

            data.append([signal_temp, label])

        except Exception as e:
            print(file, str(e))

    random.shuffle(data)
    np.save("data.npy", data)


def reformat(data_intensity):

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

                # Normal input
                #input_matrix = reformat(iterable_data[0]*1000)
                #input_matrix_tensor = torch.from_numpy(input_matrix).view(-1, 35, 35)
                #self.X_data.append(input_matrix_tensor)

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



def preprocess(model_name, nb_photon, batch_size,
               loading=True):  # flag to load data if necessary, SHOULD BE FALSE if data is already loaded

    # Load data from JSON file if necessary (LOAD_DATA = True)
    if loading:
        #to_json(nb_photon)
        clean_data(nb_photon)
        load_data(nb_photon)

    # Import pre-loaded data from .npy file
    data_load = np.load("data.npy", allow_pickle=True)

    # Separating data into validation and training sets
    validation_percentage = 0.20  # we reserve 20% of our data for validation
    validation_size = int(np.size(data_load, 0) * validation_percentage)
    data_train = data_load[:-validation_size]
    data_validation = data_load[-validation_size:]

    input_size = int(np.sqrt(len(data_load[0][0])))
    data_train_custom = CustomDataSetLoaderCNN(data_train, input_size)
    data_validation_custom = CustomDataSetLoaderCNN(data_validation, input_size)
    data_accuracy_custom = CustomDataSetLoaderCNN(data_validation, input_size)

    train_loader = DataLoader(dataset=data_train_custom, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=data_validation_custom, batch_size=batch_size, shuffle=True)
    accuracy_loader = DataLoader(dataset=data_accuracy_custom, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, accuracy_loader


def gen_noise(signal, nb_photons, longueur_onde, dim):
    eta = 0.9
    c = 3e8
    h = 6.626e-34
    q = 1.602*10**(-19)
    kb = 1.380649*10**(-23)
    temperature = 273  # in Kelvin
    resistance = 1e6  # in Ohms

    nu = c/(longueur_onde * 1e-9)
    delta_f = c/(20 * 1e-9)

    source_power = (nb_photons / 5 * 1e-9) * (h * nu)
    sigma_shot = np.sqrt(((eta*q**2)/(h*nu))*source_power*delta_f)
    sigma_temp = np.sqrt(((4*kb*temperature*delta_f)/resistance))

    shot_noise_poisson = np.random.poisson(sigma_shot, dim)
    temp_noise = np.random.normal(0, sigma_temp, dim)

    signal_shot_poisson = signal + shot_noise_poisson
    signal_temp = signal + temp_noise + shot_noise_poisson

    return signal_shot_poisson, signal_temp

import pandas as pd
import numpy as np
import librosa
import pickle
import os

from sklearn.model_selection import train_test_split


def extract_features(audio, k = 128):
    try:

        señal, frecuencia_muestreo = librosa.load(audio)
        mfcc = librosa.feature.mfcc(y=señal, sr=frecuencia_muestreo, n_mfcc=k)
        mfcc_promedio = np.mean(mfcc, axis=1)
        return mfcc_promedio

        # audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # # Set the hop length
        # hop_length = int(len(audio)/128)
        # mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, hop_length=hop_length)
        # mfccsscaled = np.mean(mfccs,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", audio)
        return None



# Path: sound_dataset/extraxt_vectores_caracteristicos.py
def extract_features_from_dataset(dataset):
    features = []
    for index, row in dataset.iterrows():
        class_label = row["class"]
        data = extract_features(row["file_name"])
        features.append([data, class_label])
    return features

# Path: sound_dataset/extraxt_vectores_caracteristicos.py
def save_features_to_csv(features, file_name):
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
    featuresdf.to_csv(file_name, index=False)
    print("Features saved to: ", file_name)

# Path: sound_dataset/extraxt_vectores_caracteristicos.py
def load_features_from_csv(file_name):
    data = pd.read_csv(file_name)
    return data


# read the files from a folder path with has  2 folders and each of this folder has 24 folders and then we can get files
def read_files_from_folder(path):
    files = []
    for r, d, f in os.walk(path):
        for folder in d:
            for r1, d1, f1 in os.walk(path+folder):
                for folder2 in d1:
                    path_folder2 = path+folder+"/"+folder2
                    for file in os.listdir(path_folder2):
                        # print(file)
                        files.append([path_folder2+"/"+file, folder2, folder])
    return files


if __name__ == '__main__':
    path = "/Users/noemirc/Documents/ia/sound_dataset/"
    # read all files from a folder path
    files = read_files_from_folder(path)
    # create a dataframe with the files
    df = pd.DataFrame(files, columns=['file_name', 'class', 'class2'])
    # extract features from the dataframe
    features = extract_features_from_dataset(df)

    ## Dataframe completo
    # save the features to a pickle file
    pickle.dump(features, open("g1_proyecto_ia_features.pkl", "wb"))
    # save the features to a csv file
    save_features_to_csv(features, "g1_proyecto_ia_features.csv")


    ## Dataframe de testing y training (70% y 30%)
    # split the dataframe into training and testing dataset randomly
    training, testing = train_test_split(df, test_size=0.3, random_state=42)
    # extract features from the training dataset
    training_features = extract_features_from_dataset(training)
    # extract features from the testing dataset
    testing_features = extract_features_from_dataset(testing)

    # save the features to a pickle file
    pickle.dump(training_features, open("g1_proyecto_ia_training_features.pkl", "wb"))
    pickle.dump(testing_features, open("g1_proyecto_ia_testing_features.pkl", "wb"))

    # save the features to a csv file
    save_features_to_csv(training_features, "g1_proyecto_ia_training_features.csv")
    save_features_to_csv(testing_features, "g1_proyecto_ia_testing_features.csv")



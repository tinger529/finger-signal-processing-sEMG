import pandas as pd
import numpy as np
import os

'''
Create multi-label classification dataset
'''

def name_to_labels(filename):
    """
    Convert filename to labels.

    :param filename: Name of file to convert.
    :return: List of labels.
    """
    labels = [0,0,0,0,0]
    if '1' in filename:
        labels[0] = 1
    if '2' in filename:
        labels[1] = 1
    if '3' in filename:
        labels[2] = 1
    if '4' in filename:
        labels[3] = 1
    if '5' in filename:
        labels[4] = 1
    return labels


def read_training_data(data_path, files, shuffle=False, sub_split=False, n_classes=5, n_channels=3, n_rounds=3):
    """
    Read data in from CSV files and format properly for neural networks.

    data_path: Absolute file path to data.
    files: filenames of training data 
    shuffle: Whether data should be kept in sequential order or shuffled.
    sub_split: Should the data be split in half and returned as a two-tuple.
    n_classes: number of classes
    n_channels: number of channels
    n_rounds: number of rounds (same label files)
    """

    # get minimum number of rows
    min_n_rows = None
    for file in files:
        for round_num in range(1, n_rounds+1):

            full_round_path = os.path.join(data_path, 'm%d' % round_num)
            full_file_path = os.path.join(full_round_path, file)
            
            if not os.path.isfile(full_file_path):  # check if file exists
                continue
            df = pd.read_csv(full_file_path, header=None)
            if min_n_rows == None or len(df[0]) < min_n_rows:
                min_n_rows = len(df[0])

    # Fixed params
    n_steps = 150
    start_step = 1000
    end_step = min_n_rows // n_steps * n_steps
    total_steps = end_step - start_step

    # create labels
    labels = np.concatenate(
        (
            [[name_to_labels(filename) for _ in range(total_steps // n_steps)] for filename in files]
        )
    )

    # output channels (initially empty)
    channels = {i : [] for i in range(n_channels)}

    first_rest = True
    
    for file in files:
        for round_num in range(1, n_rounds+1):

            full_round_path = os.path.join(data_path, 'm%d' % round_num)
            full_file_path = os.path.join(full_round_path, file)
            
            if not os.path.isfile(full_file_path):  # check if file exists
                continue

            # read csv
            df = pd.read_csv(full_file_path, header=None)
            channels_of_df = np.array(
                [np.array(df[i][start_step:end_step]) for i in range(n_channels)]
            )
            
            # split channels and store into output channels
            for num_channel in range(n_channels):
                step = 0
                while (step + n_steps) <= total_steps:
                    split_channel = np.array(
                        channels_of_df[num_channel][step:step+n_steps]
                    )
                    channels[num_channel].append(split_channel)
                    step += n_steps
        
        # do normalization
        if file == 'rest.csv':
            restchannels = channels

        for num_channel in range(n_channels):
            channels[num_channel] = (channels[num_channel] - np.mean(restchannels[num_channel]))
            channels[num_channel] = channels[num_channel].tolist()
                
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))

    for num_channel in range(n_channels):
        X[:, :, num_channel] = np.array(channels[num_channel])
        list_of_channels.append(num_channel)
   

    if shuffle:
        print(labels)
        shuff_labels = np.zeros((len(labels), n_classes, n_channels))
        shuff_labels[:, :, 0] = labels
        shuff_labels[:, :, 1] = labels
        print(shuff_labels.shape)
        print(X.shape)

        new_data = np.concatenate([shuff_labels, X], axis=1)

        np.reshape(new_data, (n_steps + n_classes, len(labels), n_channels))
        np.random.shuffle(new_data)
        np.reshape(new_data, (len(labels), n_steps + n_classes, n_channels))

        final_data = new_data[:, 5:, :]
        final_labels = np.array(new_data[:, :n_classes, 0]).astype(int)
        print(final_labels)
        print(final_data)

        # Return (train, test)
        if sub_split:
            # train:test = 2:1
            return (
                final_data[int(len(final_labels) / 3):, :, :],
                final_labels[int(len(final_labels) / 3):],
                list_of_channels,
                final_data[:int(len(final_labels) / 3), :, :],
                final_labels[:int(len(final_labels) / 3)],
                list_of_channels
            )
        else:
            return final_data, final_labels, list_of_channels

    else:
        return X, labels, list_of_channels


def standardize(train, test):
    """
    Standardize data.

    :param train: Train data split.
    :param test: Test data split.
    :return: Normalized data set.
    """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
    X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]

    return X_train, X_test


def one_hot(labels, n_classes=6):
    """
    One-hot encoding.

    :param labels: Labels to encode.
    :param n_classes: Number of classes.
    :return: One-hot encoded labels.
    """
    expansion = np.eye(n_classes)
    y = expansion[:, labels].T 

    assert y.shape[1] == n_classes, "Wrong number of labels!"

    return y


def get_batches(X, y, batch_size=100):
    """
    Return a generator for batches.

    :param X: Data set.
    :param y: Labels.
    :param batch_size: Batch size.
    :return: Portion of data in batch-size increments.
    """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]


# not finished: revisit after finishing model
def read_testing_data(data_path, files, n_classes, n_channels=3, n_steps=150):

    start_step = 1000
    end_step = 10000
    total_steps = end_step - start_step

    channels = {i : [] for i in range(n_channels)}
    restchannels = None

    labels = np.concatenate(
        (
            [[name_to_labels(filename) for _ in range(total_steps // n_steps)] for filename in files]
        )
    )
    
    for file in files:
        full_file_path = os.path.join(data_path, file)
        df = pd.read_csv(full_file_path, header=None)
        
        channels_of_df = np.array(
            [np.array(df[i][start_step:end_step]) for i in range(n_channels)]
        )
        
        for num_channel in range(n_channels):
            step = 0
            while (step + n_steps) <= total_steps:
                split_channel = np.array(
                    channels_of_df[num_channel][step:step+n_steps]
                )
                channels[num_channel].append(split_channel)
                step += n_steps
        if file == 'rest.csv':
            restchannels = channels
        for num_channel in range(n_channels):
            channels[num_channel] = (channels[num_channel] - np.mean(restchannels[num_channel]))
            channels[num_channel] = channels[num_channel].tolist()

    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))

    for num_channel in range(n_channels):
        X[:, :, num_channel] = np.array(channels[num_channel])
        list_of_channels.append(num_channel)

    return X, labels, list_of_channels
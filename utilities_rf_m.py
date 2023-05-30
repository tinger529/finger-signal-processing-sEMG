import pandas as pd
import numpy as np
import os

'''
Create dataset from CSV files
'''


def read_training_data(data_path, files, shuffle=False, sub_split=False):
    """
    Read data in from CSV files and format properly for neural networks.

    :param data_path: Absolute file path to data.
    :param files: filenames of training data 
    :param shuffle: Whether data should be kept in sequential order or shuffled.
    :param sub_split: Should the data be split in half and returned as a two-tuple.
    :return: Data formatted for neural network training.
    """
    n_classes = 24
    n_channels = 3
    n_rounds = 3

    # get minimum number of rows
    min_n_rows = None
    for file in files:
        for round_num in range(1, n_rounds+1):
            full_round_path = os.path.join(data_path, 'm%d' % round_num)
            full_file_path = os.path.join(full_round_path, file)
            print(full_file_path)
            # check if file exists
            if not os.path.isfile(full_file_path):
                continue
            df = pd.read_csv(full_file_path, header=None)
            if min_n_rows == None or len(df[0]) < min_n_rows:
                min_n_rows = len(df[0])

    # Fixed params
    n_steps = 1000 # TODO 
    start_step = 1000 # drop rows before <start_step> of csv
    end_step = min_n_rows // n_steps * n_steps
    total_steps = end_step - start_step
    
    
    # Assign numeric label to categories:
    #
    # rest = 0
    # first = 1
    # second = 2
    # third = 3
    # fourth = 4
    # fifth = 5
    #

    labels = np.concatenate(
        (
            [[class_id for _ in range(total_steps // n_steps)] for class_id in range(n_classes)]
        )
    )

    print(len(labels))

    channels = {i : [] for i in range(n_channels)}
    
    for file in files:
        for round_num in range(1, n_rounds+1):
            full_round_path = os.path.join(data_path, 'm%d' % round_num)
            full_file_path = os.path.join(full_round_path, file)
            # check if file exists
            if not os.path.isfile(full_file_path):
                continue
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
            
    print(len(channels[0]))
                
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))

    for num_channel in range(n_channels):
        X[:, :, num_channel] = np.array(channels[num_channel])
        list_of_channels.append(num_channel)
   

    if shuffle:
        shuff_labels = np.zeros((len(labels), 1, n_channels))
        shuff_labels[:, 0, 0] = labels
        shuff_labels[:, 0, 1] = labels

        new_data = np.concatenate([shuff_labels, X], axis=1)

        np.reshape(new_data, (n_steps + 1, len(labels), n_channels))
        np.random.shuffle(new_data)
        np.reshape(new_data, (len(labels), n_steps + 1, n_channels))

        final_data = new_data[:, 1:, :]
        final_labels = np.array(new_data[:, 0, 0]).astype(int)

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



def read_testing_data(data_path, files, n_classes):

    n_channels = 3
    n_steps = 1000
    start_step = 1000
    end_step = 10000
    total_steps = end_step - start_step

    channels = {i : [] for i in range(n_channels)}
    restchannels = None

    print(n_classes)

    labels = np.concatenate(
        (
            [[class_id for _ in range(total_steps // n_steps)] for class_id in [0, 11]]
        )
    )

    channels = {i : [] for i in range(n_channels)}
    restchannels = None
    
    
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
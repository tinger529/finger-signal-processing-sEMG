import pandas as pd
import numpy as np
import os

'''
Create dataset from CSV files
'''

# set random seed
np.random.seed(0)


def read_data(data_path, split_type="train", shuffle=False, sub_split=False):
    """
    Read data in from CSV files and format properly for neural networks.

    :param data_path: Absolute file path to data.
    :param split_type: If splitting the same dataset, which split to designate this one as.
    :param shuffle: Whether data should be kept in sequential order or shuffled.
    :param sub_split: Should the data be split in half and returned as a two-tuple.
    :return: Data formatted for neural network training.
    """
    # Fixed params
    n_class = 6
    n_channels = 3
    n_steps = 1000 # TODO 
    start_step = 1000 # drop rows before <start_step> of csv
    end_step = 30000
    total_steps = end_step - start_step
    
    # dataset2/test1 or dataset2/test2
    test_round = [1, 2]    
    
    # Assign numeric label to categories:
    #
    # rest = 0
    # first = 1
    # second = 2
    # third = 3
    # fourth = 4
    # fifth = 5
    #
    # print(total_steps // n_steps * len(test_round))

    labels = np.concatenate(
        (
            [[class_id for _ in range(total_steps // n_steps * len(test_round))] for class_id in range(n_class)]
        )
    )

    # print(labels.shape)

    files = [
        'rest.csv',
        '1st.csv',
        '2nd.csv',
        '3rd.csv',
        '4th.csv',
        '5th.csv'
    ]

    channels = {i : [] for i in range(n_channels)}
    restchannels = None
    
    for file in files:
        for round_num in test_round:
            full_round_path = os.path.join(data_path, 's%d' % round_num)
            full_file_path = os.path.join(full_round_path, file)

            df = pd.read_csv(full_file_path, header=None)
            # print("df shape")
            # print(df.shape)
            channels_of_df = np.array(
                [np.array(df[i][start_step:end_step]) for i in range(n_channels)]
            )
            # print("shape")
            # print(channels_of_df.shape)
            
            for num_channel in range(n_channels):
                step = 0
                while (step + n_steps) <= total_steps:
                    split_channel = np.array(
                        channels_of_df[num_channel][step:step+n_steps]
                    )
                    channels[num_channel].append(split_channel)
                    step += n_steps
        if file == 'rest.csv':
            # print("rest")
            # print(len(channels[0]))
            restchannels = channels
        elif restchannels != None:
            for num_channel in range(n_channels):
                channels[num_channel] = (channels[num_channel] - np.mean(restchannels[num_channel]))
                channels[num_channel] = channels[num_channel].tolist()
                
                
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
            # change to 1:2
            print("label")
            print(len(final_labels))
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
        if sub_split:
            # change to 1:2
            return (
                X[int(len(labels) / 3):, :, :],
                labels[int(len(labels) / 3):],
                list_of_channels,
                X[:int(len(labels) / 3), :, :],
                labels[:int(len(labels) / 3)],
                list_of_channels
            )
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


def one_hot(labels, n_class=6):
    """
    One-hot encoding.

    :param labels: Labels to encode.
    :param n_class: Number of classes.
    :return: One-hot encoded labels.
    """
    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T

    assert y.shape[1] == n_class, "Wrong number of labels!"

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

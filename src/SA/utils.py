import re
import numpy as np

def decontract_phrase(phrase):
    phrase = phrase.lower()

    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase

def get_target_list_for_polarity(polarity):
    if polarity == 'positive':
        return [0, 0, 1]
    if polarity == 'negative':
        return [1, 0, 0]
    return [0, 1, 0]

def get_polarity_from_target_list(target_list):
    if target_list == [0, 0, 1]:
        return 'positive'
    if target_list == [1, 0, 0]:
        return 'negative'
    return 'neutral'

def print_split_by_labels(train_dataset, test_dataset):
    print("Train Dataset")
    targets_values = np.argmax(list(train_dataset.targets), axis=1)

    print(f'\tPositive: {len(targets_values[targets_values == 2])}')
    print(f'\tNeutral: {len(targets_values[targets_values == 1])}')
    print(f'\tNegative: {len(targets_values[targets_values == 0])}')

    print("Test Dataset")
    targets_values = np.argmax(list(test_dataset.targets), axis=1)

    print(f'\tPositive: {len(targets_values[targets_values == 2])}')
    print(f'\tNeutral: {len(targets_values[targets_values == 1])}')
    print(f'\tNegative: {len(targets_values[targets_values == 0])}')
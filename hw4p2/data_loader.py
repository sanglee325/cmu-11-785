from libri_dataset import LibriSamples, LibriSamplesTest

from config import *

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    n = len(letter_list)
    letter2index = {letter_list[i]:i for i in range(n)}
    index2letter = {i:letter_list[i] for i in range(n)}

    return letter2index, index2letter
    

def transform_index_to_letter(batch_indices):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    for r in batch_indices:
        curr = ""
        for i in r:
            curr += index2letter[i]
        transcripts.append(curr)

    return transcripts

def load_dataset(args, batch_size):
    train_data = LibriSamples(args.data_path, 'train')
    val_data = LibriSamples(args.data_path, 'dev')
    test_data = LibriSamplesTest(args.data_path, 'test_order.csv')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, collate_fn=LibriSamples.collate_fn) 
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, collate_fn=LibriSamples.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, collate_fn=LibriSamplesTest.collate_fn)

    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    return train_loader, val_loader, test_loader

letter2index, index2letter = create_dictionaries(LETTER_LIST)
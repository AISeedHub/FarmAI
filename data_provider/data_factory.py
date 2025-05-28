from data_provider.data_loader import (Dataset_Custom,
                                       Dataset_TextPrompt)
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    "farm": Dataset_TextPrompt
}


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(DataLoader):
    """A custom DataLoader that ensures infinite iteration for multiple epochs.
    
    This class wraps PyTorch's DataLoader to provide infinite iteration capability by 
    reusing the batch sampler across epochs. It maintains internal state to track 
    initialization and iteration.

    Args:
        *args: Variable length argument list to be passed to DataLoader.
        **kwargs: Arbitrary keyword arguments to be passed to DataLoader.
        
    Example:
        >>> dataset = Dataset_Custom()
        >>> loader = MultiEpochsDataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        >>>     # Process batch
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the number of batches in a single epoch.
        
        Returns:
            int: Length of the batch sampler for one epoch.
        """
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates an iterator that yields batches of data.
        
        Yields:
            Any: Next batch of data from the dataset.
        """
        for i in range(len(self)):
            yield next(self.iterator)


def data_provider(args, flag):
    Dataset = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Dataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))

    data_loader = MultiEpochsDataLoader(data_set,
                                        batch_size=batch_size,
                                        shuffle=shuffle_flag,
                                        drop_last=drop_last)

    return data_set, data_loader

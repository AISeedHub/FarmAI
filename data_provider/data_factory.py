from data_provider.data_loader import (Dataset_Custom, Dataset_Pred,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def data_provider(args, flag):
    Dataset = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Dataset = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'GY':
        data_set = Dataset(args.root_path,
                        flag=flag,
                        inp_seq_len=96,
                        label_seq_en=48,
                        pred_seq_len=24)
    else:
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

    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=16,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     multiprocessing_context='spawn',
    #     drop_last=drop_last)
    return data_set, data_loader

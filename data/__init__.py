from typing import Union
from torch.utils.data import Dataset
import os

class Corpus(Dataset):
    def __init__(self, contain_only: Union[None, str] = None, multiplex: bool = True, cutoff: int | None = None, validation_set: bool = False,
                 bidirectional: bool = False):
        self.multiplex = multiplex
        self.dataset_path = os.path.dirname(os.path.realpath(__file__))
        self.dataset_path = os.path.join(self.dataset_path, "iwslt_en_vi")
        if contain_only:
            self.load_vi = contain_only[:2] == "vi"
            self.load_en = contain_only[:2] == "en"
        else:
            self.load_vi = True
            self.load_en = True

        self.en = []
        if self.load_en:
            ds_files = ['train.en'] if not validation_set else ['tst2012.en', 'tst2013.en']
            for english in ds_files:
                with open(os.path.join(self.dataset_path, english)) as corpus:
                    self.en += corpus.readlines()

        self.vi = []
        if self.load_vi:
            ds_files = ['train.vi'] if not validation_set else ['tst2012.vi', 'tst2013.vi']
            for vietnamese in ds_files:
                with open(os.path.join(self.dataset_path, vietnamese)) as corpus:
                    self.vi += corpus.readlines()

        # Cleaning
        self.__len = len(self.en) if self.load_en else len(self.vi)
        if self.load_vi and self.load_en:
            _corpus = [(en.strip(), vi.strip()) for en, vi in zip(self.en, self.vi) if en.strip() != '' and vi.strip() != '']
            self.en, self.vi = zip(*_corpus)
            assert len(self.en) == len(self.vi)
            self.__len = len(self.en)
            if not multiplex:
                self.__len *= 2
        if cutoff is not None:
            self.__len = min(cutoff, self.__len)
        if bidirectional:
            assert contain_only is None
            assert multiplex
            self.__len = self.__len * 2
            _en = self.en + self.vi
            _vi = self.vi + self.en
            self.en = _en
            self.vi = _vi



    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        if self.load_en and self.load_vi:
            if self.multiplex:
                return self.en[index], self.vi[index]
            else:
                return (self.en if index % 2 == 0 else self.vi)[index // 2]
        else:
            return (self.en if self.load_en else self.vi)[index]


if __name__ == "__main__":
    corpus = Corpus()
    for i in corpus:
        print(*i, sep=f"\n{'-'*16}SEP{'-'*16}\n", end='\n'*4)
        if (i[0].strip() == '' or i[1].strip() == ''):
            break

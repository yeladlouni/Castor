import numpy as np
import json
import torch
from torchtext.data import Dataset, Example
from torchtext.data.field import Field, RawField
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors
from torchtext.data.pipeline import Pipeline

from datasets.castor_dataset import CastorPairDataset
from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


def get_class_probs(sim, *args):
    """
    Convert a single label into class probabilities.
    """
    class_probs = np.zeros(Semeval.NUM_CLASSES)
    class_probs[int(sim)] = 1
    return class_probs


class Semeval(Dataset):
    NAME = 'Semeval'
    NUM_CLASSES = 2
    QID_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True)
    QAID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, postprocessing=Pipeline(get_class_probs))
    RAW_TEXT_FIELD = RawField()

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path, **kwargs):
        """
        Create a Semeval dataset instance
        """

        fields = [
            ('qid', self.QID_FIELD),
            ('qaid', self.QID_FIELD),
            ('label', self.LABEL_FIELD),
            ('sentence_1', self.TEXT_FIELD),
            ('sentence_2', self.TEXT_FIELD),
            ('sentence_1_raw', self.RAW_TEXT_FIELD),
            ('sentence_2_raw', self.RAW_TEXT_FIELD),
            ('ext_feats', self.EXT_FEATS_FIELD)
        ]

        examples = []

        with open(path) as infile:
            for line in infile:
                content = json.loads(line)

                sent_list_1 = content['question']
                sent_list_2 = content['qaquestion']

                word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
                overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)
                overlap_feats = []
                values = [
                    content['qid'],
                    content['qaid'],
                    content['qarel'],
                    content['question'],
                    content['qaquestion'],
                    ' '.join(content['question']),
                    ' '.join(content['qaquestion']),
                    overlap_feats
                ]

                examples.append(Example.fromlist(values, fields))

        super(Semeval, self).__init__(examples, fields, **kwargs)



    @classmethod
    def splits(cls, path, train='train.json', validation='dev.json', test='test.json', **kwargs):
        return super(Semeval, cls).splits(path, train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_dir: directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param pt_file: load cached embedding file from disk if it is true
        :param unk_init: function used to generate vector for OOV words
        :return:
        """

        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, validation, test = cls.splits(path)

        cls.LABEL_FIELD.build_vocab(train, validation, test)
        cls.TEXT_FIELD.build_vocab(train, validation, test, vectors=vectors)
        return BucketIterator.splits((train, validation, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)
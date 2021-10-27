# coding: utf-8
"""
Data module
"""
import itertools
import os
import sys
import random

import tensorflow_datasets as tfds
import tensorflow as tf
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

import torch
import torchtext
from torchtext import data
# from torchtext.data import Dataset,Iterator
import socket

from torchtext.data import RawField, Field, BucketIterator, Dataset, Iterator
from torchtext.data.iterator import batch,pool
from slt.signjoey.dataset import SignTranslationDataset
from slt.signjoey.vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)


def load_data(data_cfg: dict): # -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    If you set ``random_dev_subset``, a random selection of this size is used
    from the dev development instead of the full development set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """

    if data_cfg["version"] == 'autsl':
        config = SignDatasetConfig(name="include-videos", version="1.0.0", include_video=True, fps=30)
        autsl = tfds.load(name='autsl', builder_kwargs=dict(config=config))

        # for datum in itertools.islice(autsl["train"], 0, 20):
        #     print(datum['sample'].numpy(), datum['id'].numpy().decode('utf-8'), datum['gloss_id'].numpy())

    if data_cfg["version"] == 'phoenix_2014_trans':
        data_path = "/home/nlp/dorink/project/slt/data" #data_cfg.get("data_path", "./data")

        # Get the datasets path.(?)
        if isinstance(data_cfg["train"], list):
            train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
            dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
            test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
            pad_feature_size = sum(data_cfg["feature_size"])
        else:
            train_paths = os.path.join(data_path, data_cfg["train"])
            dev_paths = os.path.join(data_path, data_cfg["dev"])
            test_paths = os.path.join(data_path, data_cfg["test"])
            pad_feature_size = data_cfg["feature_size"]

    level = data_cfg["level"]
    if data_cfg["version"] == 'phoenix_2014_trans':
        txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    if data_cfg["version"] == 'phoenix_2014_trans':
        def tokenize_text(text):
            if level == "char":
                return list(text)
            else:
                return text.split()

    def tokenize_features(features):
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    # NOTE (Cihan): The something was necessary to match the function signature.
    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    sequence_field = RawField()
    signer_field = RawField()

    if data_cfg["version"] == 'phoenix_2014_trans':
        sgn_field = Field(
            use_vocab=False,
            init_token=None,
            dtype=torch.float32,
            preprocessing=tokenize_features,
            tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
            batch_first=True,
            include_lengths=True,
            postprocessing=stack_features,
            pad_token=torch.zeros((pad_feature_size,)),
        )
    else:
        sgn_field = Field(
            use_vocab=False,
            init_token=None,
            dtype=torch.float32,
            preprocessing=tokenize_features,
            tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
            batch_first=True,
            include_lengths=True,
            postprocessing=stack_features,
            # pad_token=torch.zeros((pad_feature_size,)),
        )

    if data_cfg["version"] == 'phoenix_2014_trans':
        gls_field = Field(
                pad_token=PAD_TOKEN,
                tokenize=tokenize_text,
                batch_first=True,
                lower=False,
                include_lengths=True,
            )
    else:
        gls_field = Field(
            sequential=False,#?
            use_vocab=False,
            pad_token=PAD_TOKEN,
            # tokenize=tokenize_text, #??
            batch_first=True,
            lower=False,
            include_lengths=True,
        )

    if data_cfg["version"] == 'phoenix_2014_trans':
        txt_field = Field(
            init_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
            tokenize=tokenize_text,
            unk_token=UNK_TOKEN,
            batch_first=True,
            lower=txt_lowercase,
            include_lengths=True,
        )

    # Get the preprocessed training set.
    if data_cfg["version"] == 'phoenix_2014_trans':
        train_data = SignTranslationDataset(
            dataset_type=data_cfg["version"],
            path= train_paths,#rwth_phoenix2014_t["train"],
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
            filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
            and len(vars(x)["txt"]) <= max_sent_length,
        )
    else:
        train_data = SignTranslationDataset(
            dataset_type=data_cfg["version"],
            path=autsl["train"],  # rwth_phoenix2014_t["train"],
            fields=(sequence_field, signer_field, sgn_field, gls_field),
            filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
                                  and len(vars(x)["txt"]) <= max_sent_length,
        )

    # Set the maximal vocab size and minimum items frequency of the gloss and text vocabs.
    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    # Get the vocabs if already exists, otherwise set them to None.
    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)

    # Build the gloss and text vocabs based on the training set.
    gls_vocab = build_vocab(
        version=data_cfg["version"],
        field="gls",
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        dataset=train_data,
        vocab_file=gls_vocab_file,
    )
    txt_vocab = build_vocab(
        version=data_cfg["version"],
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data,
        vocab_file=txt_vocab_file,
    )
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        train_data = keep

    # Get the preprocessed validation set.
    dev_data = SignTranslationDataset(
        dataset_type=data_cfg["version"],
        path= dev_paths,#rwth_phoenix2014_t["validation"],
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )
    # dev_data =rwth_phoenix2014_t["validation"]

    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        # select this many development examples randomly and discard the rest
        keep_ratio = random_dev_subset / len(dev_data)
        keep, _ = dev_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        dev_data = keep

    # Get the preprocessed test set.
    # check if target exists
    test_data = SignTranslationDataset(
        dataset_type=data_cfg["version"],
        path= test_paths,#rwth_phoenix2014_t["test"],
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )
    # test_data =rwth_phoenix2014_t["test"]

    gls_field.vocab = gls_vocab
    txt_field.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab


# TODO (Cihan): I don't like this use of globals.
#  Need to find a more elegant solution for this it at some point.
# pylint: disable=global-at-module-level
global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


# class SignTranslationIterator(BucketIterator):
#
#     def get_element(self,num):
#         elem =[]
#         # for i,s in enumerate(itertools.islice(self.dataset, num, num+1)):
#         for s in self.dataset[num,num+1]:
#             elem.append({
#                     "id": s["id"].numpy().decode('utf-8').rstrip('\n'),
#                     "signer": s["signer"].numpy().decode('utf-8').rstrip('\n'),
#                     "gloss": s["gloss"].numpy().decode('utf-8').strip().rstrip('\n'),
#                     "text": s["text"].numpy().decode('utf-8').strip().rstrip('\n'),
#                     "video": s["video"],
#                 })
#         return elem[0]
#
#     def data(self):
#         """Return the examples in the dataset in order, sorted, or shuffled."""
#         if self.sort:
#             xs = sorted(self.dataset, key=self.sort_key)
#         elif self.shuffle:
#             xs = [self.get_element(i) for i in self.random_shuffler(range(len(self.dataset)))]
#         else:
#             xs = self.dataset
#         return xs
#
#     def create_batches(self):
#         if self.sort:
#             self.batches = batch(self.data(), self.batch_size,
#                                  self.batch_size_fn)
#         else:
#             self.examples = [self.get_element(i) for i in range(len(self.dataset))]
#             self.batches = pool(self.data(), self.batch_size,
#                                 self.sort_key, self.batch_size_fn,
#                                 random_shuffler=self.random_shuffler,
#                                 shuffle=self.shuffle,
#                                 sort_within_batch=self.sort_within_batch)

def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.sgn),
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter

# # coding: utf-8
# """
# Data module
# """
# import os
# import sys
# import random
#
# import torch
# from torchtext import data
# from torchtext.data import Dataset, Iterator
# import socket
# from signjoey.dataset import SignTranslationDataset
# from signjoey.vocabulary import (
#     build_vocab,
#     Vocabulary,
#     UNK_TOKEN,
#     EOS_TOKEN,
#     BOS_TOKEN,
#     PAD_TOKEN,
# )
#
#
# def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
#     """
#     Load train, dev and optionally test data as specified in configuration.
#     Vocabularies are created from the training set with a limit of `voc_limit`
#     tokens and a minimum token frequency of `voc_min_freq`
#     (specified in the configuration dictionary).
#
#     The training data is filtered to include sentences up to `max_sent_length`
#     on source and target side.
#
#     If you set ``random_train_subset``, a random selection of this size is used
#     from the training set instead of the full training set.
#
#     If you set ``random_dev_subset``, a random selection of this size is used
#     from the dev development instead of the full development set.
#
#     :param data_cfg: configuration dictionary for data
#         ("data" part of configuration file)
#     :return:
#         - train_data: training dataset
#         - dev_data: development dataset
#         - test_data: test dataset if given, otherwise None
#         - gls_vocab: gloss vocabulary extracted from training data
#         - txt_vocab: spoken text vocabulary extracted from training data
#     """
#
#     config = SignDatasetConfig(name="only-annotations", version="3.0.0", include_video=True)
#     rwth_phoenix2014_t = tfds.load(name='rwth_phoenix2014_t', builder_kwargs=dict(config=config))
#
#     # data_path = data_cfg.get("data_path", "./data")
#     #
#     # if isinstance(data_cfg["train"], list):
#     #     train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
#     #     dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
#     #     test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
#     #     pad_feature_size = sum(data_cfg["feature_size"])
#     #
#     # else:
#     #     train_paths = os.path.join(data_path, data_cfg["train"])
#     #     dev_paths = os.path.join(data_path, data_cfg["dev"])
#     #     test_paths = os.path.join(data_path, data_cfg["test"])
#     #     pad_feature_size = data_cfg["feature_size"]
#
#     # pad_feature_size = sum(data_cfg["feature_size"]) # ???
#
#     level = data_cfg["level"]
#     txt_lowercase = data_cfg["txt_lowercase"]
#     max_sent_length = data_cfg["max_sent_length"]
#
#     def tokenize_text(text):
#         if level == "char":
#             return list(text)
#         else:
#             return text.split()
#
#     def tokenize_features(features):
#         ft_list = torch.split(features, 1, dim=0)
#         return [ft.squeeze() for ft in ft_list]
#
#     # NOTE (Cihan): The something was necessary to match the function signature.
#     def stack_features(features, something):
#         return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
#
#     sequence_field = data.RawField()
#     signer_field = data.RawField()
#
#     sgn_field = data.Field(
#         use_vocab=False,
#         init_token=None,
#         dtype=torch.float32,
#         preprocessing=tokenize_features,
#         tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
#         batch_first=True,
#         include_lengths=True,
#         postprocessing=stack_features,
#         # pad_token=torch.zeros((pad_feature_size,)),
#     )
#
#     gls_field = data.Field(
#         pad_token=PAD_TOKEN,
#         tokenize=tokenize_text,
#         batch_first=True,
#         lower=False,
#         include_lengths=True,
#     )
#
#     txt_field = data.Field(
#         init_token=BOS_TOKEN,
#         eos_token=EOS_TOKEN,
#         pad_token=PAD_TOKEN,
#         tokenize=tokenize_text,
#         unk_token=UNK_TOKEN,
#         batch_first=True,
#         lower=txt_lowercase,
#         include_lengths=True,
#     )
#
#     train_data = SignTranslationDataset(
#         path= rwth_phoenix2014_t["train"], #train_paths,
#         fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
#         filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
#         and len(vars(x)["txt"]) <= max_sent_length,
#     )
#
#     gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
#     gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
#     txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
#     txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)
#
#     gls_vocab_file = data_cfg.get("gls_vocab", None)
#     txt_vocab_file = data_cfg.get("txt_vocab", None)
#
#     gls_vocab = build_vocab(
#         field="gls",
#         min_freq=gls_min_freq,
#         max_size=gls_max_size,
#         dataset=train_data,
#         vocab_file=gls_vocab_file,
#     )
#     txt_vocab = build_vocab(
#         field="txt",
#         min_freq=txt_min_freq,
#         max_size=txt_max_size,
#         dataset=train_data,
#         vocab_file=txt_vocab_file,
#     )
#     random_train_subset = data_cfg.get("random_train_subset", -1)
#     if random_train_subset > -1:
#         # select this many training examples randomly and discard the rest
#         keep_ratio = random_train_subset / len(train_data)
#         keep, _ = train_data.split(
#             split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
#         )
#         train_data = keep
#
#     dev_data = SignTranslationDataset(
#         path=dev_paths,
#         fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
#     )
#     random_dev_subset = data_cfg.get("random_dev_subset", -1)
#     if random_dev_subset > -1:
#         # select this many development examples randomly and discard the rest
#         keep_ratio = random_dev_subset / len(dev_data)
#         keep, _ = dev_data.split(
#             split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
#         )
#         dev_data = keep
#
#     # check if target exists
#     test_data = SignTranslationDataset(
#         path=test_paths,
#         fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
#     )
#
#     gls_field.vocab = gls_vocab
#     txt_field.vocab = txt_vocab
#     return train_data, dev_data, test_data, gls_vocab, txt_vocab
#
#
# # TODO (Cihan): I don't like this use of globals.
# #  Need to find a more elegant solution for this it at some point.
# # pylint: disable=global-at-module-level
# global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
#
#
# # pylint: disable=unused-argument,global-variable-undefined
# def token_batch_size_fn(new, count, sofar):
#     """Compute batch size based on number of tokens (+padding)"""
#     global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
#     if count == 1:
#         max_sgn_in_batch = 0
#         max_gls_in_batch = 0
#         max_txt_in_batch = 0
#     max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
#     max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
#     max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
#     sgn_elements = count * max_sgn_in_batch
#     gls_elements = count * max_gls_in_batch
#     txt_elements = count * max_txt_in_batch
#     return max(sgn_elements, gls_elements, txt_elements)
#
#
# def make_data_iter(
#     dataset: Dataset,
#     batch_size: int,
#     batch_type: str = "sentence",
#     train: bool = False,
#     shuffle: bool = False,
# ) -> Iterator:
#     """
#     Returns a torchtext iterator for a torchtext dataset.
#
#     :param dataset: torchtext dataset containing sgn and optionally txt
#     :param batch_size: size of the batches the iterator prepares
#     :param batch_type: measure batch size by sentence count or by token count
#     :param train: whether it's training time, when turned off,
#         bucketing, sorting within batches and shuffling is disabled
#     :param shuffle: whether to shuffle the data before each epoch
#         (no effect if set to True for testing)
#     :return: torchtext iterator
#     """
#
#     batch_size_fn = token_batch_size_fn if batch_type == "token" else None
#
#     if train:
#         # optionally shuffle and sort during training
#         data_iter = data.BucketIterator(
#             repeat=False,
#             sort=False,
#             dataset=dataset,
#             batch_size=batch_size,
#             batch_size_fn=batch_size_fn,
#             train=True,
#             sort_within_batch=True,
#             sort_key=lambda x: len(x.sgn),
#             shuffle=shuffle,
#         )
#     else:
#         # don't sort/shuffle for validation/inference
#         data_iter = data.BucketIterator(
#             repeat=False,
#             dataset=dataset,
#             batch_size=batch_size,
#             batch_size_fn=batch_size_fn,
#             train=False,
#             sort=False,
#         )
#
#     return data_iter


############################ phoneix 17/5 ############################

# # coding: utf-8
# """
# Data module
# """
# import itertools
# import os
# import sys
# import random
#
# import tensorflow_datasets as tfds
# import tensorflow as tf
# import sign_language_datasets.datasets
# from sign_language_datasets.datasets.config import SignDatasetConfig
#
# import torch
# import torchtext
# from torchtext import data
# # from torchtext.data import Dataset,Iterator
# import socket
#
# from torchtext.legacy.data import RawField, Field, BucketIterator, Dataset, Iterator
# from torchtext.legacy.data.iterator import batch,pool
# from signjoey.dataset import SignTranslationDataset
# from signjoey.vocabulary import (
#     build_vocab,
#     Vocabulary,
#     UNK_TOKEN,
#     EOS_TOKEN,
#     BOS_TOKEN,
#     PAD_TOKEN,
# )
#
#
# def load_data(data_cfg: dict):# -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
#     """
#     Load train, dev and optionally test data as specified in configuration.
#     Vocabularies are created from the training set with a limit of `voc_limit`
#     tokens and a minimum token frequency of `voc_min_freq`
#     (specified in the configuration dictionary).
#
#     The training data is filtered to include sentences up to `max_sent_length`
#     on source and target side.
#
#     If you set ``random_train_subset``, a random selection of this size is used
#     from the training set instead of the full training set.
#
#     If you set ``random_dev_subset``, a random selection of this size is used
#     from the dev development instead of the full development set.
#
#     :param data_cfg: configuration dictionary for data
#         ("data" part of configuration file)
#     :return:
#         - train_data: training dataset
#         - dev_data: development dataset
#         - test_data: test dataset if given, otherwise None
#         - gls_vocab: gloss vocabulary extracted from training data
#         - txt_vocab: spoken text vocabulary extracted from training data
#     """
#
#     # config = SignDatasetConfig(name="only-annotations", version="3.0.0", include_video=True)
#     # rwth_phoenix2014_t = tfds.load(name='rwth_phoenix2014_t', builder_kwargs=dict(config=config))
#
#     data_path = "/home/nlp/dorink/project/slt/data"#data_cfg.get("data_path", "./data")
#
#     if isinstance(data_cfg["train"], list):
#         train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
#         dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
#         test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
#         pad_feature_size = sum(data_cfg["feature_size"])
#     else:
#         train_paths = os.path.join(data_path, data_cfg["train"])
#         dev_paths = os.path.join(data_path, data_cfg["dev"])
#         test_paths = os.path.join(data_path, data_cfg["test"])
#         pad_feature_size = data_cfg["feature_size"]
#
#     level = data_cfg["level"]
#     txt_lowercase = data_cfg["txt_lowercase"]
#     max_sent_length = data_cfg["max_sent_length"]
#
#     def tokenize_text(text):
#         if level == "char":
#             return list(text)
#         else:
#             return text.split()
#
#     def tokenize_features(features):
#         ft_list = torch.split(features, 1, dim=0)
#         return [ft.squeeze() for ft in ft_list]
#
#     # NOTE (Cihan): The something was necessary to match the function signature.
#     def stack_features(features, something):
#         return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
#
#     sequence_field = RawField()
#     signer_field = RawField()
#
#     sgn_field = Field(
#         use_vocab=False,
#         init_token=None,
#         dtype=torch.float32,
#         preprocessing=tokenize_features,
#         tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
#         batch_first=True,
#         include_lengths=True,
#         postprocessing=stack_features,
#         pad_token=torch.zeros((pad_feature_size,)),
#     )
#
#     gls_field = Field(
#         pad_token=PAD_TOKEN,
#         tokenize=tokenize_text,
#         batch_first=True,
#         lower=False,
#         include_lengths=True,
#     )
#
#     txt_field = Field(
#         init_token=BOS_TOKEN,
#         eos_token=EOS_TOKEN,
#         pad_token=PAD_TOKEN,
#         tokenize=tokenize_text,
#         unk_token=UNK_TOKEN,
#         batch_first=True,
#         lower=txt_lowercase,
#         include_lengths=True,
#     )
#
#     train_data = SignTranslationDataset(
#         path= train_paths,#rwth_phoenix2014_t["train"],
#         fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
#         filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
#         and len(vars(x)["txt"]) <= max_sent_length,
#     )
#
#     gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
#     gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
#     txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
#     txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)
#
#     gls_vocab_file = data_cfg.get("gls_vocab", None)
#     txt_vocab_file = data_cfg.get("txt_vocab", None)
#
#     # train_data =rwth_phoenix2014_t["train"]
#     gls_vocab = build_vocab(
#         field="gls",
#         min_freq=gls_min_freq,
#         max_size=gls_max_size,
#         dataset=train_data,
#         vocab_file=gls_vocab_file,
#     )
#     txt_vocab = build_vocab(
#         field="txt",
#         min_freq=txt_min_freq,
#         max_size=txt_max_size,
#         dataset=train_data,
#         vocab_file=txt_vocab_file,
#     )
#     random_train_subset = data_cfg.get("random_train_subset", -1)
#     if random_train_subset > -1:
#         # select this many training examples randomly and discard the rest
#         keep_ratio = random_train_subset / len(train_data)
#         keep, _ = train_data.split(
#             split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
#         )
#         train_data = keep
#
#     dev_data = SignTranslationDataset(
#         path= dev_paths,#rwth_phoenix2014_t["validation"],
#         fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
#     )
#     # dev_data =rwth_phoenix2014_t["validation"]
#
#     random_dev_subset = data_cfg.get("random_dev_subset", -1)
#     if random_dev_subset > -1:
#         # select this many development examples randomly and discard the rest
#         keep_ratio = random_dev_subset / len(dev_data)
#         keep, _ = dev_data.split(
#             split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
#         )
#         dev_data = keep
#
#     # check if target exists
#     test_data = SignTranslationDataset(
#         path= test_paths,#rwth_phoenix2014_t["test"],
#         fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
#     )
#     # test_data =rwth_phoenix2014_t["test"]
#
#     gls_field.vocab = gls_vocab
#     txt_field.vocab = txt_vocab
#     return train_data, dev_data, test_data, gls_vocab, txt_vocab
#
#
# # TODO (Cihan): I don't like this use of globals.
# #  Need to find a more elegant solution for this it at some point.
# # pylint: disable=global-at-module-level
# global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
#
#
# # pylint: disable=unused-argument,global-variable-undefined
# def token_batch_size_fn(new, count, sofar):
#     """Compute batch size based on number of tokens (+padding)"""
#     global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
#     if count == 1:
#         max_sgn_in_batch = 0
#         max_gls_in_batch = 0
#         max_txt_in_batch = 0
#     max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
#     max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
#     max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
#     sgn_elements = count * max_sgn_in_batch
#     gls_elements = count * max_gls_in_batch
#     txt_elements = count * max_txt_in_batch
#     return max(sgn_elements, gls_elements, txt_elements)
#
#
# # class SignTranslationIterator(BucketIterator):
# #
# #     def get_element(self,num):
# #         elem =[]
# #         # for i,s in enumerate(itertools.islice(self.dataset, num, num+1)):
# #         for s in self.dataset[num,num+1]:
# #             elem.append({
# #                     "id": s["id"].numpy().decode('utf-8').rstrip('\n'),
# #                     "signer": s["signer"].numpy().decode('utf-8').rstrip('\n'),
# #                     "gloss": s["gloss"].numpy().decode('utf-8').strip().rstrip('\n'),
# #                     "text": s["text"].numpy().decode('utf-8').strip().rstrip('\n'),
# #                     "video": s["video"],
# #                 })
# #         return elem[0]
# #
# #     def data(self):
# #         """Return the examples in the dataset in order, sorted, or shuffled."""
# #         if self.sort:
# #             xs = sorted(self.dataset, key=self.sort_key)
# #         elif self.shuffle:
# #             xs = [self.get_element(i) for i in self.random_shuffler(range(len(self.dataset)))]
# #         else:
# #             xs = self.dataset
# #         return xs
# #
# #     def create_batches(self):
# #         if self.sort:
# #             self.batches = batch(self.data(), self.batch_size,
# #                                  self.batch_size_fn)
# #         else:
# #             self.examples = [self.get_element(i) for i in range(len(self.dataset))]
# #             self.batches = pool(self.data(), self.batch_size,
# #                                 self.sort_key, self.batch_size_fn,
# #                                 random_shuffler=self.random_shuffler,
# #                                 shuffle=self.shuffle,
# #                                 sort_within_batch=self.sort_within_batch)
#
# def make_data_iter(
#     dataset: Dataset,
#     batch_size: int,
#     batch_type: str = "sentence",
#     train: bool = False,
#     shuffle: bool = False,
# ) -> Iterator:
#     """
#     Returns a torchtext iterator for a torchtext dataset.
#
#     :param dataset: torchtext dataset containing sgn and optionally txt
#     :param batch_size: size of the batches the iterator prepares
#     :param batch_type: measure batch size by sentence count or by token count
#     :param train: whether it's training time, when turned off,
#         bucketing, sorting within batches and shuffling is disabled
#     :param shuffle: whether to shuffle the data before each epoch
#         (no effect if set to True for testing)
#     :return: torchtext iterator
#     """
#
#     batch_size_fn = token_batch_size_fn if batch_type == "token" else None
#
#     if train:
#         # optionally shuffle and sort during training
#         data_iter = BucketIterator(
#             repeat=False,
#             sort=False,
#             dataset=dataset,
#             batch_size=batch_size,
#             batch_size_fn=batch_size_fn,
#             train=True,
#             sort_within_batch=True,
#             sort_key=lambda x: len(x.sgn),
#             shuffle=shuffle,
#         )
#     else:
#         # don't sort/shuffle for validation/inference
#         data_iter = BucketIterator(
#             repeat=False,
#             dataset=dataset,
#             batch_size=batch_size,
#             batch_size_fn=batch_size_fn,
#             train=False,
#             sort=False,
#         )
#
#     return data_iter
# coding: utf-8
"""
Data module
"""
import torchvision
from tensorflow.python.ops.gen_dataset_ops import PrefetchDataset
from torchtext import data
# from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
from torch import Tensor

import itertools

from torchtext.data import Dataset, RawField, Field, Example


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        dataset_type,
        path, #: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if dataset_type == 'phoenix_2014_trans':
            if not isinstance(fields[0], (tuple, list)):
                fields = [
                    ("sequence", fields[0]),
                    ("signer", fields[1]),
                    ("sgn", fields[2]),
                    ("gls", fields[3]),
                    ("txt", fields[4]),
                ]
        else:
            if not isinstance(fields[0], (tuple, list)):
                fields = [
                    ("sequence", fields[0]),
                    ("signer", fields[1]),
                    ("sgn", fields[2]),
                    ("gls", fields[3]),
                ]

        image_encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
        image_encoder.eval()

        if dataset_type == 'phoenix_2014_trans':
            if not isinstance(path, list):
                path = [path]

            samples = {}
            for annotation_file in path:
                tmp = load_dataset_file(annotation_file)
                for s in tmp:
                    seq_id = s["name"]
                    if seq_id in samples:
                        assert samples[seq_id]["name"] == s["name"]
                        assert samples[seq_id]["signer"] == s["signer"]
                        assert samples[seq_id]["gloss"] == s["gloss"]
                        assert samples[seq_id]["text"] == s["text"]
                        samples[seq_id]["sign"] = torch.cat(
                            [samples[seq_id]["sign"], s["sign"]], axis=1
                        )
                    else:
                        samples[seq_id] = {
                            "name": s["name"],
                            "signer": s["signer"],
                            "gloss": s["gloss"],
                            "text": s["text"],
                            "sign": s["sign"],
                        }
        else:
            samples = {}
            examples = []

            for i, s in enumerate(itertools.islice(path, 0, len(path))):
                # seq_id = s["id"].numpy().decode('utf-8')
                # if seq_id in samples:
                #     assert samples[seq_id]["name"] == s["id"].numpy().decode('utf-8')
                #     assert samples[seq_id]["signer"] == s["signer"].numpy()
                #     assert samples[seq_id]["gloss"] == s["gloss_id"].numpy()
                #     samples[seq_id]["sign"] = torch.cat([samples[seq_id]["sign"], s["video"]], axis=1)
                # # else:
                seq_id = s["id"].numpy().decode('utf-8')
                samples[seq_id]=s
                # samples[s["id"].numpy().decode('utf-8')]={
                #                                              "name":s["id"].numpy().decode('utf-8'),"signer":s["signer"].numpy(),
                #                 "sign":image_encoder(Tensor(s["video"].numpy()).view(-1, s['video'].shape[3],
                #                                                                   s['video'].shape[1],
                #                                                                   s['video'].shape[2])),"gloss":s["gloss_id"].numpy()}

                # samples.append((s["id"].numpy().decode('utf-8'),s["signer"].numpy(),
                #                 image_encoder(Tensor(s["video"].numpy()).view(-1, s['video'].shape[3],
                #                                                                   s['video'].shape[1],
                #                                                                   s['video'].shape[2])),s["gloss_id"].numpy()))
                                # torch.from_numpy(s["video"].numpy())))
                # samples[seq_id] = {
                #     "name": s["id"].numpy().decode('utf-8'),
                #     "signer": s["signer"].numpy(),
                #     "gloss": s["gloss_id"].numpy(),
                #     "sign": torch.from_numpy(s["video"].numpy())#s["video"],
                # }
                # examples.append(
                #     Example.fromlist(
                #     [
                #             s["id"].numpy().decode('utf-8',),#.rstrip('\n'),
                #             s["signer"].numpy(),#.decode('utf-8').rstrip('\n'),
                #             image_encoder(Tensor(s["video"].numpy()).view(-1, s['video'].shape[3],
                #                                                                  s['video'].shape[1],
                #                                                                  s['video'].shape[2])),
                #             # torch.from_numpy(s["video"].numpy()), #s["video"], #+ 1e-8,
                #             s["gloss_id"].numpy(),  # .decode('utf-8'),
                #             # s["gloss"].numpy().decode('utf-8').strip().rstrip('\n'),
                #             # s["text"].numpy().decode('utf-8').strip().rstrip('\n'),
                #             # samples[seq_id]["id"],
                #             # samples[seq_id]["signer"],
                #             # # This is for numerical stability
                #             # samples[seq_id]["video"]+ 1e-8,
                #             # samples[seq_id]["gloss"],#.strip(),
                #             # samples[seq_id]["text"],#.strip(),
                #     ],
                #     fields,
                #     )
                # )
                if (i+1)%1000==0:
                    print(i+1)

        if dataset_type == 'phoenix_2014_trans':
            examples = []
            for s in samples:
                sample = samples[s]
                examples.append(
                    Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            # This is for numerical stability
                            sample["sign"] + 1e-8,
                            sample["gloss"].strip(),
                            sample["text"].strip(),
                        ],
                        fields,
                    )
                )
        else:               # TODO: Get it into the previous loop, trying to save space
            examples = []
            # for s in samples:
            #     sample = samples[s]
            #     examples.append(
            #         Example.fromlist(
            #             [
            #                 sample["name"].rstrip('\n'),
            #                 sample["signer"].rstrip('\n'),
            #                 # This is for numerical stability
            #                 sample["sign"], # + 1e-8,
            #                 sample["gloss"],
            #             ],
            #             fields,
            #         )
            #     )

            for i, s in enumerate(itertools.islice(path, 0, len(path))):
                examples.append(
                    Example.fromlist(
                        [
                            s["id"].numpy().decode('utf-8').rstrip('\n'),
                            s["signer"].numpy(),
                            # This is for numerical stability
                            torch.from_numpy(s["video"].numpy()), # + 1e-8,
                            s["gloss_id"],
                        ],
                        fields,
                    )
                )
                if (i+1)%1000==0:
                    print(i+1)

        # # samples = {}
        # examples = list()
        # limit = round(len(dataset)*0.5) if len(dataset) > 1000 else len(dataset)
        # for i,s in enumerate(itertools.islice(dataset, 0, limit)):
        #     # for annotation_file in path:
        #     #     tmp = load_dataset_file(annotation_file)
        #     #     for s in annotation_file:
        #     # seq_id = s["id"].numpy().decode('utf-8')
        #     # if seq_id in samples:
        #     #     assert samples[seq_id]["id"] == s["id"].numpy().decode('utf-8')
        #     #     assert samples[seq_id]["signer"] == s["signer"].numpy().decode('utf-8')
        #     #     assert samples[seq_id]["gloss"] == s["gloss"].numpy().decode('utf-8').strip()
        #     #     assert samples[seq_id]["text"] == s["text"].numpy().decode('utf-8').strip()
        #     #     samples[seq_id]["video"] = torch.cat([samples[seq_id]["video"], s["video"]], axis=1)
        #     # # else:
        #     # samples[seq_id] = {
        #     #     "id": s["id"].numpy().decode('utf-8'),
        #     #     "signer": s["signer"].numpy().decode('utf-8'),
        #     #     "gloss": s["gloss"].numpy().decode('utf-8').strip(),
        #     #     "text": s["text"].numpy().decode('utf-8').strip(),
        #     #     "video": s["video"],
        #     # }
        #     examples.append(
        #         Example.fromlist(
        #         [
        #                 s["id"].numpy().decode('utf-8').rstrip('\n'),
        #                 s["signer"].numpy().decode('utf-8').rstrip('\n'),
        #                 # s["gloss"].numpy().decode('utf-8'),
        #                 torch.from_numpy(s["video"].numpy()), #s["video"], #+ 1e-8,
        #                 s["gloss"].numpy().decode('utf-8').strip().rstrip('\n'),
        #                 s["text"].numpy().decode('utf-8').strip().rstrip('\n'),
        #                 # samples[seq_id]["id"],
        #                 # samples[seq_id]["signer"],
        #                 # # This is for numerical stability
        #                 # samples[seq_id]["video"]+ 1e-8,
        #                 # samples[seq_id]["gloss"],#.strip(),
        #                 # samples[seq_id]["text"],#.strip(),
        #         ],
        #         fields,
        #         )
        #     )

        # examples = []
        # for s in samples:
        #     sample = samples[s]
        #     examples.append(
        #         Example.fromlist(
        #             [
        #                 sample["id"],
        #                 sample["signer"],
        #                 # This is for numerical stability
        #                 sample["video"]+ 1e-8,
        #                 sample["gloss"],#.strip(),
        #                 sample["text"],#.strip(),
        #             ],
        #             fields,
        #         )
        #     )
        super().__init__(examples, fields, **kwargs)

# # coding: utf-8
# """
# Data module
# """
# from torchtext import data
# from torchtext.data import Field, RawField
# from typing import List, Tuple
# import pickle
# import gzip
# import torch
#
#
# def load_dataset_file(filename):
#     with gzip.open(filename, "rb") as f:
#         loaded_object = pickle.load(f)
#         return loaded_object
#
#
# class SignTranslationDataset(data.Dataset):
#     """Defines a dataset for machine translation."""
#
#     @staticmethod
#     def sort_key(ex):
#         return data.interleave_keys(len(ex.sgn), len(ex.txt))
#
#     def __init__(
#         self,
#         path: str,
#         fields: Tuple[RawField, RawField, Field, Field, Field],
#         **kwargs
#     ):
#         """Create a SignTranslationDataset given paths and fields.
#
#         Arguments:
#             path: Common prefix of paths to the data files for both languages.
#             exts: A tuple containing the extension to path for each language.
#             fields: A tuple containing the fields that will be used for data
#                 in each language.
#             Remaining keyword arguments: Passed to the constructor of
#                 data.Dataset.
#         """
#
#         if not isinstance(fields[0], (tuple, list)):
#             fields = [
#                 ("sequence", fields[0]),
#                 ("signer", fields[1]),
#                 ("sgn", fields[2]),
#                 ("gls", fields[3]),
#                 ("txt", fields[4]),
#             ]
#
#         if not isinstance(path, list):
#             path = [path]
#
#         samples = {}
#         for annotation_file in path:
#             tmp = load_dataset_file(annotation_file)
#             for s in tmp:
#                 seq_id = s["name"]
#                 if seq_id in samples:
#                     assert samples[seq_id]["name"] == s["name"]
#                     assert samples[seq_id]["signer"] == s["signer"]
#                     assert samples[seq_id]["gloss"] == s["gloss"]
#                     assert samples[seq_id]["text"] == s["text"]
#                     samples[seq_id]["sign"] = torch.cat(
#                         [samples[seq_id]["sign"], s["sign"]], axis=1
#                     )
#                 else:
#                     samples[seq_id] = {
#                         "name": s["name"],
#                         "signer": s["signer"],
#                         "gloss": s["gloss"],
#                         "text": s["text"],
#                         "sign": s["sign"],
#                     }
#
#         examples = []
#         for s in samples:
#             sample = samples[s]
#             examples.append(
#                 data.Example.fromlist(
#                     [
#                         sample["name"],
#                         sample["signer"],
#                         # This is for numerical stability
#                         sample["sign"] + 1e-8,
#                         sample["gloss"].strip(),
#                         sample["text"].strip(),
#                     ],
#                     fields,
#                 )
#             )
#         super().__init__(examples, fields, **kwargs)



################################ phoneix 17/5 ################################
# # coding: utf-8
# """
# Data module
# """
# from tensorflow.python.ops.gen_dataset_ops import PrefetchDataset
# from torchtext import data
# # from torchtext.data import Field, RawField
# from typing import List, Tuple
# import pickle
# import gzip
# import torch
#
# import itertools
#
# from torchtext.legacy.data import Dataset, RawField, Field, Example
#
#
# def load_dataset_file(filename):
#     with gzip.open(filename, "rb") as f:
#         loaded_object = pickle.load(f)
#         return loaded_object
#
#
# class SignTranslationDataset(Dataset):
#     """Defines a dataset for machine translation."""
#
#     @staticmethod
#     def sort_key(ex):
#         return data.interleave_keys(len(ex.sgn), len(ex.txt))
#
#     def __init__(
#         self,
#         path: str,
#         fields: Tuple[RawField, RawField, Field, Field, Field],
#         **kwargs
#     ):
#         """Create a SignTranslationDataset given paths and fields.
#
#         Arguments:
#             path: Common prefix of paths to the data files for both languages.
#             exts: A tuple containing the extension to path for each language.
#             fields: A tuple containing the fields that will be used for data
#                 in each language.
#             Remaining keyword arguments: Passed to the constructor of
#                 data.Dataset.
#         """
#         if not isinstance(fields[0], (tuple, list)):
#             fields = [
#                 ("sequence", fields[0]),
#                 ("signer", fields[1]),
#                 ("sgn", fields[2]),
#                 ("gls", fields[3]),
#                 ("txt", fields[4]),
#             ]
#
#         if not isinstance(path, list):
#             path = [path]
#
#         samples = {}
#         for annotation_file in path:
#             tmp = load_dataset_file(annotation_file)
#             for s in tmp:
#                 seq_id = s["name"]
#                 if seq_id in samples:
#                     assert samples[seq_id]["name"] == s["name"]
#                     assert samples[seq_id]["signer"] == s["signer"]
#                     assert samples[seq_id]["gloss"] == s["gloss"]
#                     assert samples[seq_id]["text"] == s["text"]
#                     samples[seq_id]["sign"] = torch.cat(
#                         [samples[seq_id]["sign"], s["sign"]], axis=1
#                     )
#                 else:
#                     samples[seq_id] = {
#                         "name": s["name"],
#                         "signer": s["signer"],
#                         "gloss": s["gloss"],
#                         "text": s["text"],
#                         "sign": s["sign"],
#                     }
#
#         examples = []
#         for s in samples:
#             sample = samples[s]
#             examples.append(
#                 Example.fromlist(
#                     [
#                         sample["name"],
#                         sample["signer"],
#                         # This is for numerical stability
#                         sample["sign"] + 1e-8,
#                         sample["gloss"].strip(),
#                         sample["text"].strip(),
#                     ],
#                     fields,
#                 )
#             )
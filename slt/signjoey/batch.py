import math
import random
import torch
import numpy as np

from slt.signjoey.helpers import DEVICE


class Batch:
    """
    Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
            self,
            dataset_type,  # TODO: Mine.
            torch_batch,
            txt_pad_index,
            sgn_dim,
            is_train: bool = False,
            use_cuda: bool = False,
            frame_subsampling_ratio: int = None,
            random_frame_subsampling: bool = None,
            random_frame_masking_ratio: float = None,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign), gls (gloss), and txt (text) length,
        masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        if dataset_type == 'phoenix_2014_trans':  # TODO: Mine.
            self.sequence = torch_batch.sequence
            self.signer = torch_batch.signer
            # Sign
            self.sgn, self.sgn_lengths = torch_batch.sgn

        else:  # TODO: Mine.
            self.sequence = torch_batch["sequence"]
            self.signer = torch_batch["signer"]

            # Sign
            # TODO: Check with the phoenix dataset to which dimension the sgn_lengths are referring to. V
            #  Ans: sgn_lengts is a tensor containing the number of frames in each video of the batch.
            #  and how sgn and sgn_lengths should look like.    V
            #  Ans: sgn is a padded batch of videos
            self.sgn, self.sgn_lengths = torch_batch["sgn"]

        # TODO: Conditional expression: False.  V
        # Here be dragons
        if frame_subsampling_ratio:

            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)

            for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):

                features = features.clone()
                if random_frame_subsampling and is_train:
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data = features[: length.long(), :]
                tmp_data = tmp_data[init_frame::frame_subsampling_ratio]
                tmp_sgn[idx, 0: tmp_data.shape[0]] = tmp_data
                tmp_sgn_lengths[idx] = tmp_data.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths

        # TODO: Conditional expression: False.  V
        if random_frame_masking_ratio and is_train:

            tmp_sgn = torch.zeros_like(self.sgn)
            num_mask_frames = ((self.sgn_lengths * random_frame_masking_ratio).floor().long())

            for idx, features in enumerate(self.sgn):
                features = features.clone()
                mask_frame_idx = np.random.permutation(int(self.sgn_lengths[idx].long().numpy()))[
                                 : num_mask_frames[idx]]
                features[mask_frame_idx, :] = 1e-8
                tmp_sgn[idx] = features

            self.sgn = tmp_sgn

        self.sgn_dim = sgn_dim
        self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        # TODO: Conditional expression: False.  V   ~ should be True
        #  to match it to the AUTSL attribute name. VVV
        # hasattr returns whether the object has an attribute with the given name.
        if hasattr(torch_batch, "txt") or "txt" in torch_batch:  # TODO: Addition for asynchronous dataset. V
            if dataset_type == 'phoenix_2014_trans':
                txt, txt_lengths = torch_batch.txt
            else:  # TODO: Mine.
                txt, txt_lengths = torch_batch["txt"]
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        # TODO: Conditional expression: False.  V   ~ should be True
        #  to match it to the AUTSL attribute name. VVV
        # hasattr returns whether the object has an attribute with the given name.
        if hasattr(torch_batch, "gls") or "gls" in torch_batch:  # TODO: Addition for asynchronous dataset. V
            if dataset_type == 'phoenix_2014_trans':
                self.gls, self.gls_lengths = torch_batch.gls
            else:  # TODO: Mine.
                self.gls, self.gls_lengths = torch_batch["gls"]
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU.

        :return:
        """
        self.sgn = self.sgn.cuda(DEVICE)
        self.sgn_mask = self.sgn_mask.cuda(DEVICE)

        if self.txt_input is not None:
            self.txt = self.txt.cuda(DEVICE)
            self.txt_mask = self.txt_mask.cuda(DEVICE)
            self.txt_input = self.txt_input.cuda(DEVICE)

    def make_cpu(self):  # TODO: Mine.
        """
        Move the batch back to CPU.

        :return:
        """
        self.sgn = self.sgn.detach().cpu()
        self.sgn_mask = self.sgn_mask.detach().cpu()

        if self.txt_input is not None:
            self.txt = self.txt.detach().cpu()
            self.txt_mask = self.txt_mask.detach().cpu()
            self.txt_input = self.txt_input.detach().cpu()

    def sort_by_sgn_lengths(self):  # TODO: Check if there is something to update here. Not needed. VVV
        """
        Sort by sgn length (descending) and return index to revert sort.

        :return:
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index

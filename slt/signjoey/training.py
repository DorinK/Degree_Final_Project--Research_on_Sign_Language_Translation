#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

# TODO: Add relevant imports.   V
import itertools
import sys
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Dataset

import argparse
import numpy as np
import os
import shutil
import time
import queue

from slt.signjoey.model import build_model
from slt.signjoey.batch import Batch
from slt.signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from slt.signjoey.model import SignModel
from slt.signjoey.prediction import validate_on_data
from slt.signjoey.loss import XentLoss
from slt.signjoey.data import load_data, make_data_iter
from slt.signjoey.builders import build_optimizer, build_scheduler, build_gradient_clipper
from slt.signjoey.prediction import test
from slt.signjoey.metrics import wer_single, wacc_single
from slt.signjoey.vocabulary import SIL_TOKEN, TextVocabulary, build_vocab, GlossVocabulary
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from slt.signjoey.helpers import DEVICE  # TODO: Use the relevant GPU device.   V
# from torchtext.data import Dataset
from typing import List, Dict

# TODO: Add relevant imports.   V
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig
import gc


# TODO: Adapt to AUTSL and ChicagoFSWild datasets.  VVV
# pylint: disable=too-many-instance-attributes
class TrainManager:
    """
    Manages training loop, validations, learning rate scheduling and early stopping.
    """

    def __init__(self, model: SignModel, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        # Get the training configurations.
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = make_model_dir(train_config["model_dir"], overwrite=train_config.get("overwrite", False))
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # input
        self.feature_size = (
            sum(config["data"]["feature_size"])
            if isinstance(config["data"]["feature_size"], list)
            else config["data"]["feature_size"]
        )

        # Get the dataset name.
        self.dataset_version = config["data"].get("version", "phoenix_2014_trans")

        # model
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self._log_parameters_list()
        # Check if we are doing only recognition or only translation or both
        self.do_recognition = (config["training"].get("recognition_loss_weight", 1.0) > 0.0)
        self.do_translation = (config["training"].get("translation_loss_weight", 1.0) > 0.0)

        # Get Recognition and Translation specific parameters
        if self.do_recognition:
            self._get_recognition_params(train_config=train_config)
        if self.do_translation:
            self._get_translation_params(train_config=train_config)

        # optimization
        self.last_best_lr = train_config.get("learning_rate", -1)
        # self.last_best_lr = train_config.get("learning_rate", -1) * 4 # TODO: Tried to run multi-GPU training and failed.
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))
        # TODO: Change eval_metric in the AUTSL config file from BLEU to WER, since bleu is used for translation
        #  tasks and not for recognition tasks. V
        # TODO: Change eval_metric in the ChicagoFSWild config file from BLEU to WAcc (Word Accuracy Rate). V
        self.eval_metric = train_config.get("eval_metric", "bleu")  # Get the evaluation metric.
        if self.eval_metric not in ["bleu", "chrf", "wer", "rouge", "wacc"]:  # TODO: Add WAcc to the list. V
            raise ValueError("Invalid setting for 'eval_metric': {}".format(self.eval_metric))
        self.early_stopping_metric = train_config.get("early_stopping_metric", "eval_metric")

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize early_stopping_metric
        # decides on how to find the early stopping point: ckpts are written when there's a new high/low
        # score for this metric
        if self.early_stopping_metric in [
            "ppl",
            "translation_loss",
            "recognition_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge", "wacc"]:   # TODO: Add WAcc too the list.  V
                # assert self.do_translation  # TODO: Issue here -> Commented out.  V
                self.minimize_metric = False
            else:  # eval metric that has to get minimized (not yet implemented)
                self.minimize_metric = True
        else:
            raise ValueError("Invalid setting for 'early_stopping_metric': {}".format(self.early_stopping_metric))

        # data_augmentation parameters
        self.frame_subsampling_ratio = config["data"].get("frame_subsampling_ratio", None)
        self.random_frame_subsampling = config["data"].get("random_frame_subsampling", None)
        self.random_frame_masking_ratio = config["data"].get("random_frame_masking_ratio", None)

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ValueError("Invalid segmentation level': {}".format(self.level))

        # Configurations of shuffling, epochs and batches
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        # self.batch_size = train_config["batch_size"] * 4  # TODO: Tried to run multi-GPU training and failed.
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)

        # Use GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            # TODO: Tried to run multi-GPU training and failed.
            # num_gpus = torch.cuda.device_count()
            # device_ids = list(range(4))
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            # self.model = self.model.to(device)
            self.model.cuda(DEVICE)  # TODO: Use the relevant GPU device.   V
            if self.do_translation:
                self.translation_loss_function.cuda(DEVICE)  # TODO: Use the relevant GPU device.   V
            if self.do_recognition:
                self.recognition_loss_function.cuda(DEVICE)  # TODO: Use the relevant GPU device.   V

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_txt_tokens = 0
        self.total_gls_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        # comparison function for scores
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )

    def _get_recognition_params(self, train_config) -> None:
        # NOTE (Cihan): The blank label is the silence index in the gloss vocabulary.
        #   There is an assertion in the GlossVocabulary class's __init__.
        #   This is necessary to do TensorFlow decoding, as it is hardcoded
        #   Currently it is hardcoded as 0.
        self.gls_silence_token = self.model.gls_vocab.stoi[SIL_TOKEN]
        assert self.gls_silence_token == 0

        # Set the recognition loss function.
        self.recognition_loss_function = torch.nn.CTCLoss(blank=self.gls_silence_token, zero_infinity=True)
        self.recognition_loss_weight = train_config.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = train_config.get("eval_recognition_beam_size", 1)

    def _get_translation_params(self, train_config) -> None:

        self.label_smoothing = train_config.get("label_smoothing", 0.0)

        # Set the translation loss function.
        self.translation_loss_function = XentLoss(pad_index=self.txt_pad_index, smoothing=self.label_smoothing)
        self.translation_normalization_mode = train_config.get("translation_normalization", "batch")

        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError("Invalid normalization {}.".format(self.translation_normalization_mode))

        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get("eval_translation_beam_size", 1)
        self.eval_translation_beam_alpha = train_config.get("eval_translation_beam_alpha", -1)
        self.translation_max_output_length = train_config.get("translation_max_output_length", None)

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.
        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens if self.do_translation else 0,
            "total_gls_tokens": self.total_gls_tokens if self.do_recognition else 0,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(state, model_path)

        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning("Wanted to delete old checkpoint %s but " "file does not exist.", to_delete)

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint
        symlink_update("{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir))

    def init_from_checkpoint(
            self,
            path: str,
            reset_best_ckpt: bool = False,
            reset_scheduler: bool = False,
            reset_optimizer: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also scheduler and optimizer states,
        see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (
                    model_checkpoint["scheduler_state"] is not None
                    and self.scheduler is not None
            ):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_txt_tokens = model_checkpoint["total_txt_tokens"]
        self.total_gls_tokens = model_checkpoint["total_gls_tokens"]

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_all_ckpt_scores = model_checkpoint["best_all_ckpt_scores"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda(DEVICE)  # TODO: Use the relevant GPU device.   V

    def train_and_validate_phoenix(self, train_data: Dataset, valid_data: Dataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """

        # Create iterator for the training set.
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            train=True,
            shuffle=self.shuffle,
        )

        epoch_no = None
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            # Set the model to training mode.
            self.model.train()
            start = time.time()
            total_valid_duration = 0
            count = self.batch_multiplier - 1

            if self.do_recognition:
                processed_gls_tokens = self.total_gls_tokens
                epoch_recognition_loss = 0
            if self.do_translation:
                processed_txt_tokens = self.total_txt_tokens
                epoch_translation_loss = 0

            # For each batch in the training set
            for batch in iter(train_iter):
                # reactivate training
                # create a Batch object from torchtext batch
                batch = Batch(
                    dataset_type=self.dataset_version,  # TODO: Add the dataset name as an argument.    V
                    is_train=True,
                    torch_batch=batch,
                    txt_pad_index=self.txt_pad_index,
                    sgn_dim=self.feature_size,
                    use_cuda=self.use_cuda,
                    frame_subsampling_ratio=self.frame_subsampling_ratio,
                    random_frame_subsampling=self.random_frame_subsampling,
                    random_frame_masking_ratio=self.random_frame_masking_ratio,
                )

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672
                update = count == 0

                recognition_loss, translation_loss = self._train_batch(batch, update=update)

                if self.do_recognition:
                    self.tb_writer.add_scalar("train/train_recognition_loss", recognition_loss, self.steps)
                    epoch_recognition_loss += recognition_loss.detach().cpu().numpy()

                if self.do_translation:
                    self.tb_writer.add_scalar("train/train_translation_loss", translation_loss, self.steps)
                    epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "step"
                        and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(epoch_no + 1, self.steps, )

                    if self.do_recognition:
                        elapsed_gls_tokens = (self.total_gls_tokens - processed_gls_tokens)
                        processed_gls_tokens = self.total_gls_tokens
                        log_out += "Batch Recognition Loss: {:10.6f} => ".format(recognition_loss)
                        log_out += "Gls Tokens per Sec: {:8.0f} || ".format(elapsed_gls_tokens / elapsed)

                    if self.do_translation:
                        elapsed_txt_tokens = (self.total_txt_tokens - processed_txt_tokens)
                        processed_txt_tokens = self.total_txt_tokens
                        log_out += "Batch Translation Loss: {:10.6f} => ".format(translation_loss)
                        log_out += "Txt Tokens per Sec: {:8.0f} || ".format(elapsed_txt_tokens / elapsed)

                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()
                    total_valid_duration = 0

                # TODO: Changed validation_freq from 100 to 1 for testing - Change back at the end. V
                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()
                    # TODO (Cihan): There must be a better way of passing
                    #   these recognition only and translation only parameters!
                    #   Maybe have a NamedTuple with optional fields?
                    #   Hmm... Future Cihan's problem.
                    val_res = validate_on_data(
                        model=self.model,
                        data=valid_data,
                        batch_size=self.eval_batch_size,
                        use_cuda=self.use_cuda,
                        batch_type=self.eval_batch_type,
                        dataset_version=self.dataset_version,
                        sgn_dim=self.feature_size,
                        txt_pad_index=self.txt_pad_index,
                        # Recognition Parameters
                        do_recognition=self.do_recognition,
                        recognition_loss_function=self.recognition_loss_function if self.do_recognition else None,
                        recognition_loss_weight=self.recognition_loss_weight if self.do_recognition else None,
                        recognition_beam_size=self.eval_recognition_beam_size if self.do_recognition else None,
                        # Translation Parameters
                        do_translation=self.do_translation,
                        translation_loss_function=self.translation_loss_function if self.do_translation else None,
                        translation_max_output_length=self.translation_max_output_length if self.do_translation else None,
                        level=self.level if self.do_translation else None,
                        translation_loss_weight=self.translation_loss_weight if self.do_translation else None,
                        translation_beam_size=self.eval_translation_beam_size if self.do_translation else None,
                        translation_beam_alpha=self.eval_translation_beam_alpha if self.do_translation else None,
                        frame_subsampling_ratio=self.frame_subsampling_ratio,
                    )
                    self.model.train()

                    if self.do_recognition:
                        # Log Losses and ppl
                        self.tb_writer.add_scalar(
                            "valid/valid_recognition_loss",
                            val_res["valid_recognition_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar("valid/wer", val_res["valid_scores"]["wer"], self.steps)
                        self.tb_writer.add_scalars(
                            "valid/wer_scores",
                            val_res["valid_scores"]["wer_scores"],
                            self.steps,
                        )

                    if self.do_translation:
                        self.tb_writer.add_scalar(
                            "valid/valid_translation_loss",
                            val_res["valid_translation_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar("valid/valid_ppl", val_res["valid_ppl"], self.steps)

                        # Log Scores
                        self.tb_writer.add_scalar("valid/chrf", val_res["valid_scores"]["chrf"], self.steps)
                        self.tb_writer.add_scalar("valid/rouge", val_res["valid_scores"]["rouge"], self.steps)
                        self.tb_writer.add_scalar("valid/bleu", val_res["valid_scores"]["bleu"], self.steps)
                        self.tb_writer.add_scalars(
                            "valid/bleu_scores",
                            val_res["valid_scores"]["bleu_scores"],
                            self.steps,
                        )

                    if self.early_stopping_metric == "recognition_loss":
                        assert self.do_recognition
                        ckpt_score = val_res["valid_recognition_loss"]
                    elif self.early_stopping_metric == "translation_loss":
                        assert self.do_translation
                        ckpt_score = val_res["valid_translation_loss"]
                    elif self.early_stopping_metric in ["ppl", "perplexity"]:
                        assert self.do_translation
                        ckpt_score = val_res["valid_ppl"]
                    else:
                        ckpt_score = val_res["valid_scores"][self.eval_metric]

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_all_ckpt_scores = val_res["valid_scores"]
                        self.best_ckpt_iteration = self.steps
                        self.logger.info("Hooray! New best validation result [%s]!", self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if (
                            self.scheduler is not None
                            and self.scheduler_step_at == "validation"
                    ):
                        prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                        self.scheduler.step(ckpt_score)
                        now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

                        if prev_lr != now_lr:
                            if self.last_best_lr != prev_lr:
                                self.stop = True

                    # append to validation report
                    self._add_report(
                        dataset=self.dataset_version,
                        valid_scores=val_res["valid_scores"],
                        valid_recognition_loss=val_res["valid_recognition_loss"] if self.do_recognition else None,
                        valid_translation_loss=val_res["valid_translation_loss"] if self.do_translation else None,
                        valid_ppl=val_res["valid_ppl"] if self.do_translation else None,
                        eval_metric=self.eval_metric,
                        new_best=new_best,
                    )
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                        "Recognition Beam Size: %d\t"
                        "Translation Beam Size: %d\t"
                        "Translation Beam Alpha: %d\n\t"
                        "Recognition Loss: %4.5f\t"
                        "Translation Loss: %4.5f\t"
                        "PPL: %4.5f\n\t"
                        "Eval Metric: %s\n\t"
                        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        epoch_no + 1,
                        self.steps,
                        valid_duration,
                        self.eval_recognition_beam_size if self.do_recognition else -1,
                        self.eval_translation_beam_size if self.do_translation else -1,
                        self.eval_translation_beam_alpha if self.do_translation else -1,
                        val_res["valid_recognition_loss"] if self.do_recognition else -1,
                        val_res["valid_translation_loss"] if self.do_translation else -1,
                        val_res["valid_ppl"] if self.do_translation else -1,
                        self.eval_metric.upper(),
                        # WER
                        val_res["valid_scores"]["wer"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["del_rate"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["ins_rate"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["sub_rate"] if self.do_recognition else -1,
                        # BLEU
                        val_res["valid_scores"]["bleu"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu1"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu2"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu3"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu4"] if self.do_translation else -1,
                        # Other
                        val_res["valid_scores"]["chrf"] if self.do_translation else -1,
                        val_res["valid_scores"]["rouge"] if self.do_translation else -1,
                    )

                    valid_seq = [s for s in valid_data.sequence]

                    self._log_examples(
                        sequences=valid_seq,  # TODO: Problem here -> Fixed.    V
                        gls_references=val_res["gls_ref"] if self.do_recognition else None,
                        gls_hypotheses=val_res["gls_hyp"] if self.do_recognition else None,
                        txt_references=val_res["txt_ref"] if self.do_translation else None,
                        txt_hypotheses=val_res["txt_hyp"] if self.do_translation else None,
                    )

                    # valid_seq = [s for s in valid_data.sequence]  # TODO: Move up.    V

                    # store validation set outputs and references
                    if self.do_recognition:
                        self._store_outputs("dev.hyp.gls", valid_seq, val_res["gls_hyp"], "gls")
                        self._store_outputs("references.dev.gls", valid_seq, val_res["gls_ref"])

                    if self.do_translation:
                        self._store_outputs("dev.hyp.txt", valid_seq, val_res["txt_hyp"], "txt")
                        self._store_outputs("references.dev.txt", valid_seq, val_res["txt_ref"])

                if self.stop:
                    break

            if self.stop:
                if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "validation"
                        and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

            self.logger.info(
                "Epoch %3d: Total Training Recognition Loss %.2f "
                " Total Training Translation Loss %.2f ",
                epoch_no + 1,
                epoch_recognition_loss if self.do_recognition else -1,
                epoch_translation_loss if self.do_translation else -1,
            )
        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)

        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()  # close Tensorboard writer

    # TODO: Adapt the training procedure to AUTSL dataset.  V
    def train_and_validate_autsl(self, train_data, valid_data) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        epoch_no = None
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            # Set the model to training mode.
            self.model.train()
            self.model.image_encoder.train()

            start = time.time()
            total_valid_duration = 0
            count = self.batch_multiplier - 1

            if self.do_recognition:
                processed_gls_tokens = self.total_gls_tokens
                epoch_recognition_loss = 0
            if self.do_translation:
                processed_txt_tokens = self.total_txt_tokens
                epoch_translation_loss = 0

            index = 0
            while index < len(train_data):  # TODO: Handle the last chunk of train_data.    V

                batch_size = self.batch_size if len(train_data) - index >= self.batch_size else len(train_data) - index

                valid = 0
                total = 0
                sequence = []
                signer = []
                samples = []
                sgn_lengths = []
                gls = []
                gls_lengths = [int(1)] * batch_size

                for i, datum in enumerate(itertools.islice(train_data, index, index + batch_size)):
                    if datum['video'].shape[0] <= 115:
                        sequence.append(datum['id'].numpy().decode('utf-8'))
                        signer.append(datum["signer"].numpy())
                        samples.append(
                            Tensor(datum['video'].numpy()).view(-1, datum['video'].shape[3], datum['video'].shape[1],
                                                                datum['video'].shape[2]))
                        sgn_lengths.append(datum['video'].shape[0])
                        gls.append([int(self.model.gls_vocab.stoi[datum['gloss_id'].numpy()])])
                        valid += 1
                    total += 1

                while valid < batch_size:
                    for i, datum in enumerate(
                            itertools.islice(train_data, index + total, index + total + (batch_size - valid))):
                        if datum['video'].shape[0] <= 115:
                            sequence.append(datum['id'].numpy().decode('utf-8'))
                            signer.append(datum["signer"].numpy())
                            samples.append(Tensor(datum['video'].numpy()).view(-1, datum['video'].shape[3],
                                                                               datum['video'].shape[1],
                                                                               datum['video'].shape[2]))
                            sgn_lengths.append(datum['video'].shape[0])
                            gls.append([int(self.model.gls_vocab.stoi[datum['gloss_id'].numpy()])])
                            valid += 1
                        total += 1

                    # TODO: Handle a case where there are no more samples and/or valid samples to complete the batch.   V
                    if index + total + (batch_size - valid) >= len(train_data):
                        if valid < batch_size:
                            # print("index: ", index)
                            # print("batch_size: ", batch_size)
                            # print("valid: ", valid)
                            # print("total: ", total)
                            gls_lengths = [int(1)] * valid
                            # print("In")
                            break

                # print("Out")
                if valid == 0:
                    # print("Valid is 0")
                    index += total
                    # print('so far:', index)
                    del sequence, signer, samples, sgn_lengths, gls, gls_lengths
                    gc.collect()
                    continue

                # print("Valid is not 0")

                sgn = []
                for k, sample in enumerate(samples, 1):
                    # print('index:', k)
                    # print('sequence:', sequence[k - 1])
                    # print('sgn_lengths:', sgn_lengths[k - 1])
                    # print('signer:', signer[k - 1])
                    # print('gls:', gls[k - 1])
                    # print()
                    out = self.model.image_encoder(sample.cuda(DEVICE))
                    sgn.append(out.detach().cpu())
                    sample.detach().cpu()
                    del sample, out
                    gc.collect()

                pad_sgn = pad_sequence(sgn, batch_first=True, padding_value=0)
                batch_prep = {'sequence': sequence, 'signer': signer, 'sgn': (pad_sgn, Tensor(sgn_lengths)),
                              'gls': (Tensor(gls), Tensor(gls_lengths))}

                index += total
                if self.steps % self.logging_freq == 0:
                    print('so far:', index)
                # print('index:',index)
                # print('sequence:',sequence)
                # print('sgn_lengths:',sgn_lengths)
                # print('signer:',signer)
                # print('gls:',gls)
                del sequence, signer, samples, sgn_lengths, gls, gls_lengths, sgn, pad_sgn
                gc.collect()

                # reactivate training
                # create a Batch object from torchtext batch
                batch = Batch(
                    dataset_type=self.dataset_version,
                    is_train=True,
                    torch_batch=batch_prep,
                    txt_pad_index=self.txt_pad_index,
                    sgn_dim=self.feature_size,
                    use_cuda=self.use_cuda,
                    frame_subsampling_ratio=self.frame_subsampling_ratio,
                    random_frame_subsampling=self.random_frame_subsampling,
                    random_frame_masking_ratio=self.random_frame_masking_ratio,
                )

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672
                update = count == 0

                recognition_loss, translation_loss = self._train_batch(batch, update=update)

                if self.do_recognition:
                    self.tb_writer.add_scalar("train/train_recognition_loss", recognition_loss, self.steps)
                    epoch_recognition_loss += recognition_loss.detach().cpu().numpy()

                if self.do_translation:
                    self.tb_writer.add_scalar("train/train_translation_loss", translation_loss, self.steps)
                    epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "step"
                        and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(epoch_no + 1, self.steps)

                    if self.do_recognition:
                        elapsed_gls_tokens = (self.total_gls_tokens - processed_gls_tokens)
                        processed_gls_tokens = self.total_gls_tokens
                        log_out += "Batch Recognition Loss: {:10.6f} => ".format(recognition_loss)
                        log_out += "Gls Tokens per Sec: {:8.0f} || ".format(elapsed_gls_tokens / elapsed)

                    if self.do_translation:
                        elapsed_txt_tokens = (self.total_txt_tokens - processed_txt_tokens)
                        processed_txt_tokens = self.total_txt_tokens
                        log_out += "Batch Translation Loss: {:10.6f} => ".format(translation_loss)
                        log_out += "Txt Tokens per Sec: {:8.0f} || ".format(elapsed_txt_tokens / elapsed)

                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()
                    total_valid_duration = 0

                # TODO: Changed validation_freq from 100 to 1 for testing - Change back at the end. V
                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()
                    # TODO (Cihan): There must be a better way of passing
                    #   these recognition only and translation only parameters!
                    #   Maybe have a NamedTuple with optional fields?
                    #   Hmm... Future Cihan's problem.
                    val_res = validate_on_data(
                        model=self.model,
                        data=valid_data,
                        batch_size=self.eval_batch_size,
                        use_cuda=self.use_cuda,
                        batch_type=self.eval_batch_type,
                        dataset_version=self.dataset_version,
                        sgn_dim=self.feature_size,
                        txt_pad_index=self.txt_pad_index,
                        # Recognition Parameters
                        do_recognition=self.do_recognition,
                        recognition_loss_function=self.recognition_loss_function if self.do_recognition else None,
                        recognition_loss_weight=self.recognition_loss_weight if self.do_recognition else None,
                        recognition_beam_size=self.eval_recognition_beam_size if self.do_recognition else None,
                        # Translation Parameters
                        do_translation=self.do_translation,
                        translation_loss_function=self.translation_loss_function if self.do_translation else None,
                        translation_max_output_length=self.translation_max_output_length if self.do_translation else None,
                        level=self.level if self.do_translation else None,
                        translation_loss_weight=self.translation_loss_weight if self.do_translation else None,
                        translation_beam_size=self.eval_translation_beam_size if self.do_translation else None,
                        translation_beam_alpha=self.eval_translation_beam_alpha if self.do_translation else None,
                        frame_subsampling_ratio=self.frame_subsampling_ratio,
                    )
                    self.model.train()
                    self.model.image_encoder.train()

                    if self.do_recognition:
                        # Log Losses and ppl
                        self.tb_writer.add_scalar(
                            "valid/valid_recognition_loss",
                            val_res["valid_recognition_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar("valid/wer", val_res["valid_scores"]["wer"], self.steps)
                        self.tb_writer.add_scalars(
                            "valid/wer_scores",
                            val_res["valid_scores"]["wer_scores"],
                            self.steps,
                        )

                    if self.do_translation:
                        self.tb_writer.add_scalar(
                            "valid/valid_translation_loss",
                            val_res["valid_translation_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar("valid/valid_ppl", val_res["valid_ppl"], self.steps)

                        # Log Scores
                        self.tb_writer.add_scalar("valid/chrf", val_res["valid_scores"]["chrf"], self.steps)
                        self.tb_writer.add_scalar("valid/rouge", val_res["valid_scores"]["rouge"], self.steps)
                        self.tb_writer.add_scalar("valid/bleu", val_res["valid_scores"]["bleu"], self.steps)
                        self.tb_writer.add_scalars(
                            "valid/bleu_scores",
                            val_res["valid_scores"]["bleu_scores"],
                            self.steps,
                        )

                    if self.early_stopping_metric == "recognition_loss":
                        assert self.do_recognition
                        ckpt_score = val_res["valid_recognition_loss"]
                    elif self.early_stopping_metric == "translation_loss":
                        assert self.do_translation
                        ckpt_score = val_res["valid_translation_loss"]
                    elif self.early_stopping_metric in ["ppl", "perplexity"]:
                        assert self.do_translation
                        ckpt_score = val_res["valid_ppl"]
                    else:
                        ckpt_score = val_res["valid_scores"][self.eval_metric]

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_all_ckpt_scores = val_res["valid_scores"]
                        self.best_ckpt_iteration = self.steps
                        self.logger.info("Hooray! New best validation result [%s]!",
                                         self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if (
                            self.scheduler is not None
                            and self.scheduler_step_at == "validation"
                    ):
                        prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                        self.scheduler.step(ckpt_score)
                        now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

                        if prev_lr != now_lr:
                            if self.last_best_lr != prev_lr:
                                self.stop = True

                    # append to validation report
                    self._add_report(
                        dataset=self.dataset_version,   # Add the dataset name as an argument.  V
                        valid_scores=val_res["valid_scores"],
                        valid_recognition_loss=val_res["valid_recognition_loss"] if self.do_recognition else None,
                        valid_translation_loss=val_res["valid_translation_loss"] if self.do_translation else None,
                        valid_ppl=val_res["valid_ppl"] if self.do_translation else None,
                        eval_metric=self.eval_metric,
                        new_best=new_best,
                    )
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                        "Recognition Beam Size: %d\t"
                        "Translation Beam Size: %d\t"
                        "Translation Beam Alpha: %d\n\t"
                        "Recognition Loss: %4.5f\t"
                        "Translation Loss: %4.5f\t"
                        "PPL: %4.5f\n\t"
                        "Eval Metric: %s\n\t"
                        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        epoch_no + 1,
                        self.steps,
                        valid_duration,
                        self.eval_recognition_beam_size if self.do_recognition else -1,
                        self.eval_translation_beam_size if self.do_translation else -1,
                        self.eval_translation_beam_alpha if self.do_translation else -1,
                        val_res["valid_recognition_loss"] if self.do_recognition else -1,
                        val_res["valid_translation_loss"] if self.do_translation else -1,
                        val_res["valid_ppl"] if self.do_translation else -1,
                        self.eval_metric.upper(),
                        # WER
                        val_res["valid_scores"]["wer"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["del_rate"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["ins_rate"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["sub_rate"] if self.do_recognition else -1,
                        # BLEU
                        val_res["valid_scores"]["bleu"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu1"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu2"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu3"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu4"] if self.do_translation else -1,
                        # Other
                        val_res["valid_scores"]["chrf"] if self.do_translation else -1,
                        val_res["valid_scores"]["rouge"] if self.do_translation else -1,
                    )

                    # TODO: Adapt to AUTSL dataset. V
                    valid_seq = [datum['id'].numpy().decode('utf-8') for datum in itertools.islice(valid_data, len(valid_data))]

                    self._log_examples(
                        sequences=valid_seq,  # TODO: Problem here -> Fixed.    V
                        gls_references=val_res["gls_ref"] if self.do_recognition else None,
                        gls_hypotheses=val_res["gls_hyp"] if self.do_recognition else None,
                        txt_references=val_res["txt_ref"] if self.do_translation else None,
                        txt_hypotheses=val_res["txt_hyp"] if self.do_translation else None,
                    )

                    # valid_seq = [s for s in valid_data.sequence]  # moved up

                    # store validation set outputs and references
                    if self.do_recognition:
                        self._store_outputs("dev.hyp.gls", valid_seq, val_res["gls_hyp"], "gls")
                        self._store_outputs("references.dev.gls", valid_seq, val_res["gls_ref"])

                    if self.do_translation:
                        self._store_outputs("dev.hyp.txt", valid_seq, val_res["txt_hyp"], "txt")
                        self._store_outputs("references.dev.txt", valid_seq, val_res["txt_ref"])

                batch.make_cpu()  # TODO: Move batch back to CPU.   V
                del batch  # TODO: Remove unnecessary variable. V
                gc.collect()  # TODO: Call the garbage collector.   V

                if self.stop:
                    break

            if self.stop:
                if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "validation"
                        and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

            self.logger.info(
                "Epoch %3d: Total Training Recognition Loss %.2f "
                " Total Training Translation Loss %.2f ",
                epoch_no + 1,
                epoch_recognition_loss if self.do_recognition else -1,
                epoch_translation_loss if self.do_translation else -1,
            )
        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)
        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()  # close Tensorboard writer

    # TODO: Adapt the training procedure to ChicagoFSWild dataset.  V
    def train_and_validate_chicago(self, train_data, valid_data) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        epoch_no = None
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            # Set the model to training mode.
            self.model.train()
            self.model.image_encoder.train()

            start = time.time()
            total_valid_duration = 0
            count = self.batch_multiplier - 1

            if self.do_recognition:
                processed_gls_tokens = self.total_gls_tokens
                epoch_recognition_loss = 0
            if self.do_translation:
                processed_txt_tokens = self.total_txt_tokens
                epoch_translation_loss = 0

            index = 0
            while index < len(train_data):  # TODO: Handle the last chunk of train_data.    V

                batch_size = self.batch_size if len(train_data) - index >= self.batch_size else len(train_data) - index

                valid = 0
                total = 0
                sequence = []
                signer = []
                samples = []
                sgn_lengths = []
                txt = []
                txt_lengths = []

                for i, datum in enumerate(itertools.islice(train_data, index, index + batch_size)):
                    if datum['video'].shape[0] <= 131:
                        sequence.append(datum['id'].numpy().decode('utf-8'))
                        signer.append(datum["signer"].numpy().decode('utf-8'))
                        samples.append(
                            Tensor(datum['video'].numpy()).view(-1, datum['video'].shape[3], datum['video'].shape[1],
                                                                datum['video'].shape[2]))
                        sgn_lengths.append(datum['video'].shape[0])
                        sample_txt = datum['text'].numpy().decode('utf-8').split()
                        txt.append(
                            [2] + [int(self.model.txt_vocab.stoi[t]) for t in sample_txt] + (
                                    [1] * (10 - len(sample_txt) - 1)))
                        txt_lengths.append(len(sample_txt))
                        valid += 1
                    total += 1

                while valid < batch_size:
                    for i, datum in enumerate(
                            itertools.islice(train_data, index + total, index + total + (batch_size - valid))):
                        if datum['video'].shape[0] <= 131:
                            sequence.append(datum['id'].numpy().decode('utf-8'))
                            signer.append(datum["signer"].numpy())
                            samples.append(
                                Tensor(datum['video'].numpy()).view(-1, datum['video'].shape[3],
                                                                    datum['video'].shape[1],
                                                                    datum['video'].shape[2]))
                            sgn_lengths.append(datum['video'].shape[0])
                            sample_txt = datum['text'].numpy().decode('utf-8').split()
                            txt.append(
                                [2] + [int(self.model.txt_vocab.stoi[t]) for t in sample_txt] + (
                                        [1] * (10 - len(sample_txt) - 1)))
                            txt_lengths.append(len(sample_txt))
                            valid += 1
                        total += 1

                    # TODO: Handle a case where there are no more samples and/or valid samples to complete the batch.   V
                    if index + total + (batch_size - valid) >= len(train_data):
                        if valid < batch_size:
                            # print("index: ", index)
                            # print("batch_size: ", batch_size)
                            # print("valid: ", valid)
                            # print("total: ", total)
                            # # print(len(txt_lengths))
                            # # txt_lengths = [int(1)] * valid
                            # # print(len(txt_lengths))
                            # print("In")
                            break

                # print("Out")
                if valid == 0:
                    # print("Valid is 0")
                    index += total
                    # print('so far:', index)
                    del sequence, signer, samples, sgn_lengths, txt, txt_lengths
                    gc.collect()
                    continue

                # print("Valid is not 0")

                sgn = []
                for k, sample in enumerate(samples, 1):
                    # print('index:', k)
                    # print('sequence:', sequence[k - 1])
                    # print('sgn_lengths:', sgn_lengths[k - 1])
                    # print('signer:', signer[k - 1])
                    # print('txt:', txt[k - 1])
                    # print('txt_lengths:', txt_lengths[k - 1])
                    # print()
                    out = self.model.image_encoder(sample.cuda(DEVICE))
                    sgn.append(out.detach().cpu())
                    sample.detach().cpu()
                    del sample, out
                    gc.collect()

                pad_sgn = pad_sequence(sgn, batch_first=True, padding_value=1)
                batch_prep = {'sequence': sequence, 'signer': signer, 'sgn': (pad_sgn, Tensor(sgn_lengths)),
                              'txt': (torch.LongTensor(txt), torch.LongTensor(txt_lengths))}

                index += total
                if self.steps % self.logging_freq == 0:
                    print('so far:', index)

                del sequence, signer, samples, sgn_lengths, txt, txt_lengths, sgn, pad_sgn
                gc.collect()

                # reactivate training
                # create a Batch object from torchtext batch
                batch = Batch(
                    dataset_type=self.dataset_version,
                    is_train=True,
                    torch_batch=batch_prep,
                    txt_pad_index=self.txt_pad_index,
                    sgn_dim=self.feature_size,
                    use_cuda=self.use_cuda,
                    frame_subsampling_ratio=self.frame_subsampling_ratio,
                    random_frame_subsampling=self.random_frame_subsampling,
                    random_frame_masking_ratio=self.random_frame_masking_ratio,
                )

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672
                update = count == 0

                recognition_loss, translation_loss = self._train_batch(
                    batch, update=update
                )

                if self.do_recognition:
                    self.tb_writer.add_scalar("train/train_recognition_loss", recognition_loss, self.steps)
                    epoch_recognition_loss += recognition_loss.detach().cpu().numpy()

                if self.do_translation:
                    self.tb_writer.add_scalar("train/train_translation_loss", translation_loss, self.steps)
                    epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "step"
                        and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(epoch_no + 1, self.steps)

                    if self.do_recognition:
                        elapsed_gls_tokens = (self.total_gls_tokens - processed_gls_tokens)
                        processed_gls_tokens = self.total_gls_tokens
                        log_out += "Batch Recognition Loss: {:10.6f} => ".format(recognition_loss)
                        log_out += "Gls Tokens per Sec: {:8.0f} || ".format(elapsed_gls_tokens / elapsed)

                    if self.do_translation:
                        elapsed_txt_tokens = (self.total_txt_tokens - processed_txt_tokens)
                        processed_txt_tokens = self.total_txt_tokens
                        log_out += "Batch Translation Loss: {:10.6f} => ".format(translation_loss)
                        log_out += "Txt Tokens per Sec: {:8.0f} || ".format(elapsed_txt_tokens / elapsed)

                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()
                    total_valid_duration = 0

                # TODO: Changed validation_freq from 100 to 1 for testing - Change back at the end. V
                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()
                    # TODO (Cihan): There must be a better way of passing
                    #   these recognition only and translation only parameters!
                    #   Maybe have a NamedTuple with optional fields?
                    #   Hmm... Future Cihan's problem.
                    val_res = validate_on_data(
                        model=self.model,
                        data=valid_data,
                        # image_encoder=self.image_encoder,
                        batch_size=self.eval_batch_size,
                        use_cuda=self.use_cuda,
                        batch_type=self.eval_batch_type,
                        dataset_version=self.dataset_version,
                        sgn_dim=self.feature_size,
                        txt_pad_index=self.txt_pad_index,
                        # Recognition Parameters
                        do_recognition=self.do_recognition,
                        recognition_loss_function=self.recognition_loss_function if self.do_recognition else None,
                        recognition_loss_weight=self.recognition_loss_weight if self.do_recognition else None,
                        recognition_beam_size=self.eval_recognition_beam_size if self.do_recognition else None,
                        # Translation Parameters
                        do_translation=self.do_translation,
                        translation_loss_function=self.translation_loss_function if self.do_translation else None,
                        translation_max_output_length=self.translation_max_output_length if self.do_translation else None,
                        level=self.level if self.do_translation else None,
                        translation_loss_weight=self.translation_loss_weight if self.do_translation else None,
                        translation_beam_size=self.eval_translation_beam_size if self.do_translation else None,
                        translation_beam_alpha=self.eval_translation_beam_alpha if self.do_translation else None,
                        frame_subsampling_ratio=self.frame_subsampling_ratio,
                    )
                    self.model.train()
                    self.model.image_encoder.train()

                    if self.do_recognition:
                        # Log Losses and ppl
                        self.tb_writer.add_scalar(
                            "valid/valid_recognition_loss",
                            val_res["valid_recognition_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar("valid/wer", val_res["valid_scores"]["wer"], self.steps)
                        self.tb_writer.add_scalars(
                            "valid/wer_scores",
                            val_res["valid_scores"]["wer_scores"],
                            self.steps,
                        )

                    if self.do_translation:
                        self.tb_writer.add_scalar(
                            "valid/valid_translation_loss",
                            val_res["valid_translation_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar("valid/valid_ppl", val_res["valid_ppl"], self.steps)

                        # TODO: Adjust the evaluation metrics to ChicagoFSWild dataset. V
                        self.tb_writer.add_scalar(
                            "valid/wacc", val_res["valid_scores"]["wacc"], self.steps
                        )
                        self.tb_writer.add_scalars(
                            "valid/wer_scores", val_res["valid_scores"]["wer_scores"], self.steps
                        )
                        self.tb_writer.add_scalar(
                            "valid/seq_accuracy", val_res["valid_scores"]["seq_accuracy"], self.steps
                        )

                    if self.early_stopping_metric == "recognition_loss":
                        assert self.do_recognition
                        ckpt_score = val_res["valid_recognition_loss"]
                    elif self.early_stopping_metric == "translation_loss":
                        assert self.do_translation
                        ckpt_score = val_res["valid_translation_loss"]
                    elif self.early_stopping_metric in ["ppl", "perplexity"]:
                        assert self.do_translation
                        ckpt_score = val_res["valid_ppl"]
                    else:
                        ckpt_score = val_res["valid_scores"][self.eval_metric]

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_all_ckpt_scores = val_res["valid_scores"]
                        self.best_ckpt_iteration = self.steps
                        self.logger.info("Hooray! New best validation result [%s]!", self.early_stopping_metric, )
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if (
                            self.scheduler is not None
                            and self.scheduler_step_at == "validation"
                    ):
                        prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                        self.scheduler.step(ckpt_score)
                        now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

                        if prev_lr != now_lr:
                            if self.last_best_lr != prev_lr:
                                self.stop = True

                    # append to validation report
                    self._add_report(
                        dataset=self.dataset_version,
                        valid_scores=val_res["valid_scores"],
                        valid_recognition_loss=val_res["valid_recognition_loss"] if self.do_recognition else None,
                        valid_translation_loss=val_res["valid_translation_loss"] if self.do_translation else None,
                        valid_ppl=val_res["valid_ppl"] if self.do_translation else None,
                        eval_metric=self.eval_metric,
                        new_best=new_best,
                    )
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(  # TODO: Adjust the evaluation metrics to ChicagoFSWild dataset.  V
                        "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                        "Recognition Beam Size: %d\t"
                        "Translation Beam Size: %d\t"
                        "Translation Beam Alpha: %d\n\t"
                        "Recognition Loss: %4.5f\t"
                        "Translation Loss: %4.5f\t"
                        "PPL: %4.5f\n\t"
                        "Eval Metric: %s\n\t"
                        "WAcc (Word Accuracy Rate) %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
                        "Sequence Accuracy %.2f",
                        epoch_no + 1,
                        self.steps,
                        valid_duration,
                        self.eval_recognition_beam_size if self.do_recognition else -1,
                        self.eval_translation_beam_size if self.do_translation else -1,
                        self.eval_translation_beam_alpha if self.do_translation else -1,
                        val_res["valid_recognition_loss"] if self.do_recognition else -1,
                        val_res["valid_translation_loss"] if self.do_translation else -1,
                        val_res["valid_ppl"] if self.do_translation else -1,
                        self.eval_metric.upper(),
                        # WAcc + WER scores
                        val_res["valid_scores"]["wacc"] if self.do_translation else -1,
                        val_res["valid_scores"]["wer_scores"]["del_rate"] if self.do_translation else -1,
                        val_res["valid_scores"]["wer_scores"]["ins_rate"] if self.do_translation else -1,
                        val_res["valid_scores"]["wer_scores"]["sub_rate"] if self.do_translation else -1,
                        val_res["valid_scores"]["seq_accuracy"] if self.do_translation else -1,
                    )

                    # TODO: Adapt to ChicagoFSWild dataset. V
                    valid_seq = [datum['id'].numpy().decode('utf-8') for datum in itertools.islice(valid_data, len(valid_data))]

                    self._log_examples(
                        sequences=valid_seq,  # TODO: Problem here -> Fixed.    V
                        gls_references=val_res["gls_ref"] if self.do_recognition else None,
                        gls_hypotheses=val_res["gls_hyp"] if self.do_recognition else None,
                        txt_references=val_res["txt_ref"] if self.do_translation else None,
                        txt_hypotheses=val_res["txt_hyp"] if self.do_translation else None,
                    )

                    # valid_seq = [s for s in valid_data.sequence]  # TODO: Move up.    V

                    # store validation set outputs and references
                    if self.do_recognition:
                        self._store_outputs("dev.hyp.gls", valid_seq, val_res["gls_hyp"], "gls")
                        self._store_outputs("references.dev.gls", valid_seq, val_res["gls_ref"])

                    if self.do_translation:
                        self._store_outputs("dev.hyp.txt", valid_seq, val_res["txt_hyp"], "txt")
                        self._store_outputs("references.dev.txt", valid_seq, val_res["txt_ref"])

                batch.make_cpu()  # TODO: Move batch back to CPU.   V
                del batch  # TODO: Remove unnecessary variable. V
                gc.collect()  # TODO: Call the garbage collector.   V

                if self.stop:
                    break

            if self.stop:
                if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "validation"
                        and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

            self.logger.info(
                "Epoch %3d: Total Training Recognition Loss %.2f "
                " Total Training Translation Loss %.2f ",
                epoch_no + 1,
                epoch_recognition_loss if self.do_recognition else -1,
                epoch_translation_loss if self.do_translation else -1,
            )
        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)

        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()  # close Tensorboard writer

    def _train_batch(self, batch: Batch, update: bool = True) -> (Tensor, Tensor):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return normalized_recognition_loss: Normalized recognition loss
        :return normalized_translation_loss: Normalized translation loss
        """

        recognition_loss, translation_loss = self.model.get_loss_for_batch(
            batch=batch,
            recognition_loss_function=self.recognition_loss_function if self.do_recognition else None,
            translation_loss_function=self.translation_loss_function if self.do_translation else None,
            recognition_loss_weight=self.recognition_loss_weight if self.do_recognition else None,
            translation_loss_weight=self.translation_loss_weight if self.do_translation else None,
        )

        # normalize translation loss
        if self.do_translation:

            if self.translation_normalization_mode == "batch":
                txt_normalization_factor = batch.num_seqs
            elif self.translation_normalization_mode == "tokens":
                txt_normalization_factor = batch.num_txt_tokens
            else:
                raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

            # division needed since loss.backward sums the gradients until updated
            normalized_translation_loss = translation_loss / (txt_normalization_factor * self.batch_multiplier)

        else:
            normalized_translation_loss = 0

        # TODO (Cihan): Add Gloss Token normalization (?)
        #   I think they are already being normalized by batch
        #   I need to think about if I want to normalize them by token.
        if self.do_recognition:
            normalized_recognition_loss = recognition_loss / self.batch_multiplier
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss + normalized_translation_loss
        # compute gradients
        total_loss.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens
        if self.do_translation:
            self.total_txt_tokens += batch.num_txt_tokens

        return normalized_recognition_loss, normalized_translation_loss

    def _add_report(  # TODO: Check this function.  V
            self,
            dataset: str,   # Add the dataset name as a parameter.  V
            valid_scores: Dict,
            valid_recognition_loss: float,
            valid_translation_loss: float,
            valid_ppl: float,
            eval_metric: str,
            new_best: bool = False,
    ) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: Dictionary of validation scores
        :param valid_recognition_loss: validation loss (sum over whole validation set)
        :param valid_translation_loss: validation loss (sum over whole validation set)
        :param valid_ppl: validation perplexity
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if new_best:
            self.last_best_lr = current_lr

        if current_lr < self.learning_rate_min:
            self.stop = True

        if dataset == 'ChicagoFSWild':  # TODO: Adjust the evaluation metrics to ChicagoFSWild dataset. V
            with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
                opened_file.write(
                    "Steps: {}\t"
                    "Recognition Loss: {:.5f}\t"
                    "Translation Loss: {:.5f}\t"
                    "PPL: {:.5f}\t"
                    "Eval Metric: {}\t"
                    "WAcc (Word Accuracy Rate) {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                    "Sequence Accuracy {:.2f}\t"
                    "LR: {:.8f}\t{}\n".format(
                        self.steps,
                        valid_recognition_loss if self.do_recognition else -1,
                        valid_translation_loss if self.do_translation else -1,
                        valid_ppl if self.do_translation else -1,
                        eval_metric,
                        # WAcc + WER scores
                        valid_scores["wacc"] if self.do_translation else -1,
                        valid_scores["wer_scores"]["del_rate"] if self.do_translation else -1,
                        valid_scores["wer_scores"]["ins_rate"] if self.do_translation else -1,
                        valid_scores["wer_scores"]["sub_rate"] if self.do_translation else -1,
                        valid_scores["seq_accuracy"] if self.do_translation else -1,
                        # Other
                        current_lr,
                        "*" if new_best else "",
                    )
                )
        else:
            with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
                opened_file.write(
                    "Steps: {}\t"
                    "Recognition Loss: {:.5f}\t"
                    "Translation Loss: {:.5f}\t"
                    "PPL: {:.5f}\t"
                    "Eval Metric: {}\t"
                    "WER {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                    "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
                    "CHRF {:.2f}\t"
                    "ROUGE {:.2f}\t"
                    "LR: {:.8f}\t{}\n".format(
                        self.steps,
                        valid_recognition_loss if self.do_recognition else -1,
                        valid_translation_loss if self.do_translation else -1,
                        valid_ppl if self.do_translation else -1,
                        eval_metric,
                        # WER
                        valid_scores["wer"] if self.do_recognition else -1,
                        valid_scores["wer_scores"]["del_rate"]
                        if self.do_recognition
                        else -1,
                        valid_scores["wer_scores"]["ins_rate"]
                        if self.do_recognition
                        else -1,
                        valid_scores["wer_scores"]["sub_rate"]
                        if self.do_recognition
                        else -1,
                        # BLEU
                        valid_scores["bleu"] if self.do_translation else -1,
                        valid_scores["bleu_scores"]["bleu1"] if self.do_translation else -1,
                        valid_scores["bleu_scores"]["bleu2"] if self.do_translation else -1,
                        valid_scores["bleu_scores"]["bleu3"] if self.do_translation else -1,
                        valid_scores["bleu_scores"]["bleu4"] if self.do_translation else -1,
                        # Other
                        valid_scores["chrf"] if self.do_translation else -1,
                        valid_scores["rouge"] if self.do_translation else -1,
                        current_lr,
                        "*" if new_best else "",
                    )
                )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters() if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(  # TODO: Check this function.    V
            self,
            sequences: List[str],
            gls_references: List[str],
            gls_hypotheses: List[str],
            txt_references: List[str],
            txt_hypotheses: List[str],
    ) -> None:
        """
        Log `self.num_valid_log` number of samples from valid.

        :param sequences: sign video sequence names (list of strings)
        :param txt_hypotheses: decoded txt hypotheses (list of strings)
        :param txt_references: decoded txt references (list of strings)
        :param gls_hypotheses: decoded gls hypotheses (list of strings)
        :param gls_references: decoded gls references (list of strings)
        """

        if self.do_recognition:
            # assert len(gls_references) == len(gls_hypotheses) # TODO: Uncomment.  V
            num_sequences = len(gls_hypotheses)
        if self.do_translation:
            # assert len(txt_references) == len(txt_hypotheses) # TODO: Uncomment.  V
            num_sequences = len(txt_hypotheses)

        rand_idx = np.sort(np.random.permutation(num_sequences)[: self.num_valid_log])
        self.logger.info("Logging Recognition and Translation Outputs")
        self.logger.info("=" * 120)

        for ri in rand_idx:

            self.logger.info("Logging Sequence: %s", sequences[ri])

            if self.do_recognition:
                gls_res = wer_single(r=gls_references[ri], h=gls_hypotheses[ri])
                self.logger.info("\tGloss Reference :\t%s", gls_res["alignment_out"]["align_ref"])
                self.logger.info("\tGloss Hypothesis:\t%s", gls_res["alignment_out"]["align_hyp"])
                self.logger.info("\tGloss Alignment :\t%s", gls_res["alignment_out"]["alignment"])

            if self.do_recognition and self.do_translation:
                self.logger.info("\t" + "-" * 116)

            if self.do_translation:
                if self.dataset_version == "ChicagoFSWild":  # TODO: Use WAcc instead of WER.   V
                    txt_res = wacc_single(r=txt_references[ri], h=txt_hypotheses[ri])
                else:
                    txt_res = wer_single(r=txt_references[ri], h=txt_hypotheses[ri])
                self.logger.info("\tText Reference  :\t%s", txt_res["alignment_out"]["align_ref"])
                self.logger.info("\tText Hypothesis :\t%s", txt_res["alignment_out"]["align_hyp"])
                self.logger.info("\tText Alignment  :\t%s", txt_res["alignment_out"]["alignment"])

            self.logger.info("=" * 120)

    def _store_outputs(
            self, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
    ) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                opened_file.write("{}|{}\n".format(seq, hyp))


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """

    # Load the configuration file.
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # TODO: Tried to run on multiple GPUs in parallel and failed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # TODO: Handle each dataset individually.   V

    if cfg["data"]["version"] == 'phoenix_2014_trans':
        # Load the dataset and create the corresponding vocabs
        train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"])

    # TODO: Adapt to the asynchronous dataset structure of AUTSL ~ ~ (Check in data.py that I didn't miss anything) V
    elif cfg["data"]["version"] == 'autsl':

        config = SignDatasetConfig(name="include-videos", version="1.0.0", include_video=True, fps=30)
        autsl = tfds.load(name='autsl',  builder_kwargs=dict(config=config), shuffle_files=True)  # TODO: Check shuffle! 7/9    V
        train_data, dev_data, test_data = autsl['train'], autsl['validation'], autsl['test']

        # Set the maximal size of the gloss vocab and the minimum frequency to each item in it.
        gls_max_size = cfg["data"].get("gls_voc_limit", sys.maxsize)
        gls_min_freq = cfg["data"].get("gls_voc_min_freq", 1)

        # Get the vocabs if already exists, otherwise set them to None.
        gls_vocab_file = cfg["data"].get("gls_vocab", None)
        txt_vocab_file = cfg["data"].get("txt_vocab", None)

        # TODO: Create vocabularies using the relevant classes. V
        # Build the gloss vocab based on the training set.
        gls_vocab = build_vocab(
            version=cfg["data"]["version"],
            field="gls",
            min_freq=gls_min_freq,
            max_size=gls_max_size,
            dataset=train_data,
            vocab_file=gls_vocab_file,
        )
        # Next, build the text vocab based on the training set.
        txt_vocab = TextVocabulary(tokens=txt_vocab_file)

    else:  # TODO: Adapt to the asynchronous dataset structure of ChicagoFSWild.    V

        config = SignDatasetConfig(name="new-setup", version="1.0.0", include_video=True, resolution=(640, 360))
        chicagofswild = tfds.load(name='chicago_fs_wild', builder_kwargs=dict(config=config), shuffle_files=True)
        train_data, dev_data, test_data = chicagofswild['train'], chicagofswild['validation'], chicagofswild['test']

        # Set the maximal size of the gloss vocab and text vocab and the minimum frequency to each item in them.
        gls_max_size = cfg["data"].get("gls_voc_limit", sys.maxsize)
        gls_min_freq = cfg["data"].get("gls_voc_min_freq", 1)
        txt_max_size = cfg["data"].get("txt_voc_limit", sys.maxsize)
        txt_min_freq = cfg["data"].get("txt_voc_min_freq", 1)

        # Get the vocabs if already exists, otherwise set them to None.
        gls_vocab_file = cfg["data"].get("gls_vocab", None)
        txt_vocab_file = cfg["data"].get("txt_vocab", None)

        # Build the gloss vocab based on the training set.
        gls_vocab = build_vocab(
            version=cfg["data"]["version"],
            field="gls",
            min_freq=gls_min_freq,
            max_size=gls_max_size,
            dataset=train_data,
            vocab_file=gls_vocab_file,
        )
        # Build the txt vocab based on the training set.
        txt_vocab = build_vocab(
            version=cfg["data"]["version"],
            field="txt",
            min_freq=txt_min_freq,
            max_size=txt_max_size,
            dataset=train_data,
            vocab_file=txt_vocab_file,
        )

    # TODO: Update translation_loss_weight to 0.0 and recognition_loss_weight to 1.0.   V
    # Get from the configuration file the losses weights.
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0

    # TODO: Update features_size.   V
    #  In Phoenix it's (frames, features) and in AUTSL it's a video rep, so tensor has 4 dimensions.
    #  For example, (58,512,512,3).
    # build model and load parameters into it
    model = build_model(
        dataset=cfg["data"]["version"],  # TODO: Add the dataset name as an argument.   V
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    # TODO: Adapt to AUTSL and ChicagoFSWild datasets.  VVV
    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # TODO: It may be necessary to adjust according to changes made in the trainer -> No need.  V
    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    # TODO: It may be necessary to adjust according to changes made in the trainer -> No need.  V
    # log all entries of config
    log_cfg(cfg, trainer.logger)

    if cfg["data"]["version"] == 'phoenix_2014_trans':
        log_data_info(
            train_data=train_data,
            valid_data=dev_data,
            test_data=test_data,
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            logging_function=trainer.logger.info,
        )

    trainer.logger.info(str(model))

    # store the vocabs
    gls_vocab_file = "{}/gls.vocab".format(cfg["training"]["model_dir"])
    gls_vocab.to_file(gls_vocab_file)
    txt_vocab_file = "{}/txt.vocab".format(cfg["training"]["model_dir"])
    txt_vocab.to_file(txt_vocab_file)

    # TODO: Calculate the number of parameters of the model.    V
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('SLT_total_params:', pytorch_total_params)
    print('SLT_total_trained_params:', pytorch_total_trained_params)

    # TODO: Adapt the 'train_and_validate' function to AUTSL and ChicagoFSWild datasets.    V
    #  Handle each dataset individually.    V
    # train the model
    if cfg["data"]["version"] == 'phoenix_2014_trans':
        trainer.train_and_validate_phoenix(train_data=train_data, valid_data=dev_data)
    elif cfg["data"]["version"] == 'autsl':
        trainer.train_and_validate_autsl(train_data=train_data, valid_data=dev_data)
    else:
        trainer.train_and_validate_chicago(train_data=train_data, valid_data=dev_data)

    # Delete to speed things up as we don't need training data anymore
    del train_data, dev_data, test_data

    # TODO: It may be necessary to adjust according to changes made in the trainer -> No need.  V
    # predict with the best model on validation and test (if test data is available)
    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)  # TODO: Update!  VVV


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)

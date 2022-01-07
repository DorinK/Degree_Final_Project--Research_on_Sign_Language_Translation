import os
import time

import numpy as np
import torch

from utils import Bar
from utils.evaluation.averagemeter import AverageMeter
from utils.evaluation.classification import performance
from utils.misc import (
    is_show,
    save_pred,
)

from utils.vizutils import viz_gt_pred

# ----------------------------------------------------------------
# monkey patch for progress bar on SLURM
if True:
    #  disabling in interactive mode
    def writeln(self, line):
        on_slurm = os.environ.get("SLURM_JOB_ID", False)
        if self.file and (self.is_tty() or on_slurm):
            self.clearln()
            end = "\n" if on_slurm else ""
            print(line, end=end, file=self.file)
            self.file.flush()

    Bar.writeln = writeln
# ----------------------------------------------------------------


# Combined train/val epoch
def do_epoch(
        setname,
        loader,
        model,
        criterion,
        epochno=-1,
        optimizer=None,
        num_classes=None,
        debug=False,
        checkpoint=None,
        mean=torch.Tensor([0.5, 0.5, 0.5]),
        std=torch.Tensor([1.0, 1.0, 1.0]),
        feature_dim=1024,
        save_logits=False,
        save_features=False,
        num_figs=100,
        topk=[1],
        save_feature_dir="",
        save_fig_dir="",
):
    assert setname == "train" or setname == "val" or setname == "test"  # TODO: Adjusting.  V

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter()]

    perfs = []
    for k in topk:
        perfs.append(AverageMeter())  # TODO: AverageMeter() - Computes and stores the average and current value.

    if save_logits:
        all_logits = torch.Tensor(loader.dataset.__len__(), num_classes)
    if save_features:
        all_features = torch.Tensor(loader.dataset.__len__(), feature_dim)

    if setname == "train":
        model.train()
    elif setname == "val":
        model.eval()
    elif setname == "test":  # TODO: Add test set option.   V
        model.eval()

    end = time.time()

    gt_win, pred_win, fig_gt_pred = None, None, None
    bar = Bar("E%d" % (epochno + 1), max=len(loader))

    for i, data in enumerate(loader):

        if data.get("gpu_collater", False):
            #  We handle collation on the GPU to enable faster data augmentation

            with torch.no_grad():
                data["rgb"] = data["rgb"].cuda()
                collater_kwargs = {}
                if isinstance(loader.dataset, torch.utils.data.ConcatDataset):
                    cat_datasets = loader.dataset.datasets
                    collater = cat_datasets[0].gpu_collater
                    cat_datasets = {type(x).__name__.lower(): x for x in cat_datasets}
                    collater_kwargs["concat_datasets"] = cat_datasets
                else:
                    collater = loader.dataset.gpu_collater
                data = collater(minibatch=data, **collater_kwargs)

        # measure data loading time
        data_time.update(time.time() - end)

        inputs = data["rgb"]
        targets = data["class"]

        inputs_cuda = inputs.cuda()
        targets_cuda = targets.cuda()

        # forward pass
        outputs_cuda = model(inputs_cuda)

        # compute the loss
        logits = outputs_cuda["logits"].data.cpu()
        loss = criterion(outputs_cuda["logits"], targets_cuda)
        topk_acc = performance(logits, targets, topk=topk)  # TODO: performance - Returns the accuracy at top-k over a batch.

        for ki, acc in enumerate(topk_acc):
            perfs[ki].update(acc, inputs.size(0))

        losses[0].update(loss.item(), inputs.size(0))

        # generate predictions
        if save_logits:
            all_logits[data["index"]] = logits
        if save_features:
            all_features[data["index"]] = outputs_cuda["embds"].squeeze().data.cpu()  # TODO

        if (debug or is_show(num_figs, i, len(loader))):
            fname = "pred_%s_epoch%02d_iter%05d" % (setname, epochno, i)
            save_path = save_fig_dir / fname
            gt_win, pred_win, fig_gt_pred = viz_gt_pred(
                inputs,
                logits,
                targets,
                mean,
                std,
                data,
                gt_win,
                pred_win,
                fig_gt_pred,
                save_path=save_path,
                show=debug,
            )

        # compute gradient and do optim step
        if setname == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.1f}s | Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:} | Perf: {perf:}".format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=", ".join([f"{losses[i].avg:.3f}" for i in range(len(losses))]),
            perf=", ".join([f"{perfs[i].avg:.3f}" for i in range(len(perfs))]),
        )
        bar.next()

    bar.finish()

    # from sklearn.preprocessing import MultiLabelBinarizer
    # mlb = MultiLabelBinarizer()
    # mlb.fit_transform([(1, 2), (3,)])
    # save outputs

    if save_logits or save_features:
        meta = {
            "clip_gt": np.asarray(loader.dataset.get_set_classes()),
            "clip_ix": loader.dataset.valid,
            "video_names": loader.dataset.get_all_videonames(),
        }

    if save_logits:
        save_pred( all_logits, checkpoint=save_feature_dir, filename="preds.mat", meta=meta)
    if save_features:
        save_pred(all_features, checkpoint=save_feature_dir, filename="features.mat", meta=meta)

    return losses, perfs

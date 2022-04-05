import fastai
import tensorboard
from fastai.basics import *
from fastai.callback.fp16 import ModelToHalf
from fastai.callback.hook import hook_output
from fastai.vision.all import *
from torch.utils.tensorboard import SummaryWriter


class TensorBoardBaseCallback(Callback):
    order = Recorder.order + 1
    "Base class for tensorboard callbacks"

    def __init__(self):
        self.run_projector = False

    def after_pred(self):
        if self.run_projector:
            self.feat = _add_projector_features(self.learn, self.h, self.feat)

    def after_validate(self):
        if not self.run_projector:
            return
        self.run_projector = False
        self._remove()
        _write_projector_embedding(self.learn, self.writer, self.feat)

    def after_fit(self):
        if self.run:
            self.writer.close()

    def _setup_projector(self):
        self.run_projector = True
        self.h = hook_output(self.learn.model[1][1] if not self.layer else self.layer)
        self.feat = {}

    def _setup_writer(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def __del__(self):
        self._remove()

    def _remove(self):
        if getattr(self, "h", None):
            self.h.remove()


class TensorBoardCallback(TensorBoardBaseCallback):
    "Saves model topology, losses & metrics for tensorboard and tensorboard projector during training"

    def __init__(
        self,
        log_dir=None,
        trace_model=True,
        log_preds=True,
        n_preds=9,
        projector=False,
        layer=None,
    ):
        super().__init__()
        store_attr()

    def before_fit(self):
        self.run = (
            not hasattr(self.learn, "lr_finder")
            and not hasattr(self, "gather_preds")
            and rank_distrib() == 0
        )
        if not self.run:
            return
        self._setup_writer()
        if self.trace_model:
            if hasattr(self.learn, "mixed_precision"):
                raise Exception(
                    "Can't trace model in mixed precision, pass `trace_model=False` or don't use FP16."
                )
            b = self.dls.one_batch()
            self.learn._split(b)
            self.writer.add_graph(self.model, *self.xb)

    def after_batch(self):
        self.writer.add_scalar("train_loss", self.smooth_loss, self.train_iter)
        for i, h in enumerate(self.opt.hypers):
            for k, v in h.items():
                self.writer.add_scalar(f"{k}_{i}", v, self.train_iter)

    def after_epoch(self):
        for n, v in zip(self.recorder.metric_names[2:-1], self.recorder.log[2:-1]):
            self.writer.add_scalar(n, v, self.train_iter)
        if self.log_preds:
            b = self.dls.valid.one_batch()
            self.learn.one_batch(0, b)
            preds = getattr(self.loss_func, "activation", noop)(self.pred)
            out = getattr(self.loss_func, "decodes", noop)(preds)
            x, y, its, outs = self.dls.valid.show_results(
                b, out, show=False, max_n=self.n_preds
            )
            tensorboard_log(x, y, its, outs, self.writer, self.train_iter)

    def before_validate(self):
        if self.projector:
            self._setup_projector()


@typedispatch
def tensorboard_log(x: TensorImage, y: TensorCategory, samples, outs, writer, step):
    fig, axs = get_grid(len(samples), return_fig=True)
    for i in range(2):
        axs = [b.show(ctx=c) for b, c in zip(samples.itemgot(i), axs)]
    axs = [
        r.show(ctx=c, color="green" if b == r else "red")
        for b, r, c in zip(samples.itemgot(1), outs.itemgot(0), axs)
    ]
    writer.add_figure("Sample results", fig, step)

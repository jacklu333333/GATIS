import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import strategy, ddp


class ComplexStrategy(strategy.Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ddp = ddp.DDPStrategy(*args, **kwargs)

    def setup(self, model):
        self._ddp.setup(model)

    def train(self, model):
        self._ddp.train(model)

    def test(self, model):
        self._ddp.test(model)

    def predict(self, model):
        self._ddp.predict(model)

    def save_checkpoint(self, checkpoint):
        self._ddp.save_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        self._ddp.load_checkpoint(checkpoint)

    def get_train_dataloader(self):
        return self._ddp.get_train_dataloader()

    def get_val_dataloaders(self):
        return self._ddp.get_val_dataloaders()

    def get_test_dataloaders(self):
        return self._ddp.get_test_dataloaders()

    def get_predict_dataloader(self):
        return self._ddp.get_predict_dataloader()

    def barrier(self, name: str = None):
        self._ddp.barrier(name)

    def broadcast(self, obj, src=0):
        self._ddp.broadcast(obj, src)

    def reduce(self, tensor, dst=None, op=None):
        self._ddp.reduce(tensor, dst, op)

    def all_reduce(self, tensor, op=None):
        self._ddp.all_reduce(tensor, op)

    def gather(self, tensor, dst=0):
        self._ddp.gather(tensor, dst)

    def scatter(self, tensor, src=0):
        self._ddp.scatter(tensor, src)

    def all_gather(self, tensor):
        self._ddp.all_gather(tensor)

    def sync_barrier(self):
        self._ddp.sync_barrier()

    def barrier(self, name: str = None):
        self._ddp.barrier(name)

    def broadcast(self, obj, src=0):
        self._ddp.broadcast(obj, src)

    def reduce(self, tensor, dst=None, op=None):
        self._ddp.reduce(tensor, dst, op)

    def all_reduce(self, tensor, op=None):
        self._ddp.all_reduce(tensor, op)

    def gather(self, tensor, dst=0):
        self._ddp.gather(tensor, dst)

    # def scatter(self,
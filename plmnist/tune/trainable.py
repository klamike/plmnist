from typing import Optional
from pathlib import Path

from ray.tune import Trainable
from plmnist.plmnist import build_model, run_training, test

from plmnist.tune.config import STEP_EPOCHS, RAY_RESULTS_DIR
from plmnist.tune.utils import hide_logs


# now the main part: defining the tune.Trainable class
class MNISTTrainable(Trainable):
    # two methods are required: `setup()` and `step()`
    #   `setup()` is called once, when initializing the trainable (or when resetting it)
    #   `step()` is called at each tuning step

    # this class also implements `save_checkpoint()`, `load_checkpoint()`, and `reset_config()`
    #   to allow for saving and restoring the state of the trainable
    #   this is useful for using fancier search/scheduling algorithms that can do early stopping/resuming.

    _session_name: Optional[str] = None  # will be set once the experiment is started

    def setup(self, config: dict):
        hide_logs()
        # create the model and the pytorch lightning trainer
        self.seed = config["seed"]
        self.config = config
        self.trainer, self.model = build_model(
            max_epochs=STEP_EPOCHS,
            verbose=False,
            log_path=(
                # if session_name is given, save checkpoints persistently
                # otherwise, checkpoints will be deleted upon tuning exit
                (RAY_RESULTS_DIR / self._session_name)
                if self._session_name is not None
                else self.logdir
            ),
            **self.config,
        )
        self.resume_ckpt_path = None

    def step(self):
        # set max_epochs appropriately for this step
        self.trainer.fit_loop.max_epochs = STEP_EPOCHS * (self.training_iteration + 1)

        # run training for STEP_EPOCHS epochs, potentially resuming from a checkpoint
        run_training(self.trainer, self.model, ckpt_path=self.resume_ckpt_path)

        # clear the resume_ckpt_path so we don't accidentally resume from the old checkpoint next step
        self.resume_ckpt_path = None

        # test the model and return the results dictionary
        results = test(self.trainer, self.seed, verbose=False)
        return results

    def reset_config(self, new_config):
        # to avoid some overhead in creating the Trainable class,
        #   we can define the reset_config() method.
        #   since we don't edit the Trainable's state during step(), we can just call setup() again.
        self.setup(new_config)

        return True  # must return True

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[dict]:
        # save the model checkpoint
        ckpt_path = Path(checkpoint_dir).resolve() / "model.ckpt"
        self.trainer.save_checkpoint(ckpt_path)

        # return a dictionary here that contains a path to the model checkpoint
        #   note you can return the checkpoint itself, but this should be nicer
        #   since the dictionary will be pickled and (potentially) sent over the network.
        #   you can also return `None`, which will instead call `load_checkpoint` using `checkpoint_dir`
        return {
            "model_path": ckpt_path,
            "config": self.config,
        }

    def load_checkpoint(self, checkpoint: dict):
        # load the model checkpoint
        self.reset_config(checkpoint["config"])

        # make sure to set `resume_ckpt_path` so the next step will load the weights from the given checkpoint
        self.resume_ckpt_path = checkpoint["model_path"]

    # add a resource request for GPU training. this is low-memory, so 8 can easily fit on one GPU.
    def default_resource_request(self):
        return {"cpu": 1, "gpu": 0.125}

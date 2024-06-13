def hide_logs():
    import logging, warnings, os

    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

    # do not use SLURMEnvironment
    from pytorch_lightning.plugins.environments import SLURMEnvironment

    SLURMEnvironment.detect = lambda: False

    # needs to be run on each worker
    warnings.filterwarnings("ignore", "Checkpoint directory .* exists and is not empty")
    for module in logging.Logger.manager.loggerDict:
        if module.startswith(
            (
                "lightning",
                "pytorch",
                "ray.train._internal.storage",
                "ray.tune.execution.experiment_state",
            )
        ):
            logging.getLogger(module).setLevel(logging.FATAL)


def make_trial_name():
    from datetime import datetime, timedelta
    from plmnist.tune.config import MAX_HOURS

    now = datetime.now()
    end = now + timedelta(hours=MAX_HOURS)
    name = "mnist_" + now.strftime("%Y%m%d-%H%M%S")

    return name, end

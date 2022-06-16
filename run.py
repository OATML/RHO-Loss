import os

import hydra
from omegaconf import DictConfig, omegaconf
from src.utils.utils import save_repo_status

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # log repo state
    commit = save_repo_status(os.getcwd())
    with omegaconf.open_dict(config):
        config["git_repo_hash"] = commit 

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(
            config,
            fields=(
                "trainer",
                "selection_method",
                "model",
                "irreducible_loss_generator",
                "datamodule",
                "callbacks",
                "logger",
                "seed",
                "optimizer",
            ),
            resolve=True,
        )

    # Train model
    from src.train import train

    return train(config)


if __name__ == "__main__":
    main()
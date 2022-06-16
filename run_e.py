import hydra
from omegaconf import DictConfig


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
    from src.train_e import train

    return train(config)


if __name__ == "__main__":
    main()

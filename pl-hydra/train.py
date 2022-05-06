'''
Modified from https://github.com/phlippe/uvadlc_notebooks.git
'''

import dotenv
import hydra
from omegaconf import DictConfig
import os
# Set the visible GPUs (curent machine has 16 GPUS [0-15])
os.environ["CUDA_VISIBLE_DEVICES"]="11"

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()

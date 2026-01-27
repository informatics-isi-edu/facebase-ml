"""
This module defines configurations for dataset collections that can be used in different model runs.
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Configure a list of datasets by specifying the RID and version of each dataset that goes into the collection.

# CIFAR-10 Split datasets (localhost catalog 4)
datasets_training = [DatasetSpecConfig(rid="3XM", version="0.4.0")]  # Training dataset with 5 images
datasets_complete = [DatasetSpecConfig(rid="3X4", version="0.2.0")]  # Complete dataset with 10 images

# Create configurations and store them into hydra-zen store.
# Note that the name of the group has to match the name of the argument in the main function that will be
# instantiated to the configuration value.
datasets_store = store(group="datasets")
datasets_store(datasets_training, name="cifar10_training")
datasets_store(datasets_complete, name="cifar10_complete")
datasets_store(datasets_training, name="default_dataset")


"""This module defines configurations for the deriva_ml package."""

from hydra_zen import  store
from deriva_ml import DerivaMLConfig

# Create two alternative DerivaML configurations and store them into hydra-zen store.
deriva_store = store(group="deriva_ml")
deriva_store(DerivaMLConfig, name="default_deriva",
             hostname="localhost",
             catalog_id=4,
             use_minid=False)
deriva_store(DerivaMLConfig, name="eye-ai",
             hostname="www.eye-ai.org",
             catalog_id="eye-ai")

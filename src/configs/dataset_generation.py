"""
Dataset generation script configurations.

Each config wraps a generation function from src/scripts/ with its parameters.
These are registered under the "script_config" hydra group and invoked via
`uv run deriva-ml-run +experiment=<name>` where the experiment overrides
script_config instead of model_config.
"""
from hydra_zen import store, builds
from scripts.generate_e155_dev_sample import generate_e155_dev_sample

script_store = store(group="script_config")

E155DevSampleConfig = builds(
    generate_e155_dev_sample,
    source_dataset_rids=["A9-D272"],
    source_version="0.3.0",
    include_tables=["file"],
    element_table="file",
    filter_name="random_sample",
    filter_params={"n": 100, "seed": 42},
    output_description=(
        "100-element random sample from the E15.5 subset (A9-D272) for "
        "rapid development iteration. Seed=42 for reproducibility."
    ),
    output_types=["Development"],
    parent_dataset_rid="A9-D272",
    zen_partial=True,
)

script_store(E155DevSampleConfig, name="e155_dev_sample")

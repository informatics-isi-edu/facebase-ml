"""
Dataset generation script configurations.

Each config wraps a generation function from src/scripts/ with its parameters.
These are registered under the "script_config" hydra group and invoked via
`uv run deriva-ml-run +experiment=<name>` where the experiment overrides
script_config instead of model_config.
"""
from hydra_zen import store, builds
from scripts.generate_subset import generate_subset
from scripts.annotate_phenotype import annotate_phenotype

script_store = store(group="script_config")

# ---------------------------------------------------------------------------
# E15.5 dev sample — 100 random files for quick iteration
# ---------------------------------------------------------------------------
E155DevSampleConfig = builds(
    generate_subset,
    source_dataset_rids=["A9-D272"],
    source_version="0.3.0",
    include_tables=["file"],
    element_table="file",
    exclude_tables=["experiment", "biosample"],
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

# ---------------------------------------------------------------------------
# E15.5 training sample — 200 random files for training experiments
# ---------------------------------------------------------------------------
E155TrainingSampleConfig = builds(
    generate_subset,
    source_dataset_rids=["A9-D272"],
    source_version="0.3.0",
    include_tables=["file"],
    element_table="file",
    exclude_tables=["experiment", "biosample"],
    filter_name="random_sample",
    filter_params={"n": 200, "seed": 123},
    output_description=(
        "200-element random sample from E15.5 subset (A9-D272) for "
        "training experiments. Seed=123 for reproducibility."
    ),
    output_types=["Training"],
    parent_dataset_rid="A9-D272",
    zen_partial=True,
)
script_store(E155TrainingSampleConfig, name="e155_training_sample")

# ---------------------------------------------------------------------------
# Annotate Phenotype — classify file genotypes as WildType/Mutated/Unknown
# ---------------------------------------------------------------------------
AnnotatePhenotypeConfig = builds(
    annotate_phenotype,
    source_dataset_rids=["A9-D272"],
    source_version="0.3.0",
    include_tables=["file", "biosample"],
    element_table="file",
    exclude_tables=["experiment"],
    feature_table="file",
    feature_name="Phenotype",
    term_column="Phenotype_Type",
    genotype_column="biosample_Genotype",
    rid_column="file_RID",
    zen_partial=True,
)
script_store(AnnotatePhenotypeConfig, name="annotate_phenotype")

# "none" placeholder — used as default when no script is needed
script_store(None, name="none")

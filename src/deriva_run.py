"""
Deriva ML Model Runner
======================

A standalone driver to execute machine learning models using hydra-zen for configuration.

This module combines the CLI entry point with the model execution logic, providing
a complete pipeline from command-line invocation to model execution within a
DerivaML environment.

Architecture Overview
---------------------
The module has two main components:

1. **Configuration Setup** (bottom of file):
   - Defines the hydra-zen configuration schema using `builds()`
   - Registers configuration groups from the `configs` package
   - Launches the hydra application when run as a script

2. **Model Execution** (run_model function):
   - Receives resolved configuration from hydra
   - Creates a DerivaML connection and execution context
   - Runs the configured model within the execution context
   - Uploads results to the Deriva catalog

Usage
-----
Run from the command line with hydra configuration overrides:

    python deriva_run.py +experiment=my_experiment
    python deriva_run.py model_config.epochs=100 dry_run=True

The hydra configuration system allows you to:
- Override individual parameters on the command line
- Compose configurations from multiple groups
- Run parameter sweeps with -m/--multirun

Customization
-------------
To adapt this template for a specific domain:
1. Replace `DerivaML` with your domain-specific class (e.g., `EyeAI`)
2. Modify the config imports to include your domain configs
3. Adjust the `run_model` signature if additional parameters are needed
"""

import logging
from typing import Any

from hydra_zen import store, zen, builds

from deriva_ml import DerivaML, DerivaMLConfig, RID
from deriva_ml.dataset import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration, Workflow


def run_model(
    deriva_ml: DerivaMLConfig,
    datasets: list[DatasetSpec],
    assets: list[RID],
    description: str,
    workflow: Workflow,
    model_config: Any,
    dry_run: bool = False,
) -> None:
    """
    Execute a machine learning model within a DerivaML execution context.

    This function serves as the main entry point called by hydra-zen after
    configuration resolution. It orchestrates the complete execution lifecycle:
    connecting to Deriva, creating an execution record, running the model,
    and uploading results.

    Parameters
    ----------
    deriva_ml : DerivaMLConfig
        Configuration for the DerivaML connection. Contains server URL,
        catalog ID, credentials, and other connection parameters.

    datasets : list[DatasetSpec]
        Specifications for datasets to use in this execution. Each DatasetSpec
        identifies a dataset in the Deriva catalog to be made available to
        the model.

    assets : list[RID]
        Resource IDs (RIDs) of assets to include in the execution. Typically
        used for model weight files, pretrained checkpoints, or other
        artifacts needed by the model.

    description : str
        Human-readable description of this execution run. Stored in the
        Deriva catalog for provenance tracking.

    workflow : Workflow
        The workflow definition to associate with this execution. Defines
        the computational pipeline and its metadata.

    model_config : Any
        A hydra-zen callable that wraps the actual model code. When called
        with `ml_instance` and `execution` arguments, it runs the model
        training or inference logic.

    dry_run : bool, optional
        If True, create the execution record but skip actual model execution.
        Useful for testing configuration without running expensive computations.
        Default is False.

    Returns
    -------
    None
        Results are uploaded to the Deriva catalog as execution outputs.

    Notes
    -----
    The function clears any logging handlers set up by hydra to avoid
    conflicts with DerivaML's logging configuration.

    The `model_config` parameter is a partially-configured callable created
    by hydra-zen's `builds()` function, not a plain dataclass. It must accept
    `ml_instance` and `execution` as keyword arguments.

    Examples
    --------
    This function is typically not called directly, but through hydra:

        # From command line:
        python deriva_run.py +experiment=cifar10_cnn dry_run=True

        # Programmatically (for testing):
        from hydra import compose, initialize
        with initialize(config_path="configs"):
            cfg = compose(config_name="deriva_model")
            run_model(**cfg)
    """
    # ---------------------------------------------------------------------------
    # Clear hydra's logging configuration
    # ---------------------------------------------------------------------------
    # Hydra sets up its own logging handlers which can interfere with DerivaML's
    # logging. Remove them to ensure consistent log output.
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # ---------------------------------------------------------------------------
    # Connect to the Deriva catalog
    # ---------------------------------------------------------------------------
    # Instantiate the DerivaML client from the configuration. For domain-specific
    # catalogs (e.g., EyeAI, GUDMAP), replace DerivaML with the appropriate class.
    ml_instance = DerivaML.instantiate(deriva_ml)

    # ---------------------------------------------------------------------------
    # Create the execution context
    # ---------------------------------------------------------------------------
    # The ExecutionConfiguration bundles together all the inputs for this run:
    # which datasets to use, which assets (model weights, etc.), and metadata.
    execution_config = ExecutionConfiguration(
        datasets=datasets,
        assets=assets,
        description=description
    )

    # Create the execution record in the catalog. This generates a unique
    # execution ID and sets up the working directories for this run.
    execution = ml_instance.create_execution(
        execution_config,
        workflow=workflow,
        dry_run=dry_run
    )

    # ---------------------------------------------------------------------------
    # Run the model within the execution context
    # ---------------------------------------------------------------------------
    # The context manager handles setup (downloading datasets, creating output
    # directories) and teardown (recording completion status, timing).
    with execution.execute() as exec_context:
        if dry_run:
            # In dry run mode, skip model execution but still test the setup
            logging.info("Dry run mode: skipping model execution")
        else:
            # Invoke the model configuration callable. The model_config is a
            # hydra-zen wrapped function that has been partially configured with
            # all model-specific parameters (e.g., learning rate, batch size).
            # We provide the runtime context here.
            model_config(ml_instance=ml_instance, execution=exec_context)

    # ---------------------------------------------------------------------------
    # Upload results to the catalog
    # ---------------------------------------------------------------------------
    # After the model completes, upload any output files (metrics, predictions,
    # model checkpoints) to the Deriva catalog for permanent storage.
    if not dry_run:
        assets = execution.upload_execution_outputs()

        # Print summary of uploaded assets
        total_files = sum(len(files) for files in assets.values())
        if total_files > 0:
            print(f"\nUploaded {total_files} asset(s) to catalog:")
            for asset_type, files in assets.items():
                for f in files:
                    print(f"  - {asset_type}: {f}")


# =============================================================================
# Hydra-Zen Configuration Setup
# =============================================================================
# This section configures the hydra-zen command-line interface. The `builds()`
# function creates a dataclass-based configuration schema that matches the
# `run_model` function signature.

# Create the main configuration schema for this application.
# The `hydra_defaults` list specifies which config groups to compose together.
deriva_model = builds(
    run_model,
    description="Simple model run",
    populate_full_signature=True,  # Include all function parameters in config
    hydra_defaults=[
        "_self_",  # Include this config's values
        {"deriva_ml": "default_deriva"},      # Connection settings
        {"datasets": "default_dataset"},       # Dataset specifications
        {"assets": "default_asset"},           # Asset RIDs
        {"workflow": "default_workflow"},      # Workflow definition
        {"model_config": "default_model"},     # Model configuration
    ],
)

# Register the main config in the hydra-zen store
store(deriva_model, name="deriva_model")

# ---------------------------------------------------------------------------
# Load configuration modules
# ---------------------------------------------------------------------------
# Dynamically import all config modules from the configs package. Each module
# registers its configurations with the hydra-zen store when imported.
#
# To add new configuration options:
# 1. Create a new module in configs/ (e.g., configs/my_model.py)
# 2. Use store() to register your configs with the appropriate group name
# 3. The module will be automatically discovered and loaded

from configs import load_all_configs  # noqa: E402
load_all_configs()


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Finalize the hydra-zen store by adding all registered configs to hydra
    store.add_to_hydra_store()

    # Launch the hydra application. This will:
    # 1. Parse command-line arguments
    # 2. Compose the configuration from defaults and overrides
    # 3. Call run_model() with the resolved configuration
    zen(run_model).hydra_main(
        config_name="deriva_model",  # Main config to use
        version_base="1.3",          # Hydra compatibility version
        config_path=None,            # Use hydra-zen store (not file-based configs)
    )

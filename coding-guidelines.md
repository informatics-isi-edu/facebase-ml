
# Recommended Workflow and Coding Guidelines

This document captures operational practices and coding standards for projects based on the DerivaML Model Template.
These are designed to help ensure that DerivaML is a robust and reliable platform for reproducible research. 
These guidelines are prescriptive, and you are encouraged to adopt them as a starting point for your own projects.


## Configuration
- Each model should live in its own repository following this template. 
- uv should be used to manage all dependencies in the project.  It *SHOULD* be possible to rebuild a complete environment from scratch using the `uv.lock` file.
- The generated `uv.lock` should be committed as part of the repository

## Git workflow
- Even if you are working by yourself, you *SHOULD* work in a Git branch and create pull requests. Rebase your branch regularly to keep it up to date with the main branch.
- You *MUST* commit any changes to the model prior to running it. This maximizes DerivaML’s ability to track the code used to produce results.
- No change is too small to properly track in GitHub and DerivaML.
- While debugging, you *MAY*n use a dry run: the `Execution.create_execution` method has a `dry_run` option that downloads inputs without creating Execution records or uploading results. Once you’re confident in your model or notebook, remove `dry_run`, create a new version tag, and perform a full run to completion.

## Coding standards
- You *SHOULD* use an established doc string format for your code. DerivaML uses the Google docstring format. See: [Google Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- You *SHOULD* use type hints wherever possible.

## Versioning and releases
- You *SHOULD* use `bump-version` to create version tags prior to running your model. DerivaML uses semantic versioning:
  - Major: breaking changes
  - Minor: new features
  - Patch: bug fixes
- Script usage:
  ```
  uv run bump-version major|minor|patch
  ```
- DerivaML uses `setuptools-scm` to determine the current version dynamically:
  ```
  uv run python -m setuptools_scm
  ```

## Notebooks
- You *MUST NOT* commit a notebook with output cells. Install and enable `nbstripout` to strip outputs automatically.
- Notebooks *SHOULD* be focused on a single task (analysis and visualization). Prefer scripts for training models.
- You *MUST* ensure that your notebook can run start-to-finish without intervention. Once satisfied, run in the Deriva-ML environment with:
  ```
  uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb --host <HOST> --catalog <ID> --kernel <repository-name>
  ```
  This will upload the executed notebook to the catalog.

## Executions and experiments
- You *MUST* always run your code from a hydra-zen configuration file, and you *SHOULD* commit your code before running.
- During debugging, use the `dry_run` option available in `Execution.create_execution`. This downloads inputs but does not create Execution records or upload results. Remove `dry_run`, tag a new version, and run to completion when ready.

## Data
- As a general rule, you *SHOULD NOT* commit data files. Store data in DerivaML instead.

## Extensibility
- DerivaML provides functions for managing generic ML workflows and is designed to be extended via inheritance for domain-specific functionality. Consider creating a domain-specific module that inherits from the DerivaML class and instantiating it in your script or notebook.

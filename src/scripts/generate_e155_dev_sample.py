"""Generate a 100-element random sample from the E15.5 subset.

For filters with requires_data=False (like random_sample), we list dataset
members directly from the catalog — no bag download needed. For filters with
requires_data=True, we download a metadata-only bag and denormalize.

Run via:
    uv run deriva-ml-run +experiment=e155_dev_sample dry_run=true
    uv run deriva-ml-run +experiment=e155_dev_sample
"""

from __future__ import annotations

import pandas as pd

from deriva_ml import DerivaML
from deriva_ml.execution import Execution
from deriva_ml.feature import FeatureRecord

from scripts.subset_filters import get_filter


def generate_e155_dev_sample(
    # Source datasets
    source_dataset_rids: list[str] | None = None,
    source_version: str | None = None,
    # Denormalization (only needed for requires_data=True filters)
    include_tables: list[str] | None = None,
    element_table: str = "file",
    exclude_tables: list[str] | None = None,
    # Feature caching (catalog-query path)
    feature_name: str | None = None,
    # Filter
    filter_name: str = "random_sample",
    filter_params: dict | None = None,
    # Output dataset
    output_description: str = "",
    output_types: list[str] | None = None,
    parent_dataset_rid: str | None = None,
    # DerivaML integration (injected by framework)
    ml_instance: DerivaML | None = None,
    execution: Execution | None = None,
) -> None:
    """Create a dataset subset by filtering a source dataset.

    The filter's requires_data flag determines the data path:
    - requires_data=False: list_dataset_members() → filter on RIDs (fast)
    - requires_data=True: download_dataset_bag() → denormalize → filter on values
    """
    if ml_instance is None:
        raise ValueError(
            "ml_instance is required — it should be injected by the DerivaML "
            "framework via deriva-ml-run."
        )

    source_dataset_rids = source_dataset_rids or []
    include_tables = include_tables or []
    filter_params = filter_params or {}
    output_types = output_types or []

    if not source_dataset_rids:
        raise ValueError(
            "source_dataset_rids must contain at least one dataset RID."
        )

    # Look up the filter and check its data requirements.
    filter_entry = get_filter(filter_name)

    if not filter_entry.requires_data:
        # ---------------------------------------------------------------
        # Fast path: list members directly from catalog (no bag download)
        # ---------------------------------------------------------------
        all_rids = []
        for rid in source_dataset_rids:
            dataset = ml_instance.lookup_dataset(rid)
            print(f"Source: {dataset.description} (RID: {rid})")
            all_members = dataset.list_dataset_members()
            members = all_members.get(element_table, [])
            member_rids = [m["RID"] for m in members]
            print(f"  Members ({element_table}): {len(member_rids)}")
            all_rids.extend(member_rids)

        rids, selection_desc = filter_entry.fn(all_rids, **filter_params)
        print(f"\n{selection_desc}")

    else:
        # ---------------------------------------------------------------
        # Data path: download bag or cache features, then filter
        # ---------------------------------------------------------------
        dataframes: dict[str, pd.DataFrame] = {}

        if feature_name:
            feature_df = ml_instance.cache_features(
                element_table,
                feature_name,
                selector=FeatureRecord.select_newest,
            )
            print(f"Cached features: {len(feature_df)} rows from {element_table}.{feature_name}")
            for rid in source_dataset_rids:
                dataset = ml_instance.lookup_dataset(rid)
                print(f"Source: {dataset.description} (RID: {rid})")
                dataframes[rid] = feature_df
        else:
            for rid in source_dataset_rids:
                dataset = ml_instance.lookup_dataset(rid)
                version = source_version or dataset.current_version
                print(f"Source: {dataset.description} (RID: {rid}, version: {version})")

                bag = dataset.download_dataset_bag(
                    version=version, materialize=False, exclude_tables=exclude_tables
                )
                df = bag.denormalize_as_dataframe(include_tables)
                print(f"  Denormalized: {len(df)} rows, {len(df.columns)} columns")
                dataframes[rid] = df

        rids, selection_desc = filter_entry.fn(
            dataframes, element_table=element_table, **filter_params
        )
        print(f"\n{selection_desc}")

    if execution is None:
        print(f"\n[DRY RUN] Would create dataset with {len(rids)} members")
        print(f"  Description: {output_description}")
        print(f"  Types: {output_types}")
        return

    # Create the new dataset
    new_dataset = execution.create_dataset(
        description=output_description,
        dataset_types=output_types,
    )

    new_dataset.add_dataset_members(
        members=rids,
        description=selection_desc,
    )

    # Nest under parent if specified
    if parent_dataset_rid:
        dd_table = ml_instance.pathBuilder().schemas["deriva-ml"].tables["Dataset_Dataset"]
        dd_table.insert([{
            "Dataset": parent_dataset_rid,
            "Nested_Dataset": new_dataset.dataset_rid,
        }])
        print(f"  Nested under parent: {parent_dataset_rid}")

    print(f"\nCreated dataset: {new_dataset.dataset_rid}")
    print(f"  Description: {output_description}")
    print(f"  Types: {output_types}")
    print(f"  Members: {len(rids)}")
    print(f"  Version: {new_dataset.current_version}")

#!/usr/bin/env python
"""Create a curated subset dataset containing only E15.5 developmental stage files.

This script filters the MusMorph file table by joining through the biosample
table to select only files whose biosample has stage = E15.5. The resulting
dataset is added as a child of the parent "Complete" dataset.

**Join path:**
    file.biosample → biosample.RID → biosample.stage → vocab.stage.id

The FaceBase ``vocab.stage`` table uses CURIE-format IDs (e.g.,
``FACEBASE:1-4GHR`` for E15.5). The ``biosample.stage`` column stores these
CURIEs, not the human-readable name.

Follows the Phase 3b (Curated Subset) workflow from the dataset-lifecycle skill
and the Base Script Template from catalog-operations-workflow.

Usage::

    # Dry run — prints what would be created without modifying the catalog
    uv run python src/scripts/create_e155_subset.py \\
        --hostname dev.facebase.org --catalog-id 19 --dry-run

    # Create the subset dataset
    uv run python src/scripts/create_e155_subset.py \\
        --hostname dev.facebase.org --catalog-id 19 \\
        --parent-dataset A9-ANT2

    # Override stage (e.g., E10.5)
    uv run python src/scripts/create_e155_subset.py \\
        --hostname dev.facebase.org --catalog-id 19 \\
        --stage-name E10.5
"""

import argparse
import sys

from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_vocab_terms(
    ml: DerivaML, vocab_table: str, terms: dict[str, str]
) -> None:
    """Ensure vocabulary terms exist, creating any that are missing.

    Catalog clones may not have all vocabulary terms from the source catalog.
    Always call this before using terms in create_dataset, create_workflow, etc.

    Args:
        ml: Connected DerivaML instance.
        vocab_table: Vocabulary table name (e.g., "Workflow_Type", "Dataset_Type").
        terms: Dict mapping term name to description for each required term.
    """
    existing = {t.name for t in ml.list_vocabulary_terms(vocab_table)}
    for name, description in terms.items():
        if name not in existing:
            print(f"  Creating {vocab_table} term: {name}")
            ml.add_term(vocab_table, name, description)


def lookup_stage_curie(ml: DerivaML, stage_name: str) -> str:
    """Look up the CURIE ID for a stage name in vocab.stage.

    The ``biosample.stage`` column stores CURIEs (e.g., ``FACEBASE:1-4GHR``),
    not human-readable names (e.g., ``E15.5``). This helper resolves the
    human-readable name to its CURIE by querying the ``vocab.stage`` table.

    Args:
        ml: Connected DerivaML instance.
        stage_name: Human-readable stage name (e.g., "E15.5").

    Returns:
        The CURIE string (e.g., "FACEBASE:1-4GHR").

    Raises:
        SystemExit: If the stage name is not found in vocab.stage.
    """
    pb = ml.pathBuilder()
    stage_table = pb.schemas["vocab"].tables["stage"]
    results = list(stage_table.filter(stage_table.name == stage_name).entities())
    if not results:
        print(
            f"ERROR: Stage '{stage_name}' not found in vocab.stage. "
            f"Check available stages with: preview_table('stage')",
            file=sys.stderr,
        )
        sys.exit(1)
    curie = results[0]["id"]
    print(f"  Stage '{stage_name}' → CURIE: {curie}")
    return curie


def query_file_rids_by_stage(
    ml: DerivaML, schema: str, stage_curie: str
) -> list[str]:
    """Query all file RIDs whose biosample has the given stage.

    Performs a two-step query:
    1. Find all biosample RIDs with the target stage
    2. Find all file RIDs linked to those biosamples

    A single joined query would be more efficient, but the two-step approach
    is simpler and more readable with the datapath API. For ~40K files this
    completes in a few seconds.

    Args:
        ml: Connected DerivaML instance.
        schema: Domain schema name (e.g., "isa").
        stage_curie: CURIE of the target stage (e.g., "FACEBASE:1-4GHR").

    Returns:
        List of file RID strings matching the stage filter.
    """
    pb = ml.pathBuilder()

    # Step 1: Get all biosample RIDs with the target stage
    biosample_table = pb.schemas[schema].tables["biosample"]
    biosamples = list(
        biosample_table.filter(biosample_table.stage == stage_curie).entities()
    )
    biosample_rids = {b["RID"] for b in biosamples}
    print(f"  Found {len(biosample_rids)} biosamples with stage {stage_curie}")

    if not biosample_rids:
        return []

    # Step 2: Get all file RIDs linked to those biosamples
    # Query all files and filter client-side by biosample membership.
    # This avoids constructing a complex server-side IN clause across
    # potentially hundreds of biosample RIDs.
    file_table = pb.schemas[schema].tables["file"]
    all_files = list(file_table.entities())
    matching_rids = [
        f["RID"] for f in all_files
        if f.get("biosample") in biosample_rids
    ]
    print(f"  Found {len(matching_rids)} files linked to those biosamples")

    return matching_rids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the E15.5 subset creation script.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Create a curated DerivaML dataset subset filtered by developmental stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hostname", required=True,
        help="Deriva server hostname (e.g., dev.facebase.org)",
    )
    parser.add_argument(
        "--catalog-id", required=True,
        help="Catalog ID (e.g., 19)",
    )
    parser.add_argument(
        "--schema", default=None,
        help="Domain schema name (auto-detected if single schema, default: isa)",
    )
    parser.add_argument(
        "--stage-name", default="E15.5",
        help="Developmental stage name to filter by (default: E15.5)",
    )
    parser.add_argument(
        "--parent-dataset", default=None,
        help="RID of parent dataset to nest this subset under (e.g., A9-ANT2)",
    )
    parser.add_argument(
        "--workflow-type", default="Data_Management",
        help="Workflow type for provenance (created if not in catalog)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be created without modifying the catalog",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Connect to catalog
    # -----------------------------------------------------------------------
    print(f"Connecting to {args.hostname} catalog {args.catalog_id}...")
    try:
        ml = DerivaML(hostname=args.hostname, catalog_id=args.catalog_id)
    except Exception as e:
        print(
            f"ERROR: Failed to connect to {args.hostname} catalog "
            f"{args.catalog_id}: {e}",
            file=sys.stderr,
        )
        return 1

    # Set default schema
    if args.schema:
        ml.default_schema = args.schema
    elif len(ml.domain_schemas) == 1:
        ml.default_schema = next(iter(ml.domain_schemas))
    else:
        # For MusMorph clone, default to "isa" if it exists
        if "isa" in ml.domain_schemas:
            ml.default_schema = "isa"
        else:
            print(
                f"ERROR: Multiple domain schemas found: {ml.domain_schemas}. "
                f"Use --schema to specify.",
                file=sys.stderr,
            )
            return 1

    print(f"  Domain schemas: {ml.domain_schemas}")
    print(f"  Default schema: {ml.default_schema}")

    # -----------------------------------------------------------------------
    # 2. Resolve stage name to CURIE and query matching files
    # -----------------------------------------------------------------------
    print(f"Looking up stage '{args.stage_name}'...")
    stage_curie = lookup_stage_curie(ml, args.stage_name)

    print(f"Querying files for stage {args.stage_name}...")
    matching_rids = query_file_rids_by_stage(ml, ml.default_schema, stage_curie)

    if not matching_rids:
        print(
            f"ERROR: No files found for stage '{args.stage_name}'. Aborting.",
            file=sys.stderr,
        )
        return 1

    # Build a description that includes the actual count.
    description = (
        f"MusMorph {args.stage_name} subset — {len(matching_rids)} files "
        f"(micro-CT scans, segmentations, and landmarks) from biosamples at "
        f"developmental stage {args.stage_name}. Filtered from the complete "
        f"MusMorph dataset."
    )

    # -----------------------------------------------------------------------
    # 3. Dry run — print what would happen and exit
    # -----------------------------------------------------------------------
    if args.dry_run:
        print(f"\n[DRY RUN] Would create dataset with {len(matching_rids)} members")
        print(f"  Stage: {args.stage_name} (CURIE: {stage_curie})")
        print(f"  Description: {description}")
        print(f"  Parent dataset: {args.parent_dataset or '(none)'}")
        print(f"  First 5 RIDs: {matching_rids[:5]}")
        return 0

    # -----------------------------------------------------------------------
    # 4. Ensure prerequisites — vocab terms may be missing in cloned catalogs
    # -----------------------------------------------------------------------
    ensure_vocab_terms(ml, "Workflow_Type", {
        args.workflow_type: "Catalog data management operations (dataset creation, ETL, curation)",
    })
    ensure_vocab_terms(ml, "Dataset_Type", {
        "Complete": "A dataset containing all available records of a given type.",
    })

    # -----------------------------------------------------------------------
    # 5. Create workflow and execution for provenance tracking
    # -----------------------------------------------------------------------
    workflow = ml.create_workflow(
        name=f"MusMorph {args.stage_name} Subset Creation",
        workflow_type=args.workflow_type,
        description=(
            f"Create a curated subset of MusMorph files filtered to "
            f"developmental stage {args.stage_name}. Joins file → biosample "
            f"→ vocab.stage to select matching records."
        ),
    )

    config = ExecutionConfiguration(
        description=(
            f"Create {args.stage_name} subset: filter "
            f"{ml.default_schema}.file by biosample stage"
        ),
    )

    # -----------------------------------------------------------------------
    # 6. Create the dataset, add members, and nest under parent
    # -----------------------------------------------------------------------
    with ml.create_execution(config, workflow=workflow) as execution:
        print(f"  Execution RID: {execution.execution_rid}")

        dataset = execution.create_dataset(
            dataset_types=["Complete"],
            description=description,
        )
        print(f"  Dataset RID: {dataset.dataset_rid}")

        # Use dict form {table: [rids]} to skip per-RID table resolution,
        # and validate=False to skip the expensive RID-by-RID lookup.
        print(f"  Adding {len(matching_rids)} members...")
        dataset.add_dataset_members(
            {"file": matching_rids}, validate=False
        )
        print(f"  Done. Version: {dataset.current_version}")

        # Nest under parent dataset if specified.
        # Uses ml.add_child_dataset() — the Dataset object method is
        # add_child_dataset, not add_nested_dataset.
        if args.parent_dataset:
            print(f"  Nesting under parent dataset {args.parent_dataset}...")
            parent = ml.lookup_dataset(args.parent_dataset)
            parent.add_child_dataset(dataset.dataset_rid)
            print(f"  Added as child of {args.parent_dataset}")

    # Upload any execution output files (logs, etc.) to the catalog.
    execution.upload_execution_outputs()

    # -----------------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------------
    print(f"\nDataset created successfully:")
    print(f"  RID:     {dataset.dataset_rid}")
    print(f"  Stage:   {args.stage_name}")
    print(f"  Members: {len(matching_rids)}")
    print(f"  Version: {dataset.current_version}")
    if args.parent_dataset:
        print(f"  Parent:  {args.parent_dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

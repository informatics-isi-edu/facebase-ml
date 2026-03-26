#!/usr/bin/env python
"""Create a DerivaML dataset containing all MusMorph micro-CT scan images.

This is a bootstrap script — it creates the first dataset in the catalog by
querying all records from a specified table and adding them as dataset members.
Once this "Complete" dataset exists, subsets and train/test splits can be
derived from it.

The default target is the ``isa.file`` table in the MusMorph clone
(dev.facebase.org catalog 19), which contains ~39,000 craniofacial micro-CT
scans (.mnc files) of Mus musculus from the "Ap2: A standardized mouse
morphology dataset for MusMorph" project. The scans span multiple
developmental stages (E10.5 through Adult) and genotypes (Tfap2a wild-type,
heterozygous, and mutant).

Follows the Phase 3a (Bootstrap) workflow from the dataset-lifecycle skill
and the Base Script Template from catalog-operations-workflow.

Usage::

    # Dry run — prints what would be created without modifying the catalog
    uv run python src/scripts/create_musmorph_dataset.py \\
        --hostname dev.facebase.org --catalog-id 19 --dry-run

    # Create the dataset
    uv run python src/scripts/create_musmorph_dataset.py \\
        --hostname dev.facebase.org --catalog-id 19

    # Override defaults for a different catalog or table
    uv run python src/scripts/create_musmorph_dataset.py \\
        --hostname localhost --catalog-id 7 --schema facebase --table Scan
"""

import argparse
import sys

from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_workflow_type(
    ml: DerivaML, type_name: str, description: str
) -> None:
    """Create a workflow type if it doesn't already exist.

    Catalog clones may not have all vocabulary terms from the source catalog,
    so we check and create before attempting to use the type.

    Args:
        ml: Connected DerivaML instance.
        type_name: Workflow type name to ensure exists.
        description: Description if the type needs to be created.
    """
    existing = {t.name for t in ml.list_vocabulary_terms("Workflow_Type")}
    if type_name not in existing:
        print(f"  Creating workflow type: {type_name}")
        ml.add_term("Workflow_Type", type_name, description)


def query_all_rids(ml: DerivaML, schema: str, table: str) -> list[str]:
    """Query all RIDs from a table.

    Uses ``pathBuilder()`` (a method, not a property) to access the ERMrest
    data path API. The ``entities()`` call returns a lazy iterator, so we
    wrap it with ``list()`` to materialize all results.

    Args:
        ml: Connected DerivaML instance.
        schema: Schema name containing the table.
        table: Table name to query.

    Returns:
        List of RID strings for every record in the table.
    """
    pb = ml.pathBuilder()
    tbl = pb.schemas[schema].tables[table]
    entities = list(tbl.entities())
    return [e["RID"] for e in entities]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the dataset creation script.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Create a DerivaML dataset from all records in a table",
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
        "--table", default="file",
        help="Table to query for dataset members (default: file)",
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

    # Set default schema: use --schema if provided, auto-detect if single
    # schema, error if multiple schemas and no --schema.
    if args.schema:
        ml.default_schema = args.schema
    elif len(ml.domain_schemas) == 1:
        ml.default_schema = next(iter(ml.domain_schemas))
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
    # 2. Query all RIDs from the target table
    # -----------------------------------------------------------------------
    print(f"Querying all {ml.default_schema}.{args.table} records...")
    all_rids = query_all_rids(ml, ml.default_schema, args.table)
    print(f"  Found {len(all_rids)} records")

    if not all_rids:
        print(
            f"ERROR: No records found in {ml.default_schema}.{args.table}. "
            f"Aborting.",
            file=sys.stderr,
        )
        return 1

    # Build a description that includes the actual count.
    description = (
        "All MusMorph micro-CT scan images (.mnc) from the "
        "'Ap2: A standardized mouse morphology dataset for MusMorph' project. "
        f"Contains {len(all_rids)} craniofacial micro-CT scans of Mus musculus "
        "across multiple developmental stages (E10.5\u2013Adult) and genotypes "
        "(Tfap2a wild-type, heterozygous, mutant)."
    )

    # -----------------------------------------------------------------------
    # 3. Dry run — print what would happen and exit
    # -----------------------------------------------------------------------
    if args.dry_run:
        print(f"\n[DRY RUN] Would create dataset with {len(all_rids)} members")
        print(f"  Description: {description}")
        print(f"  Types: ['Complete']")
        print(f"  First 5 RIDs: {all_rids[:5]}")
        return 0

    # -----------------------------------------------------------------------
    # 4. Ensure prerequisites
    # -----------------------------------------------------------------------
    ensure_workflow_type(
        ml,
        args.workflow_type,
        "Catalog data management operations (dataset creation, ETL, curation)",
    )

    # -----------------------------------------------------------------------
    # 5. Create workflow and execution for provenance tracking
    # -----------------------------------------------------------------------
    # The workflow is a reusable definition; the execution is this specific run.
    # The workflow is passed to create_execution(), not to ExecutionConfiguration.
    workflow = ml.create_workflow(
        name="MusMorph Dataset Bootstrap",
        workflow_type=args.workflow_type,
        description=(
            "Create the initial 'Complete' dataset from all records in "
            f"{ml.default_schema}.{args.table} in the MusMorph clone. "
            "This is a one-time bootstrap operation."
        ),
    )

    config = ExecutionConfiguration(
        description=(
            f"Bootstrap: create Complete dataset from all "
            f"{ml.default_schema}.{args.table} records"
        ),
    )

    # -----------------------------------------------------------------------
    # 6. Create the dataset and add members
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
        print(f"  Adding {len(all_rids)} members...")
        dataset.add_dataset_members(
            {args.table: all_rids}, validate=False
        )
        print(f"  Done. Version: {dataset.current_version}")

    # Upload any execution output files (logs, etc.) to the catalog.
    execution.upload_execution_outputs()

    # -----------------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------------
    print(f"\nDataset created successfully:")
    print(f"  RID:     {dataset.dataset_rid}")
    print(f"  Members: {len(all_rids)}")
    print(f"  Version: {dataset.current_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

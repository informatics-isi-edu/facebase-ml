#!/usr/bin/env python3
"""Load FaceBase bdbag data into DerivaML catalog with full domain schema.

This script creates a 'facebase' domain schema based on the FaceBase.org
data model and loads data from a materialized bdbag export.

Domain tables created:
- Project: Research project information
- Experiment: Experiments linking datasets to biosamples
- Biosample: Biological samples with species, genotype, stage, anatomy
- Scan (asset): Micro-CT scan files (.mnc) linked to biosamples
- Landmark (asset): Anatomical landmark files (.tag) linked to biosamples

Usage:
    # Create new catalog and load data
    python load_facebase.py --hostname localhost --create-catalog facebase_demo \
        --bag-path ~/projects/facebase-snapshots/dataset_3-JQMG

    # Load into existing catalog
    python load_facebase.py --hostname localhost --catalog-id 6 \
        --bag-path ~/projects/facebase-snapshots/dataset_3-JQMG

Requirements:
    - deriva-ml package installed
    - Materialized bdbag from FaceBase
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from deriva_ml import DerivaML, TableDefinition, ColumnDefinition, BuiltinTypes
from deriva_ml.schema import create_ml_catalog

try:
    from deriva_ml.execution import ExecutionConfiguration
except ImportError:
    from deriva_ml.execution.execution_configuration import ExecutionConfiguration

from deriva.core.ermrest_model import Schema

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.propagate = False

# Ensure stdout/stderr are unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Domain schema name
DOMAIN_SCHEMA = "facebase"


def create_domain_schema(ml: DerivaML) -> dict[str, Any]:
    """Create the FaceBase domain schema with tables based on bdbag structure.

    Creates:
    - Project table: Research project metadata
    - Experiment table: Experiments with types
    - Biosample table: Biological samples with rich metadata
    - Scan asset table: Micro-CT scan files
    - Landmark asset table: Anatomical landmark files

    Args:
        ml: Connected DerivaML instance

    Returns:
        Dictionary with creation results
    """
    results = {"tables_created": [], "tables_existing": [], "vocabularies_created": []}

    # Get existing tables
    existing_tables = [t.name for t in ml.model.schemas[ml.domain_schema].tables.values()]
    logger.info(f"Existing tables in {ml.domain_schema}: {existing_tables}")

    # =========================================================================
    # 1. Create vocabularies for controlled values
    # =========================================================================
    vocabs = [v.name for schema in [ml.ml_schema, ml.domain_schema]
              for v in ml.model.schemas[schema].tables.values()
              if ml.model.is_vocabulary(v)]

    # Species vocabulary
    if "Species" not in vocabs:
        logger.info("Creating Species vocabulary...")
        ml.create_vocabulary(
            vocab_name="Species",
            comment="Species taxonomy (NCBI Taxonomy IDs)",
        )
        results["vocabularies_created"].append("Species")
        # Add common species
        ml.add_term("Species", "Mus musculus", "House mouse (NCBITAXON:10090)")

    # Developmental_Stage vocabulary
    if "Developmental_Stage" not in vocabs:
        logger.info("Creating Developmental_Stage vocabulary...")
        ml.create_vocabulary(
            vocab_name="Developmental_Stage",
            comment="Embryonic and postnatal developmental stages",
        )
        results["vocabularies_created"].append("Developmental_Stage")
        # Add common stages from FaceBase data
        stages = [
            ("E10.5", "Embryonic day 10.5"),
            ("E11.5", "Embryonic day 11.5"),
            ("E14.5", "Embryonic day 14.5"),
            ("E15.5", "Embryonic day 15.5"),
            ("E18.5", "Embryonic day 18.5"),
            ("Adult", "Adult stage"),
        ]
        for name, desc in stages:
            try:
                ml.add_term("Developmental_Stage", name, desc)
            except Exception:
                pass  # Term may already exist

    # Anatomy vocabulary
    if "Anatomy" not in vocabs:
        logger.info("Creating Anatomy vocabulary...")
        ml.create_vocabulary(
            vocab_name="Anatomy",
            comment="Anatomical structures (UBERON ontology)",
        )
        results["vocabularies_created"].append("Anatomy")
        # Add common anatomy terms
        ml.add_term("Anatomy", "head", "Head region (UBERON:0000033)")
        ml.add_term("Anatomy", "face", "Facial region (UBERON:0001456)")

    # Genotype vocabulary
    if "Genotype" not in vocabs:
        logger.info("Creating Genotype vocabulary...")
        ml.create_vocabulary(
            vocab_name="Genotype",
            comment="Mouse genotypes and genetic backgrounds",
        )
        results["vocabularies_created"].append("Genotype")

    # File_Format vocabulary
    if "File_Format" not in vocabs:
        logger.info("Creating File_Format vocabulary...")
        ml.create_vocabulary(
            vocab_name="File_Format",
            comment="File format types",
        )
        results["vocabularies_created"].append("File_Format")
        ml.add_term("File_Format", "MINC", "Medical Image NetCDF format (.mnc)")
        ml.add_term("File_Format", "MNI_TAG", "MNI tag point format (.tag)")

    # =========================================================================
    # 2. Create Project table
    # =========================================================================
    if "Project" not in existing_tables:
        logger.info("Creating Project table...")
        project_table = TableDefinition(
            name="Project",
            column_defs=[
                ColumnDefinition(name="Name", type=BuiltinTypes.text, nullok=False,
                                 comment="Project name"),
                ColumnDefinition(name="Abstract", type=BuiltinTypes.markdown, nullok=True,
                                 comment="Project abstract/description"),
                ColumnDefinition(name="Funding", type=BuiltinTypes.text, nullok=True,
                                 comment="Funding sources"),
                ColumnDefinition(name="DOI", type=BuiltinTypes.text, nullok=True,
                                 comment="Digital Object Identifier"),
                ColumnDefinition(name="URL", type=BuiltinTypes.text, nullok=True,
                                 comment="Project URL"),
                ColumnDefinition(name="Original_RID", type=BuiltinTypes.text, nullok=True,
                                 comment="RID from source FaceBase catalog"),
            ],
            comment="Research projects from FaceBase",
        )
        ml.create_table(project_table)
        results["tables_created"].append("Project")
    else:
        results["tables_existing"].append("Project")

    # =========================================================================
    # 3. Create Experiment table
    # =========================================================================
    if "Experiment" not in existing_tables:
        logger.info("Creating Experiment table...")
        experiment_table = TableDefinition(
            name="Experiment",
            column_defs=[
                ColumnDefinition(name="Local_Identifier", type=BuiltinTypes.text, nullok=True,
                                 comment="Local experiment identifier"),
                ColumnDefinition(name="Experiment_Type", type=BuiltinTypes.text, nullok=True,
                                 comment="Type of experiment (e.g., MMO:0000570 for micro-CT)"),
                ColumnDefinition(name="Protocol", type=BuiltinTypes.markdown, nullok=True,
                                 comment="Experimental protocol description"),
                ColumnDefinition(name="Original_RID", type=BuiltinTypes.text, nullok=True,
                                 comment="RID from source FaceBase catalog"),
            ],
            comment="Experiments linking datasets to biosamples",
        )
        ml.create_table(experiment_table)
        results["tables_created"].append("Experiment")
    else:
        results["tables_existing"].append("Experiment")

    # Refresh existing tables list after creating Experiment
    existing_tables = [t.name for t in ml.model.schemas[ml.domain_schema].tables.values()]

    # =========================================================================
    # 4. Create Biosample table
    # =========================================================================
    if "Biosample" not in existing_tables:
        logger.info("Creating Biosample table...")
        biosample_table = TableDefinition(
            name="Biosample",
            column_defs=[
                ColumnDefinition(name="Local_Identifier", type=BuiltinTypes.text, nullok=True,
                                 comment="Local sample identifier (e.g., ap2_3-13-10_66_e10-5)"),
                ColumnDefinition(name="Species", type=BuiltinTypes.text, nullok=True,
                                 comment="Species (references Species vocabulary)"),
                ColumnDefinition(name="Genotype", type=BuiltinTypes.text, nullok=True,
                                 comment="Genetic background/genotype"),
                ColumnDefinition(name="Strain", type=BuiltinTypes.text, nullok=True,
                                 comment="Mouse strain (e.g., MGI ID)"),
                ColumnDefinition(name="Gene", type=BuiltinTypes.text, nullok=True,
                                 comment="Gene of interest (NCBI Gene ID)"),
                ColumnDefinition(name="Stage", type=BuiltinTypes.text, nullok=True,
                                 comment="Developmental stage (references Developmental_Stage)"),
                ColumnDefinition(name="Anatomy", type=BuiltinTypes.text, nullok=True,
                                 comment="Anatomical region (references Anatomy)"),
                ColumnDefinition(name="Sex", type=BuiltinTypes.text, nullok=True,
                                 comment="Biological sex"),
                ColumnDefinition(name="Experiment", type=BuiltinTypes.text, nullok=True,
                                 comment="Associated experiment (RID)"),
                ColumnDefinition(name="Original_RID", type=BuiltinTypes.text, nullok=True,
                                 comment="RID from source FaceBase catalog"),
            ],
            comment="Biological samples with species, genotype, stage, and anatomy",
        )
        # Note: Foreign key to Experiment can be added later via ERMrest model API
        ml.create_table(biosample_table)
        results["tables_created"].append("Biosample")
    else:
        results["tables_existing"].append("Biosample")

    # Refresh existing tables list
    existing_tables = [t.name for t in ml.model.schemas[ml.domain_schema].tables.values()]

    # =========================================================================
    # 5. Create Scan asset table (for .mnc files)
    # =========================================================================
    if "Scan" not in existing_tables:
        logger.info("Creating Scan asset table...")
        ml.create_asset(
            asset_name="Scan",
            column_defs=[
                {"name": "Biosample", "type": "text", "nullok": True,
                 "comment": "Associated biosample"},
                {"name": "File_Format", "type": "text", "nullok": True,
                 "comment": "File format (MINC, etc.)"},
                {"name": "Image_Device", "type": "text", "nullok": True,
                 "comment": "Imaging device used"},
                {"name": "Original_RID", "type": "text", "nullok": True,
                 "comment": "RID from source FaceBase catalog"},
            ],
            comment="Micro-CT scan files (.mnc MINC format)",
        )
        # Add foreign key to Biosample
        # Note: create_asset doesn't support foreign_keys directly,
        # we may need to add this separately if needed
        results["tables_created"].append("Scan")
    else:
        results["tables_existing"].append("Scan")

    # =========================================================================
    # 6. Create Landmark asset table (for .tag files)
    # =========================================================================
    if "Landmark" not in existing_tables:
        logger.info("Creating Landmark asset table...")
        ml.create_asset(
            asset_name="Landmark",
            column_defs=[
                {"name": "Biosample", "type": "text", "nullok": True,
                 "comment": "Associated biosample"},
                {"name": "Scan", "type": "text", "nullok": True,
                 "comment": "Associated scan file"},
                {"name": "Point_Count", "type": "int4", "nullok": True,
                 "comment": "Number of landmark points"},
                {"name": "Original_RID", "type": "text", "nullok": True,
                 "comment": "RID from source FaceBase catalog"},
            ],
            comment="Anatomical landmark annotation files (.tag MNI format)",
        )
        results["tables_created"].append("Landmark")
    else:
        results["tables_existing"].append("Landmark")

    # =========================================================================
    # 7. Enable asset tables as dataset element types
    # =========================================================================
    logger.info("Enabling asset tables as dataset element types...")
    element_types = {t.name for t in ml.list_dataset_element_types()}

    for table_name in ["Scan", "Landmark", "Biosample"]:
        if table_name not in element_types:
            try:
                ml.add_dataset_element_type(table_name)
                logger.info(f"  Enabled {table_name} as dataset element type")
            except Exception as e:
                logger.warning(f"  Could not enable {table_name}: {e}")

    return results


def setup_dataset_types(ml: DerivaML) -> None:
    """Ensure required dataset types exist."""
    logger.info("Setting up dataset types...")

    required_types = [
        ("Complete", "A complete dataset containing all data"),
        ("FaceBase", "Dataset imported from FaceBase.org"),
        ("MicroCT", "Micro-CT imaging dataset"),
        ("Morphometric", "Morphometric analysis dataset"),
    ]

    existing_terms = {t.name for t in ml.list_vocabulary_terms("Dataset_Type")}

    for type_name, description in required_types:
        if type_name not in existing_terms:
            try:
                ml.add_term("Dataset_Type", type_name, description)
                logger.info(f"  Added dataset type: {type_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"  Could not add {type_name}: {e}")


def setup_workflow_type(ml: DerivaML) -> None:
    """Ensure FaceBase_Data_Load workflow type exists."""
    existing_types = {t.name for t in ml.list_vocabulary_terms("Workflow_Type")}

    if "FaceBase_Data_Load" not in existing_types:
        logger.info("Creating FaceBase_Data_Load workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="FaceBase_Data_Load",
            description="Workflow for loading FaceBase dataset into catalog",
        )


def load_bag_json(bag_path: Path, filename: str) -> list[dict]:
    """Load JSON data from bdbag data directory."""
    json_path = bag_path / "data" / filename
    if not json_path.exists():
        logger.warning(f"JSON file not found: {json_path}")
        return []

    with open(json_path) as f:
        data = json.load(f)

    return data if isinstance(data, list) else [data]


def insert_records(ml: DerivaML, table_name: str, records: list[dict]) -> list[dict]:
    """Insert records into a table using pathBuilder.

    Args:
        ml: DerivaML instance
        table_name: Name of the table in the domain schema
        records: List of record dictionaries to insert

    Returns:
        List of inserted records with RIDs
    """
    table = ml.pathBuilder.schemas[ml.domain_schema].tables[table_name]
    return table.insert(records)


def update_record(ml: DerivaML, table_name: str, rid: str, updates: dict) -> None:
    """Update a record in a table using pathBuilder.

    Args:
        ml: DerivaML instance
        table_name: Name of the table in the domain schema
        rid: RID of the record to update
        updates: Dictionary of column values to update
    """
    table = ml.pathBuilder.schemas[ml.domain_schema].tables[table_name]
    table.filter(table.RID == rid).update([updates])


def extract_stage_from_identifier(identifier: str) -> str | None:
    """Extract developmental stage from local identifier.

    Examples:
        'ap2_3-13-10_66_e10-5' -> 'E10.5'
        'ap2mut_1-15-10_10_e10-5' -> 'E10.5'
        'rg_nnwt_105_41' -> None
    """
    if not identifier:
        return None

    identifier = identifier.lower()
    if "_e10-5" in identifier or "_e10.5" in identifier:
        return "E10.5"
    elif "_e11-5" in identifier or "_e11.5" in identifier:
        return "E11.5"
    elif "_e14-5" in identifier or "_e14.5" in identifier:
        return "E14.5"
    elif "_e15-5" in identifier or "_e15.5" in identifier:
        return "E15.5"
    elif "_e18-5" in identifier or "_e18.5" in identifier:
        return "E18.5"
    return None


def extract_genotype_from_identifier(identifier: str) -> str | None:
    """Extract genotype hint from local identifier.

    Examples:
        'ap2mut_1-15-10_10_e10-5' -> 'Tfap2a mutant'
        'ap2wt_1-25-10_11_e10-5' -> 'Tfap2a wild-type'
        'AP2WT_3-13-10_70_e10-5' -> 'Tfap2a wild-type'
    """
    if not identifier:
        return None

    identifier = identifier.lower()
    if "ap2mut" in identifier:
        return "Tfap2a mutant"
    elif "ap2wt" in identifier:
        return "Tfap2a wild-type"
    elif identifier.startswith("ap2_"):
        return "Tfap2a heterozygous"
    return None


def load_data_from_bag(
    ml: DerivaML,
    bag_path: Path,
    batch_size: int = 100,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Load data from FaceBase bdbag into the catalog.

    Args:
        ml: Connected DerivaML instance
        bag_path: Path to materialized bdbag
        batch_size: Number of records per batch
        max_files: Maximum number of files to load (for testing)

    Returns:
        Dictionary with load statistics
    """
    results = {
        "projects": 0,
        "experiments": 0,
        "biosamples": 0,
        "scans": 0,
        "landmarks": 0,
    }

    # Setup workflow and execution
    setup_workflow_type(ml)

    workflow = ml.create_workflow(
        name="FaceBase Data Load",
        workflow_type="FaceBase_Data_Load",
        description="Load FaceBase bdbag data into DerivaML catalog",
    )

    config = ExecutionConfiguration(workflow=workflow)

    with ml.create_execution(config) as exe:
        logger.info(f"Execution RID: {exe.execution_rid}")

        # =====================================================================
        # 1. Load Project data
        # =====================================================================
        logger.info("Loading project data...")
        projects = load_bag_json(bag_path, "project.json")

        project_records = []
        project_rid_map = {}  # Map original RID to new RID

        for proj in projects:
            record = {
                "Name": proj.get("name", "Unknown Project"),
                "Abstract": proj.get("abstract"),
                "Funding": proj.get("funding"),
                "DOI": proj.get("DOI"),
                "URL": proj.get("url"),
                "Original_RID": proj.get("RID"),
            }
            project_records.append(record)

        if project_records:
            inserted = insert_records(ml,"Project", project_records)
            results["projects"] = len(inserted)
            # Map original RIDs to new RIDs
            for i, rec in enumerate(inserted):
                orig_rid = project_records[i].get("Original_RID")
                if orig_rid:
                    project_rid_map[orig_rid] = rec["RID"]
            logger.info(f"  Loaded {len(inserted)} projects")

        # =====================================================================
        # 2. Load Experiment data
        # =====================================================================
        logger.info("Loading experiment data...")
        experiments = load_bag_json(bag_path, "experiment.json")

        experiment_records = []
        experiment_rid_map = {}

        for exp in experiments:
            record = {
                "Local_Identifier": exp.get("local_identifier"),
                "Experiment_Type": exp.get("experiment_type"),
                "Protocol": exp.get("protocol"),
                "Original_RID": exp.get("RID"),
            }
            experiment_records.append(record)

        if experiment_records:
            inserted = insert_records(ml,"Experiment", experiment_records)
            results["experiments"] = len(inserted)
            for i, rec in enumerate(inserted):
                orig_rid = experiment_records[i].get("Original_RID")
                if orig_rid:
                    experiment_rid_map[orig_rid] = rec["RID"]
            logger.info(f"  Loaded {len(inserted)} experiments")

        # =====================================================================
        # 3. Load Biosample data
        # =====================================================================
        logger.info("Loading biosample data...")
        biosamples = load_bag_json(bag_path, "biosample.json")

        biosample_records = []
        biosample_rid_map = {}

        for bs in biosamples:
            # Map experiment RID to new catalog RID
            exp_rid = bs.get("experiment")
            new_exp_rid = experiment_rid_map.get(exp_rid) if exp_rid else None

            # Extract stage and genotype from identifier if not in data
            local_id = bs.get("local_identifier")
            stage = extract_stage_from_identifier(local_id)
            genotype = extract_genotype_from_identifier(local_id)

            record = {
                "Local_Identifier": local_id,
                "Species": "Mus musculus" if bs.get("species") else None,
                "Genotype": bs.get("genotype") or genotype,
                "Strain": bs.get("strain"),
                "Gene": bs.get("gene"),
                "Stage": stage,
                "Anatomy": "head" if bs.get("anatomy") else None,
                "Sex": bs.get("sex"),
                "Experiment": new_exp_rid,
                "Original_RID": bs.get("RID"),
            }
            biosample_records.append(record)

        if biosample_records:
            # Insert in batches
            for i in range(0, len(biosample_records), batch_size):
                batch = biosample_records[i:i + batch_size]
                inserted = insert_records(ml,"Biosample", batch)
                for j, rec in enumerate(inserted):
                    orig_rid = batch[j].get("Original_RID")
                    if orig_rid:
                        biosample_rid_map[orig_rid] = rec["RID"]
                results["biosamples"] += len(inserted)

                if (i + batch_size) % 500 == 0 or i + batch_size >= len(biosample_records):
                    logger.info(f"  Loaded {results['biosamples']}/{len(biosample_records)} biosamples")

        # =====================================================================
        # 4. File upload skipped - files already loaded in Brain_Image/Landmark
        # =====================================================================
        # Note: Catalog 6 already has Brain_Image and Landmark tables with files
        # from the previous load. File upload via execution is skipped here.
        # To upload new files, use the --upload-files flag (not implemented yet).
        logger.info("Skipping file upload (files already in Brain_Image/Landmark tables)")

        # =====================================================================
        # 5. Create Dataset
        # =====================================================================
        logger.info("Creating FaceBase dataset...")

        # Get dataset info from bag
        datasets = load_bag_json(bag_path, "dataset.json")
        dataset_info = datasets[0] if datasets else {}

        dataset = ml.create_dataset(
            description=dataset_info.get("description", "FaceBase imported dataset"),
            dataset_types=["Complete", "FaceBase", "MicroCT"],
        )

        dataset_rid = getattr(dataset, "dataset_rid", dataset)
        logger.info(f"  Created dataset: {dataset_rid}")

        # Add all biosamples to dataset
        all_rids = list(biosample_rid_map.values())

        # Also add existing image and landmark assets
        # Check which asset tables exist (Brain_Image for catalog 6, Scan for catalog 7)
        existing_tables = [t.name for t in ml.model.schemas[ml.domain_schema].tables.values()]

        if "Brain_Image" in existing_tables:
            brain_images = ml.list_assets("Brain_Image")
            all_rids.extend([img["RID"] for img in brain_images])
            results["scans"] = len(brain_images)
        elif "Scan" in existing_tables:
            scans = ml.list_assets("Scan")
            all_rids.extend([s["RID"] for s in scans])
            results["scans"] = len(scans)

        if "Landmark" in existing_tables:
            landmarks = ml.list_assets("Landmark")
            all_rids.extend([lm["RID"] for lm in landmarks])
            results["landmarks"] = len(landmarks)

        if all_rids:
            logger.info(f"  Adding {len(all_rids)} members to dataset...")
            ml.add_dataset_members(dataset_rid, all_rids, validate=False)

        results["dataset_rid"] = dataset_rid

    return results


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point."""
    if args is None:
        args = parse_args()

    # Create or connect to catalog
    if args.create_catalog:
        logger.info(f"Creating new catalog on {args.hostname}: {args.create_catalog}")

        catalog = create_ml_catalog(args.hostname, args.create_catalog)
        model = catalog.getCatalogModel()

        # Create domain schema
        model.create_schema(Schema.define(args.create_catalog))

        catalog_id = catalog.catalog_id
        domain_schema = args.create_catalog

        print(f"\n{'='*60}")
        print(f"  CREATED NEW CATALOG")
        print(f"  Hostname:    {args.hostname}")
        print(f"  Catalog ID:  {catalog_id}")
        print(f"  Schema:      {domain_schema}")
        print(f"{'='*60}\n")

        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(catalog_id),
            domain_schema=domain_schema,
        )
    else:
        logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(args.catalog_id),
            domain_schema=args.domain_schema,
        )
        catalog_id = args.catalog_id
        domain_schema = ml.domain_schema
        logger.info(f"Connected, domain schema: {domain_schema}")

    # Create domain schema tables
    logger.info("Creating domain schema...")
    schema_results = create_domain_schema(ml)
    logger.info(f"  Tables created: {schema_results['tables_created']}")
    logger.info(f"  Tables existing: {schema_results['tables_existing']}")
    logger.info(f"  Vocabularies created: {schema_results['vocabularies_created']}")

    # Setup dataset types
    setup_dataset_types(ml)

    # Apply catalog annotations
    logger.info("Applying catalog annotations...")
    if hasattr(ml, "apply_catalog_annotations"):
        project_name = args.create_catalog if args.create_catalog else domain_schema
        ml.apply_catalog_annotations(
            navbar_brand_text=f"FaceBase ({project_name})",
            head_title="FaceBase ML Catalog",
        )

    # Load data if bag path provided
    load_results = None
    if args.bag_path:
        bag_path = Path(args.bag_path)
        if not bag_path.exists():
            logger.error(f"Bag path does not exist: {bag_path}")
            return 1

        logger.info(f"Loading data from: {bag_path}")
        load_results = load_data_from_bag(
            ml,
            bag_path,
            batch_size=args.batch_size,
            max_files=args.max_files,
        )
        logger.info(f"Load complete: {load_results}")
    else:
        logger.info("No bag path provided, schema created but no data loaded")

    # Print summary
    print("\n" + "=" * 60)
    print("  FACEBASE LOADING COMPLETE")
    print("=" * 60)
    print(f"  Hostname:      {args.hostname}")
    print(f"  Catalog ID:    {catalog_id}")
    print(f"  Schema:        {domain_schema}")
    print("")
    print("  Schema setup:")
    print(f"    Tables created:      {schema_results['tables_created']}")
    print(f"    Vocabularies:        {schema_results['vocabularies_created']}")

    if load_results:
        print("")
        print("  Data loaded:")
        print(f"    Projects:    {load_results['projects']}")
        print(f"    Experiments: {load_results['experiments']}")
        print(f"    Biosamples:  {load_results['biosamples']}")
        print(f"    Scans:       {load_results['scans']}")
        print(f"    Landmarks:   {load_results['landmarks']}")
        print(f"    Dataset RID: {load_results.get('dataset_rid', 'N/A')}")

    print("=" * 60 + "\n")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load FaceBase bdbag data into DerivaML catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a new catalog and load data
    python load_facebase.py --hostname localhost --create-catalog facebase_demo \\
        --bag-path ~/projects/facebase-snapshots/dataset_3-JQMG

    # Create schema only (no data)
    python load_facebase.py --hostname localhost --create-catalog facebase_test

    # Load into existing catalog
    python load_facebase.py --hostname localhost --catalog-id 6 \\
        --bag-path ~/projects/facebase-snapshots/dataset_3-JQMG

    # Test with limited files
    python load_facebase.py --hostname localhost --create-catalog test \\
        --bag-path ~/data/dataset_3-JQMG --max-files 10
        """,
    )
    parser.add_argument(
        "--hostname",
        required=True,
        help="Deriva server hostname (e.g., localhost, ml.derivacloud.org)",
    )

    catalog_group = parser.add_mutually_exclusive_group(required=True)
    catalog_group.add_argument(
        "--catalog-id",
        help="Catalog ID to connect to (for existing catalogs)",
    )
    catalog_group.add_argument(
        "--create-catalog",
        metavar="PROJECT_NAME",
        help="Create a new catalog with this project name",
    )

    parser.add_argument(
        "--domain-schema",
        help="Domain schema name (auto-detected if not provided)",
    )
    parser.add_argument(
        "--bag-path",
        help="Path to materialized bdbag directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records per batch (default: 100)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to load (for testing)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())

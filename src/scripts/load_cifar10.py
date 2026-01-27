#!/usr/bin/env python3
"""Load CIFAR-10 dataset into DerivaML catalog using direct DerivaML API.

This script downloads the CIFAR-10 dataset from Kaggle and loads it into
a Deriva catalog using the DerivaML Python library directly (no MCP). It creates:
- An Image asset table for storing image files
- An Image_Class vocabulary with the 10 CIFAR-10 classes
- An Image_Classification feature linking images to their class labels
- A dataset hierarchy: Complete (all images), Segmented (Training + Testing)

Usage:
    python load_cifar10.py --hostname ml.derivacloud.org --catalog-id 99
    python load_cifar10.py --hostname localhost --create-catalog cifar10_demo

Requirements:
    - Kaggle CLI configured (~/.kaggle/kaggle.json)
    - deriva-ml package installed
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import inspect
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from deriva_ml import DerivaML
try:
    from deriva_ml.execution import ExecutionConfiguration
except ImportError:
    from deriva_ml.execution.execution_configuration import ExecutionConfiguration

from deriva_ml.schema import create_ml_catalog
try:
    from deriva_ml.core.ermrest import UploadProgress
except ImportError:
    UploadProgress = None  # not available in this deriva-ml version

from deriva.core.ermrest_model import Schema

# Configure logging with explicit handler to avoid DerivaML overriding root level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add handler directly to this logger so it works regardless of root logger config
_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.propagate = False  # Don't propagate to root logger

# Also configure the deriva_ml logger to show status messages during execution
_deriva_ml_logger = logging.getLogger("deriva_ml")
_deriva_ml_logger.setLevel(logging.INFO)
_deriva_ml_logger.addHandler(_handler)
_deriva_ml_logger.propagate = False

# Ensure stdout/stderr are unbuffered for real-time output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# CIFAR-10 class definitions
CIFAR10_CLASSES = [
    ("airplane", "Fixed-wing aircraft", ["plane", "aeroplane"]),
    ("automobile", "Motor vehicle with four wheels", ["car", "auto"]),
    ("bird", "Feathered flying vertebrate", []),
    ("cat", "Small domestic feline", ["kitten"]),
    ("deer", "Hoofed ruminant mammal", []),
    ("dog", "Domestic canine", ["puppy"]),
    ("frog", "Tailless amphibian", ["toad"]),
    ("horse", "Large domesticated hoofed mammal", ["pony"]),
    ("ship", "Large watercraft", ["boat", "vessel"]),
    ("truck", "Motor vehicle for transporting cargo", ["lorry"]),
]


def verify_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle credentials not found. Please configure ~/.kaggle/kaggle.json\n"
            "See: https://www.kaggle.com/docs/api#authentication"
        )
        return False
    return True


def download_cifar10(temp_dir: Path) -> Path:
    """Download CIFAR-10 dataset from Kaggle.

    Returns:
        Path to the extracted dataset directory
    """
    download_dir = temp_dir / "cifar10"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CIFAR-10 from Kaggle...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "cifar-10", "-p", str(download_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Kaggle download failed: {result.stderr}")
        raise RuntimeError(f"Failed to download CIFAR-10: {result.stderr}")

    # Extract the outer zip file
    zip_files = list(download_dir.glob("*.zip"))
    if zip_files:
        logger.info("Extracting outer zip archive...")
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(download_dir)

    # CIFAR-10 from Kaggle uses 7z archives for train/test data
    seven_z_files = list(download_dir.glob("*.7z"))
    if seven_z_files:
        logger.info("Extracting 7z archives (train.7z, test.7z)...")
        for seven_z_file in seven_z_files:
            result = subprocess.run(
                ["7z", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                result = subprocess.run(
                    ["7za", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to extract {seven_z_file.name}. "
                        "Please install 7-zip: brew install p7zip"
                    )

    return download_dir


def load_train_labels(data_dir: Path) -> dict[str, str]:
    """Load training labels from trainLabels.csv."""
    labels = {}
    labels_file = data_dir / "trainLabels.csv"

    if labels_file.exists():
        with open(labels_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["id"]] = row["label"]
    else:
        logger.warning("trainLabels.csv not found")

    return labels


def iter_images(data_dir: Path, split: str, labels: dict[str, str]):
    """Iterate over images with their class labels."""
    if split == "train":
        train_dir = data_dir / "train"
        if train_dir.exists():
            for img_path in sorted(train_dir.glob("*.png")):
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name:
                    yield img_path, class_name, image_id
    else:
        test_dir = data_dir / "test"
        if test_dir.exists():
            for img_path in sorted(test_dir.glob("*.png")):
                image_id = img_path.stem
                yield img_path, None, image_id


def setup_domain_model(ml: DerivaML) -> dict[str, Any]:
    """Create the domain model for CIFAR-10."""
    results = {}

    # Check existing vocabularies
    vocabs = [v.name for schema in [ml.ml_schema, ml.domain_schema]
              for v in ml.model.schemas[schema].tables.values()
              if ml.model.is_vocabulary(v)]

    # Create Image_Class vocabulary if needed
    if "Image_Class" not in vocabs:
        logger.info("Creating Image_Class vocabulary...")
        ml.create_vocabulary(
            vocab_name="Image_Class",
            comment="CIFAR-10 image classification categories",
        )
        results["vocabulary"] = {"status": "created", "name": "Image_Class"}
    else:
        logger.info("Image_Class vocabulary already exists")
        results["vocabulary"] = {"status": "exists", "name": "Image_Class"}

    # Add class terms
    existing_terms = {t.name for t in ml.list_vocabulary_terms("Image_Class")}

    logger.info("Adding CIFAR-10 class terms...")
    for class_name, description, synonyms in CIFAR10_CLASSES:
        if class_name not in existing_terms:
            ml.add_term(
                table="Image_Class",
                term_name=class_name,
                description=description,
                synonyms=synonyms,
            )
            logger.info(f"  Added term: {class_name}")
        else:
            logger.info(f"  Term exists: {class_name}")

    # Check existing tables
    tables = [t.name for t in ml.model.schemas[ml.domain_schema].tables.values()]

    # Create Image asset table if needed
    if "Image" not in tables:
        logger.info("Creating Image asset table...")
        ml.create_asset(
            asset_name="Image",
            column_defs=[],
            comment="CIFAR-10 32x32 RGB images",
        )
        results["asset_table"] = {"status": "created", "table_name": "Image"}
    else:
        logger.info("Image asset table already exists")
        results["asset_table"] = {"status": "exists", "table_name": "Image"}

    # Enable Image as dataset element type
    logger.info("Enabling Image as dataset element type...")
    element_types = {t.name for t in ml.list_dataset_element_types()}
    if "Image" not in element_types:
        ml.add_dataset_element_type("Image")

    # Create Image_Classification feature
    logger.info("Creating Image_Classification feature...")
    try:
        ml.create_feature(
            target_table="Image",
            feature_name="Image_Classification",
            comment="CIFAR-10 class label for each image",
            terms=["Image_Class"],
        )
        results["feature"] = {"status": "created", "feature_name": "Image_Classification"}
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("Image_Classification feature already exists")
            results["feature"] = {"status": "exists", "feature_name": "Image_Classification"}
        else:
            raise

    return results


def setup_workflow_type(ml: DerivaML) -> None:
    """Ensure CIFAR_Data_Load workflow type exists."""
    # Check if CIFAR_Data_Load exists in Workflow_Type vocabulary
    existing_types = {t.name for t in ml.list_vocabulary_terms("Workflow_Type")}

    if "CIFAR_Data_Load" not in existing_types:
        logger.info("Creating CIFAR_Data_Load workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="CIFAR_Data_Load",
            description="Workflow for loading CIFAR-10 dataset into catalog",
        )


def setup_dataset_types(ml: DerivaML) -> None:
    """Ensure required dataset types exist in Dataset_Type vocabulary."""
    logger.info("Setting up dataset types...")

    required_types = [
        ("Complete", "A complete dataset containing all data", ["complete", "entire"]),
        ("Training", "A dataset subset used for model training", ["training", "train", "Train"]),
        ("Testing", "A dataset subset used for model testing/evaluation", ["test", "Test"]),
        ("Split", "A dataset that contains nested dataset splits", ["split"]),
    ]

    existing_terms = {t.name for t in ml.list_vocabulary_terms("Dataset_Type")}

    for type_name, description, synonyms in required_types:
        if type_name not in existing_terms:
            try:
                ml.add_term(
                    table="Dataset_Type",
                    term_name=type_name,
                    description=description,
                    synonyms=synonyms,
                )
                logger.info(f"  Added dataset type: {type_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not add dataset type {type_name}: {e}")


def create_dataset_hierarchy(ml: DerivaML, exe: Any = None) -> dict[str, str]:
    """Create the dataset hierarchy within an execution context.

    Args:
        ml: DerivaML instance
        exe: Optional execution context. If provided, datasets are created
             within this execution for proper provenance tracking.

    Returns:
        Dictionary mapping dataset names to their RIDs
    """
    datasets = {}

    logger.info("Creating dataset hierarchy...")

    # Create Complete dataset
    if exe:
        complete_ds = exe.create_dataset(
            description="Complete CIFAR-10 dataset with all labeled images",
            dataset_types=["Complete"],
        )
    else:
        complete_ds = ml.create_dataset(
            description="Complete CIFAR-10 dataset with all labeled images",
            dataset_types=["Complete"],
        )
    datasets["complete"] = getattr(complete_ds, "dataset_rid", complete_ds)
    logger.info(f"  Created Complete dataset: {getattr(complete_ds, "dataset_rid", complete_ds)}")

    # Create Split dataset
    if exe:
        split_ds = exe.create_dataset(
            description="CIFAR-10 dataset split into training and testing subsets",
            dataset_types=["Split"],
        )
    else:
        split_ds = ml.create_dataset(
            description="CIFAR-10 dataset split into training and testing subsets",
            dataset_types=["Split"],
        )
    datasets["split"] = getattr(split_ds, "dataset_rid", split_ds)
    logger.info(f"  Created Split dataset: {getattr(split_ds, "dataset_rid", split_ds)}")

    # Create Training dataset
    if exe:
        training_ds = exe.create_dataset(
            description="CIFAR-10 training set with 50,000 labeled images",
            dataset_types=["Training"],
        )
    else:
        training_ds = ml.create_dataset(
            description="CIFAR-10 training set with 50,000 labeled images",
            dataset_types=["Training"],
        )
    datasets["training"] = getattr(training_ds, "dataset_rid", training_ds)
    logger.info(f"  Created Training dataset: {getattr(training_ds, "dataset_rid", training_ds)}")

    # Create Testing dataset
    if exe:
        testing_ds = exe.create_dataset(
            description="CIFAR-10 testing set",
            dataset_types=["Testing"],
        )
    else:
        testing_ds = ml.create_dataset(
            description="CIFAR-10 testing set",
            dataset_types=["Testing"],
        )
    datasets["testing"] = getattr(testing_ds, "dataset_rid", testing_ds)
    logger.info(f"  Created Testing dataset: {getattr(testing_ds, "dataset_rid", testing_ds)}")

    # Add Training and Testing as children of Split
    # Add Training/Testing as members of Split dataset (compatible with current deriva-ml)
    split_rid = getattr(split_ds, 'dataset_rid', split_ds)
    train_rid = getattr(training_ds, 'dataset_rid', training_ds)
    test_rid  = getattr(testing_ds, 'dataset_rid', testing_ds)
    ml.add_dataset_members(split_rid, [train_rid, test_rid], validate=False)
    logger.info("  Linked Training and Testing to Split dataset")

    return datasets


def create_upload_progress_callback(total_files: int) -> tuple[callable, dict]:
    """Create a progress callback that scales reporting frequency to the number of files.

    For small uploads (< 20 files): report every file
    For medium uploads (20-100 files): report every 10%
    For large uploads (> 100 files): report every 5% or every 100 files, whichever is more frequent

    Args:
        total_files: Total number of files to be uploaded

    Returns:
        Tuple of (callback function, state dict for tracking)
    """
    import re

    state = {
        "last_reported_percent": -1,
        "started": False,
        "callback_count": 0,
    }

    # Determine reporting interval as percentage
    if total_files < 20:
        # Report every file for small uploads (~5% intervals or every file)
        report_every_percent = max(1, 100 // total_files) if total_files > 0 else 10
    elif total_files <= 100:
        # Report every ~10% for medium uploads
        report_every_percent = 10
    else:
        # Report every 5% for large uploads
        report_every_percent = 5

    def progress_callback(progress: UploadProgress) -> None:
        """Handle upload progress updates."""
        state["callback_count"] += 1

        # Report start once
        if not state["started"]:
            state["started"] = True
            logger.info(f"  [Upload] Starting upload (reporting every ~{report_every_percent}%)...")

        # Extract current file number from message if available
        match = re.search(r"Uploading file (\d+) of (\d+)", progress.message)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            # Use the actual percent_complete from the progress object
            percent = progress.percent_complete

            # Round to nearest reporting interval for cleaner output
            report_percent = int(percent // report_every_percent) * report_every_percent

            # Report at interval boundaries
            if report_percent > state["last_reported_percent"]:
                state["last_reported_percent"] = report_percent
                logger.info(f"  [Upload] {percent:.0f}% ({current}/{total} files)")

    return progress_callback, state


def load_images(
    ml: DerivaML,
    data_dir: Path,
    batch_size: int = 500,
    max_images: int | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load images into the catalog using execution system.

    Creates datasets and loads images within a single execution for proper
    provenance tracking.

    In test mode (max_images specified), splits the limit evenly between
    training and testing images. All uploaded images go to Complete dataset,
    training images go to Training dataset, test images go to Testing dataset.

    Returns:
        Tuple of (datasets dict, load_result dict)
    """
    # Ensure CIFAR_Data_Load workflow type exists
    setup_workflow_type(ml)

    # Create workflow
    logger.info("Creating execution for data loading...")
    workflow = ml.create_workflow(
        name="CIFAR-10 Data Load",
        workflow_type="CIFAR_Data_Load",
        description="Load CIFAR-10 dataset images into DerivaML catalog",
    )

    # Create execution configuration
    config = ExecutionConfiguration(workflow=workflow)

    # Track images for dataset assignment by their filenames
    # We'll use a prefix to distinguish training from testing images
    train_filenames: list[str] = []
    test_filenames: list[str] = []
    # Track filename -> class mapping for feature assignment
    filename_to_class: dict[str, str] = {}

    # Calculate limits for training and testing
    if max_images:
        # Split limit evenly between training and testing
        train_limit = max_images // 2
        test_limit = max_images - train_limit  # Handle odd numbers
        logger.info(f"Loading {train_limit} training + {test_limit} testing images")
    else:
        train_limit = None
        test_limit = None

    # Use execution context manager
    with ml.create_execution(config) as exe:
        logger.info(f"  Execution RID: {exe.execution_rid}")

        # Create dataset hierarchy within the execution for proper provenance
        datasets = create_dataset_hierarchy(ml, exe)

        # Clear working directory to avoid uploading stale files from previous runs
        working_dir = exe.working_dir
        if working_dir.exists():
            import shutil
            for item in working_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            logger.info(f"  Cleared working directory: {working_dir}")

        # Load training labels
        labels = load_train_labels(data_dir)
        logger.info(f"Loaded {len(labels)} training labels")

        # Process training images
        logger.info("Registering training images for upload...")
        train_count = 0
        for img_path, class_name, image_id in iter_images(data_dir, "train", labels):
            if class_name is None:
                continue

            if train_limit and train_count >= train_limit:
                break

            # Create unique filename with train_ prefix and class
            new_filename = f"train_{class_name}_{image_id}.png"

            # Register file for upload
            exe.asset_file_path(
                asset_name="Image",
                file_name=str(img_path),
                asset_types=["Image"],
                copy_file=True,
                rename_file=new_filename,
            )

            train_filenames.append(new_filename)
            filename_to_class[new_filename] = class_name
            train_count += 1

            if train_count % 1000 == 0:
                logger.info(f"  Registered {train_count} training images...")

        logger.info(f"  Total training images registered: {train_count}")

        # Process test images (they don't have labels in CIFAR-10 Kaggle format)
        logger.info("Registering test images for upload...")
        test_count = 0
        for img_path, _, image_id in iter_images(data_dir, "test", labels):
            if test_limit and test_count >= test_limit:
                break

            # Create unique filename with test_ prefix
            new_filename = f"test_{image_id}.png"

            # Register file for upload
            exe.asset_file_path(
                asset_name="Image",
                file_name=str(img_path),
                asset_types=["Image"],
                copy_file=True,
                rename_file=new_filename,
            )

            test_filenames.append(new_filename)
            test_count += 1

            if test_count % 1000 == 0:
                logger.info(f"  Registered {test_count} test images...")

        logger.info(f"  Total test images registered: {test_count}")

    total_count = train_count + test_count

    # Upload outputs (after context manager exits, as per new API)
    logger.info(f"Uploading {total_count} images to catalog (this may take a while)...")

    # Create progress callback scaled to the number of files
    progress_callback, callback_state = create_upload_progress_callback(total_count)
    upload_result = exe.upload_execution_outputs(clean_folder=True, progress_callback=progress_callback)

    # Log final progress and callback statistics
    logger.info("  [Upload] 100% complete")
    logger.debug(f"  [Upload] Callback invoked {callback_state['callback_count']} times")

    uploaded_count = sum(len(files) for files in upload_result.values())
    logger.info(f"  Upload complete: {uploaded_count} files uploaded")
    for asset_type, files in upload_result.items():
        logger.info(f"    {asset_type}: {len(files)} files")

    # Get uploaded image RIDs and assign to datasets
    logger.info("Assigning images to datasets...")
    assets = ml.list_assets("Image")
    logger.info(f"  Found {len(assets)} uploaded images")

    # Build filename -> RID mapping
    filename_to_rid = {a["Filename"]: a["RID"] for a in assets}

    # Separate RIDs by dataset membership
    train_rids = [filename_to_rid[f] for f in train_filenames if f in filename_to_rid]
    test_rids = [filename_to_rid[f] for f in test_filenames if f in filename_to_rid]
    all_rids = train_rids + test_rids

    if all_rids:
        # Add all images to Complete dataset
        complete_ds = ml.lookup_dataset(datasets["complete"])
        logger.info("  Adding images to Complete dataset...")
        added = 0
        for i in range(0, len(all_rids), batch_size):
            batch = all_rids[i : i + batch_size]
            complete_ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
            logger.info(f"    Added {added}/{len(all_rids)} images")

    if train_rids:
        # Add training images to Training dataset
        training_ds = ml.lookup_dataset(datasets["training"])
        logger.info("  Adding images to Training dataset...")
        added = 0
        for i in range(0, len(train_rids), batch_size):
            batch = train_rids[i : i + batch_size]
            training_ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
            logger.info(f"    Added {added}/{len(train_rids)} images")

    if test_rids:
        # Add test images to Testing dataset
        testing_ds = ml.lookup_dataset(datasets["testing"])
        logger.info("  Adding images to Testing dataset...")
        added = 0
        for i in range(0, len(test_rids), batch_size):
            batch = test_rids[i : i + batch_size]
            testing_ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
            logger.info(f"    Added {added}/{len(test_rids)} images")

    # Add Image_Classification features for training images
    # (Training images have class labels, test images don't in CIFAR-10 Kaggle format)
    if train_rids and filename_to_class:
        logger.info("Adding Image_Classification features...")

        # Get the feature record class for Image_Classification
        ImageClassification = ml.feature_record_class("Image", "Image_Classification")

        # Create a workflow for labeling (use CIFAR_Data_Load type we already created)
        label_workflow = ml.create_workflow(
            name="CIFAR-10 Labeling",
            workflow_type="CIFAR_Data_Load",
            description="Add class labels to CIFAR-10 training images",
        )
        label_config = ExecutionConfiguration(workflow=label_workflow)

        with ml.create_execution(label_config) as label_exe:
            logger.info(f"  Labeling execution RID: {label_exe.execution_rid}")

            # Create feature records in batches
            feature_records = []
            for filename, rid in filename_to_rid.items():
                if filename in filename_to_class:
                    class_name = filename_to_class[filename]
                    feature_records.append(
                        ImageClassification(
                            Image=rid,
                            Image_Class=class_name,
                        )
                    )

            logger.info(f"  Adding {len(feature_records)} classification labels...")
            label_exe.add_features(feature_records)

        # Upload the feature values
        label_exe.upload_execution_outputs(clean_folder=True)
        logger.info(f"  Added {len(feature_records)} Image_Classification features")

    return datasets, {
        "total_images": len(all_rids),
        "training_images": len(train_rids),
        "testing_images": len(test_rids),
        "uploaded_assets": len(assets),
    }


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point."""
    if args is None:
        args = parse_args()
        # Verify Kaggle credentials
        if not args.dry_run and not verify_kaggle_credentials():
            return 1

    # Either create a new catalog or connect to existing one
    if args.create_catalog:
        logger.info(f"Creating new catalog on {args.hostname} with project name: {args.create_catalog}")

        # Create the catalog
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

        # Connect to the newly created catalog
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
        logger.info(f"Connected to catalog, domain schema: {domain_schema}")

    # Set up domain model
    logger.info("Setting up domain model...")
    setup_domain_model(ml)
    logger.info("Domain model setup complete")

    # Apply catalog annotations for Chaise web interface
    logger.info("Applying catalog annotations...")
    if hasattr(ml, "apply_catalog_annotations"):
        project_name = args.create_catalog if args.create_catalog else domain_schema
        ml.apply_catalog_annotations(
            navbar_brand_text=f"CIFAR-10 ({project_name})",
            head_title="CIFAR-10 ML Catalog",
        )
    else:
        logger.warning("apply_catalog_annotations not available in this deriva-ml; skipping.")
    # Setup dataset types
    setup_dataset_types(ml)

    datasets = None
    load_result = None
    if not args.dry_run:
        # Download CIFAR-10 from Kaggle
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = download_cifar10(temp_path)
            logger.info(f"Downloaded CIFAR-10 to: {data_dir}")

            # Load images (also creates datasets within execution for provenance)
            datasets, load_result = load_images(
                ml, data_dir, args.batch_size, max_images=args.num_images
            )
            logger.info(f"Loading complete: {load_result}")
    else:
        # In dry run mode, create datasets without execution context
        logger.info("Dry run mode - creating datasets without image upload")
        datasets = create_dataset_hierarchy(ml)

    # Get Chaise URLs for datasets if requested
    dataset_urls = {}
    if args.show_urls:
        logger.info("Fetching Chaise URLs for datasets...")
        for name, rid in datasets.items():
            try:
                url = ml.chaise_url(rid)
                dataset_urls[name] = url
                logger.info(f"  {name}: {url}")
            except Exception as e:
                logger.warning(f"  Failed to get URL for {name}: {e}")
                dataset_urls[name] = ""

    # Print summary
    print("\n" + "=" * 60)
    print("  CIFAR-10 LOADING COMPLETE")
    print("=" * 60)
    print(f"  Hostname:      {args.hostname}")
    print(f"  Catalog ID:    {catalog_id}")
    print(f"  Schema:        {domain_schema}")
    print("")
    print("  Datasets created:")
    if args.show_urls and dataset_urls:
        print(f"    - Complete:   {datasets['complete']}")
        print(f"      URL: {dataset_urls.get('complete', 'N/A')}")
        print(f"    - Split:      {datasets['split']}")
        print(f"      URL: {dataset_urls.get('split', 'N/A')}")
        print(f"    - Training:   {datasets['training']}")
        print(f"      URL: {dataset_urls.get('training', 'N/A')}")
        print(f"    - Testing:    {datasets['testing']}")
        print(f"      URL: {dataset_urls.get('testing', 'N/A')}")
    else:
        print(f"    - Complete:   {datasets['complete']}")
        print(f"    - Split:      {datasets['split']}")
        print(f"    - Training:   {datasets['training']}")
        print(f"    - Testing:    {datasets['testing']}")
    if load_result:
        print("")
        print(f"  Images loaded: {load_result['total_images']}")
        print(f"    - Training: {load_result['training_images']}")
        print(f"    - Testing:  {load_result['testing_images']}")
    if not args.show_urls:
        print("")
        print("  Tip: Use --show-urls to display Chaise URLs for each dataset")
    print("=" * 60 + "\n")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 dataset into DerivaML catalog (direct API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a new catalog and load CIFAR-10
    python load_cifar10.py --hostname localhost --create-catalog cifar10_demo

    # Load into an existing catalog
    python load_cifar10.py --hostname ml.derivacloud.org --catalog-id 99

    # Dry run (create schema/datasets only)
    python load_cifar10.py --hostname localhost --create-catalog test --dry-run

    # Load only 1000 images (500 training + 500 testing)
    python load_cifar10.py --hostname localhost --create-catalog test --num-images 1000
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
        "--batch-size",
        type=int,
        default=500,
        help="Number of images to process per batch (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up schema and datasets without downloading/uploading images",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of images to upload (default: all images)",
    )
    parser.add_argument(
        "--show-urls",
        action="store_true",
        help="Show Chaise web interface URLs for datasets in the summary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Verify Kaggle credentials
    if not args.dry_run and not verify_kaggle_credentials():
        sys.exit(1)

    sys.exit(main(args))

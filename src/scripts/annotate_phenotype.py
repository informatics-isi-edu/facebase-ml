"""Annotate file records with Phenotype feature based on biosample genotype.

Joins file → biosample → genotype to read the genotype name, then classifies
each file as:
- WildType: genotype name contains "+/+" or matches "WT"/"Wt"/"Wild type"
- Unknown: genotype name ends with "NA" suffix, or is null/empty
- Mutated: everything else (knockouts, floxed alleles, insertions, etc.)

NOTE: The file→biosample FK is a composite key (biosample + dataset), which
causes bag denormalization to fail. Instead, this script queries the catalog
directly via pathBuilder to build a file_RID → genotype_name lookup map.

Requires the Phenotype feature and Phenotype_Type vocabulary to exist in the
catalog (create them via MCP tools before running this script).

DerivaML API calls used:
- ml.feature_record_class(table_name, feature_name) → FeatureRecord subclass
- ml.lookup_dataset(rid) → Dataset
- dataset.list_dataset_members() → {table_name: [RID, ...]}
- ml.pathBuilder() → DataPath (for file→biosample→genotype join)
- execution.add_features(records: list[FeatureRecord]) → None

Run via:
    uv run deriva-ml-run +experiment=annotate_phenotype_e155 dry_run=true
    uv run deriva-ml-run +experiment=annotate_phenotype_e155
"""

from __future__ import annotations

import re

from deriva_ml import DerivaML
from deriva_ml.execution import Execution



# Known wild-type inbred strain names (case-insensitive match).
# These are standard reference strains used as controls in IMPC and other
# mouse phenotyping consortia. They don't follow the Gene+/+ naming convention
# but are wild-type for phenotyping purposes.
WILDTYPE_STRAIN_NAMES = {
    "c57bl/6n",     # IMPC reference strain
    "c57bl/6j",     # Jackson Labs reference strain
    "c57bl/6",      # Generic C57 black 6
    "ab",           # Zebrafish wild-type strain
}


def classify_genotype(genotype_name: str | None) -> str:
    """Classify a genotype name into WildType, Mutated, or Unknown.

    Rules (applied in order):
    - None/empty → Unknown
    - Known wild-type strain name (C57BL/6N, C57BL/6J, etc.) → WildType
    - Contains "+/+" → WildType (e.g., "Tfap2a+/+", "Nosip+/+", "Six1+/+")
    - Case-insensitive match for "wild type", "wild-type", "Wt", "WT" → WildType
    - Equals "Control" → WildType
    - Ends with "NA" (null allele indicator) → Unknown (e.g., "Tfap2aNA", "Fgf8NA")
    - Everything else → Mutated (e.g., "Tfap2afl/-", "Nosip-/-", "Cc2d2a-/-")
    """
    if not genotype_name or not genotype_name.strip():
        return "Unknown"

    g = genotype_name.strip()

    # Check known wild-type strain names first (exact match, case-insensitive)
    if g.lower() in WILDTYPE_STRAIN_NAMES:
        return "WildType"

    # WildType patterns
    if "+/+" in g:
        return "WildType"
    if re.search(r"\bwild[\s-]?type\b", g, re.IGNORECASE):
        return "WildType"
    if re.search(r"\bwt\b", g, re.IGNORECASE):
        return "WildType"
    if g.lower() == "control":
        return "WildType"

    # Unknown patterns — null allele suffix "NA"
    if g.endswith("NA"):
        return "Unknown"

    # Everything else is Mutated
    return "Mutated"


def _build_genotype_lookup(
    ml_instance: DerivaML,
    dataset_rid: str,
    version: str | None = None,
) -> dict[str, str | None]:
    """Build a file_RID → genotype_name lookup map using bag + catalog.

    The file→biosample FK is composite (biosample + dataset), which causes
    bag denormalization to fail (biosample table is empty in the bag).
    Workaround: use the bag's file and genotype tables, plus a catalog query
    for biosample→genotype mapping.

    Steps:
    1. Download metadata-only bag → get file table (file.biosample RIDs)
       and genotype table (genotype RID → name)
    2. Query catalog for biosample RID → genotype RID mapping
    3. Chain: file.biosample → biosample.genotype → genotype.name

    Returns:
        Dict mapping file RID to genotype name (or None if no genotype).
    """
    dataset = ml_instance.lookup_dataset(dataset_rid)
    version = version or dataset.current_version

    bag = dataset.download_dataset_bag(version=version, materialize=False)

    # Get file table from bag: file RID → biosample RID
    file_df = bag.get_table_as_dataframe("file")
    file_to_biosample = dict(zip(file_df["RID"], file_df["biosample"]))

    # Get genotype table from bag: genotype RID → genotype name
    genotype_df = bag.get_table_as_dataframe("genotype")
    # genotype.id is like "FACEBASE:3-RZ62", genotype.name is "Tfap2aNA"
    genotype_id_to_name = dict(zip(genotype_df["id"], genotype_df["name"]))
    genotype_rid_to_name = dict(zip(genotype_df["RID"], genotype_df["name"]))

    # Query catalog for biosample RID → genotype (FK value)
    # The biosample table is empty in the bag, so we need the catalog
    pb = ml_instance.pathBuilder()
    bs_table = pb.schemas["isa"].tables["biosample"]
    biosample_rows = bs_table.attributes(
        bs_table.RID, bs_table.genotype
    ).fetch()

    # biosample.genotype stores the FK value like "FACEBASE:3-RZ62"
    biosample_to_genotype_id: dict[str, str | None] = {}
    for row in biosample_rows:
        biosample_to_genotype_id[row["RID"]] = row.get("genotype")

    # Chain: file RID → biosample RID → genotype ID → genotype name
    lookup: dict[str, str | None] = {}
    for file_rid, bs_rid in file_to_biosample.items():
        if bs_rid is None:
            lookup[file_rid] = None
            continue
        genotype_id = biosample_to_genotype_id.get(bs_rid)
        if genotype_id is None:
            lookup[file_rid] = None
            continue
        # Try both ID formats
        name = genotype_id_to_name.get(genotype_id) or genotype_rid_to_name.get(genotype_id)
        lookup[file_rid] = name

    return lookup


def annotate_phenotype(
    # Source dataset
    source_dataset_rids: list[str] | None = None,
    source_version: str | None = None,
    element_table: str = "file",
    # Feature target
    feature_table: str = "file",
    feature_name: str = "Phenotype",
    term_column: str = "Phenotype_Type",
    # DerivaML integration (injected by framework)
    ml_instance: DerivaML | None = None,
    execution: Execution | None = None,
    # Unused but kept for config compatibility
    include_tables: list[str] | None = None,
    exclude_tables: list[str] | None = None,
    genotype_column: str | None = None,
    rid_column: str | None = None,
) -> None:
    """Annotate file records with Phenotype feature based on biosample genotype.

    Instead of bag denormalization (which fails on composite FKs), this:
    1. Lists dataset file members to get RIDs
    2. Queries catalog directly for file→biosample→genotype join
    3. Classifies each genotype name
    4. Writes Phenotype feature values with provenance
    """
    if ml_instance is None:
        raise ValueError(
            "ml_instance is required — it should be injected by the DerivaML "
            "framework via deriva-ml-run."
        )

    source_dataset_rids = source_dataset_rids or []

    if not source_dataset_rids:
        raise ValueError(
            "source_dataset_rids must contain at least one dataset RID."
        )

    # Get the FeatureRecord class for the Phenotype feature
    PhenotypeFeature = ml_instance.feature_record_class(feature_table, feature_name)

    # Collect file RIDs from dataset members, classify genotypes
    feature_records = []
    stats: dict[str, int] = {"WildType": 0, "Mutated": 0, "Unknown": 0}

    for rid in source_dataset_rids:
        dataset = ml_instance.lookup_dataset(rid)
        version = source_version or dataset.current_version
        print(f"Source: {dataset.description} (RID: {rid}, version: {version})")

        # Build genotype lookup: file_RID → genotype_name
        # Uses bag (file + genotype tables) + catalog query (biosample→genotype FK)
        print("  Building file→genotype lookup...")
        genotype_lookup = _build_genotype_lookup(ml_instance, rid, version)
        print(f"  Found genotype mappings for {len(genotype_lookup)} files")

        members = dataset.list_dataset_members()
        file_members = members.get(element_table, [])
        # list_dataset_members returns dicts with RID key, or plain RID strings
        file_rids = [
            m["RID"] if isinstance(m, dict) else m for m in file_members
        ]
        print(f"  File members: {len(file_rids)}")

        for file_rid in file_rids:
            genotype_name = genotype_lookup.get(file_rid)
            phenotype = classify_genotype(genotype_name)
            stats[phenotype] += 1

            record = PhenotypeFeature(**{
                feature_table: file_rid,
                term_column: phenotype,
            })
            feature_records.append(record)

    # Print distribution summary
    total = sum(stats.values())
    print(f"\nPhenotype classification ({total} files):")
    for label, count in sorted(stats.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    if execution is None:
        print(f"\n[DRY RUN] Would write {len(feature_records)} Phenotype feature values")
        return

    # Write feature values with provenance
    execution.add_features(feature_records)
    print(f"\nWrote {len(feature_records)} Phenotype feature values")

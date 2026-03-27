"""Annotate file records with Phenotype feature based on biosample genotype.

Joins file → biosample to read the Genotype field, then classifies each file as:
- WildType: genotype contains "+/+" or "wild-type" or "wt"
- Unknown: genotype is null, empty, or contains "NA" suffix
- Mutated: everything else (knockouts, floxed alleles, insertions, etc.)

Verified DerivaML API signatures:
- ml.feature_record_class(table_name, feature_name) → FeatureRecord subclass
- ml.lookup_dataset(rid) → Dataset
- dataset.list_dataset_members(version=...) → {table_name: [{RID, ...}]}
- dataset.download_dataset_bag(version=..., materialize=False) → DatasetBag
- bag.denormalize_as_dataframe(include_tables) → pd.DataFrame
- exe.add_features(records: list[FeatureRecord]) → None

Run via:
    uv run deriva-ml-run +experiment=annotate_phenotype_e155 dry_run=true
    uv run deriva-ml-run +experiment=annotate_phenotype_e155
"""

from __future__ import annotations

import re

from deriva_ml import DerivaML
from deriva_ml.execution import Execution


def classify_genotype(genotype: str | None) -> str:
    """Classify a genotype string into WildType, Mutated, or Unknown.

    Rules:
    - None/empty → Unknown
    - Contains "+/+" or "wild-type" or "wild type" or ends with "wt" → WildType
    - Ends with "NA" (null allele indicator) → Unknown
    - Everything else → Mutated
    """
    if not genotype or not genotype.strip():
        return "Unknown"

    g = genotype.strip()

    # WildType patterns
    if "+/+" in g:
        return "WildType"
    if re.search(r"\bwild[\s-]?type\b", g, re.IGNORECASE):
        return "WildType"
    if re.search(r"\bwt\b", g, re.IGNORECASE):
        return "WildType"

    # Unknown patterns — null allele suffix "NA"
    if g.upper().endswith("NA"):
        return "Unknown"

    # Everything else is Mutated
    return "Mutated"


def annotate_phenotype(
    # Source dataset
    source_dataset_rids: list[str] | None = None,
    source_version: str | None = None,
    # Denormalization
    include_tables: list[str] | None = None,
    exclude_tables: list[str] | None = None,
    element_table: str = "file",
    # Feature target
    feature_table: str = "file",
    feature_name: str = "Phenotype",
    term_column: str = "Phenotype_Type",
    # Genotype column in denormalized DataFrame
    genotype_column: str = "biosample_Genotype",
    rid_column: str = "file_RID",
    # DerivaML integration (injected by framework)
    ml_instance: DerivaML | None = None,
    execution: Execution | None = None,
) -> None:
    """Annotate file records with Phenotype feature based on biosample genotype.

    Downloads a metadata-only bag, denormalizes file + biosample tables,
    classifies each file's genotype into WildType/Mutated/Unknown, and
    writes the Phenotype feature values with provenance tracking.
    """
    if ml_instance is None:
        raise ValueError(
            "ml_instance is required — it should be injected by the DerivaML "
            "framework via deriva-ml-run."
        )

    source_dataset_rids = source_dataset_rids or []
    include_tables = include_tables or ["file", "biosample"]

    if not source_dataset_rids:
        raise ValueError(
            "source_dataset_rids must contain at least one dataset RID."
        )

    # Get the FeatureRecord class for the Phenotype feature
    PhenotypeFeature = ml_instance.feature_record_class(feature_table, feature_name)

    # Collect all file RIDs and their genotype classifications
    feature_records = []
    stats = {"WildType": 0, "Mutated": 0, "Unknown": 0}

    for rid in source_dataset_rids:
        dataset = ml_instance.lookup_dataset(rid)
        version = source_version or dataset.current_version
        print(f"Source: {dataset.description} (RID: {rid}, version: {version})")

        bag = dataset.download_dataset_bag(
            version=version,
            materialize=False,
            exclude_tables=exclude_tables,
        )
        df = bag.denormalize_as_dataframe(include_tables)
        print(f"  Denormalized: {len(df)} rows, {len(df.columns)} columns")

        # Check that required columns exist
        if genotype_column not in df.columns:
            available = [c for c in df.columns if "genotype" in c.lower() or "geno" in c.lower()]
            raise ValueError(
                f"Column '{genotype_column}' not found in denormalized DataFrame. "
                f"Available columns with 'genotype': {available}. "
                f"All columns: {list(df.columns)}"
            )
        if rid_column not in df.columns:
            available = [c for c in df.columns if "rid" in c.lower()]
            raise ValueError(
                f"Column '{rid_column}' not found in denormalized DataFrame. "
                f"Available RID columns: {available}."
            )

        for _, row in df.iterrows():
            file_rid = row[rid_column]
            genotype = row.get(genotype_column)
            phenotype = classify_genotype(genotype)
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

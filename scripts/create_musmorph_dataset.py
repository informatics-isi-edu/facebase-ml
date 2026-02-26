#!/usr/bin/env python3
"""Add all isa:file records to the MusMorph dataset on dev.facebase.org catalog 10."""

from deriva_ml import DerivaML

HOSTNAME = "dev.facebase.org"
CATALOG_ID = "10"
DOMAIN_SCHEMA = "isa"
DATASET_RID = "A5-DBAA"


def main():
    ml = DerivaML(HOSTNAME, CATALOG_ID, domain_schemas=DOMAIN_SCHEMA, default_schema=DOMAIN_SCHEMA)
    print(f"Connected to {HOSTNAME} catalog {CATALOG_ID}")

    # Get all file RIDs
    pb = ml.catalog.getPathBuilder()
    file_table = pb.schemas["isa"].tables["file"]
    file_rids = [r["RID"] for r in file_table.attributes(file_table.RID).fetch()]
    print(f"Found {len(file_rids)} files in isa:file")

    # Look up the existing dataset
    dataset = ml.lookup_dataset(DATASET_RID)
    print(f"Found dataset: RID={dataset.dataset_rid}")

    # Add all file records as members in batches
    BATCH_SIZE = 5000
    for i in range(0, len(file_rids), BATCH_SIZE):
        batch = file_rids[i:i + BATCH_SIZE]
        dataset.add_dataset_members(members={"file": batch})
        print(f"  Added batch {i // BATCH_SIZE + 1}: {len(batch)} files "
              f"(total: {min(i + BATCH_SIZE, len(file_rids))})")

    print(f"\nDone!")
    print(f"Dataset RID: {dataset.dataset_rid}")
    print(f"URL: https://{HOSTNAME}/chaise/record/#{CATALOG_ID}/deriva-ml:Dataset/RID={dataset.dataset_rid}")


if __name__ == "__main__":
    main()

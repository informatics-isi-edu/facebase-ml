# Experiment Design Decisions

Accumulated rationale for experiment design choices in this project.
Each entry captures what was decided and why.

---

### Bootstrap dataset from isa.file table (39,098 members)

Created the first dataset (RID: A9-ANT2) on dev.facebase.org catalog 19 containing all 39,098 records from `isa.file`. Used the Phase 3a bootstrap workflow (standalone script) rather than the Phase 3b subset template because no source dataset existed yet — this is the catalog's first dataset. The subset template requires downloading a bag from an existing dataset, which is a chicken-and-egg problem for the first dataset.

### Dict-form add_dataset_members with validate=False for large datasets

Used `dataset.add_dataset_members({table: rids}, validate=False)` instead of the flat list form. The flat list triggers per-RID table resolution via `resolve_rids()`, which failed with "Invalid RIDs" on the 39K-member dataset. The dict form skips resolution entirely since the table is already known. This pattern is now documented in all templates.

### Standalone script over MCP tools for dataset creation

Chose the committed-script path (`src/scripts/create_musmorph_dataset.py`) over interactive MCP tool calls. The script captures code provenance — the execution record links to a git commit hash, making the dataset creation reproducible. MCP tools create execution records but have no code reference.

### initialize_ml_schema made idempotent for clone catalogs

The MusMorph clone had empty ML vocabulary tables (Asset_Type, Asset_Role, Dataset_Type, Workflow_Type) because the clone operation copied the schema structure but not the vocabulary terms. Fixed `initialize_ml_schema` to check for existing terms before inserting (idempotent), and added a call to it in `_post_clone_operations` when the ML schema already exists.

### Complete added to standard Dataset_Type vocabulary

`Complete` was used across all dataset creation templates but was missing from the standard terms seeded by `initialize_ml_schema`. Added it so fresh and cloned catalogs have it by default without manual intervention.

### Data_Management workflow type for bootstrap operations

Used `Data_Management` as the workflow type for the bootstrap script. The standard ML vocabulary includes `Dataset_Management` but this is for the `split_dataset` and subset generation workflows. `Data_Management` is broader — it covers dataset creation, ETL, and general curation operations. The script accepts `--workflow-type` as a CLI argument so it's overridable.

## E15.5 Curated Subset (2026-03-25)

**Decision:** Created E15.5 subset dataset (RID: A9-D272) with 4,858 file members, nested under Complete dataset A9-ANT2.

**Rationale:**
- First curated subset — filters by developmental stage via file → biosample → vocab.stage join
- Includes all file types (scans, segmentations, landmarks) at E15.5, not just .mnc images
- Script is parameterized with --stage-name so it can be reused for other stages (E10.5, E18.5, etc.)
- Dataset.add_child_dataset() is the correct method, not add_nested_dataset()

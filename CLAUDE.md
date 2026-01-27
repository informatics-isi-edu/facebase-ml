# Running Notes

## Project Intent
- Learn DerivaML workflow for ML model training and tracking
- Start with small experiments using catalog integration
- Understand bdbag for data materialization

## Project Ecosystem

### ~/projects/deriva-docker
**Purpose:** Containerized DERIVA platform deployment (BETA)

**What it does:**
- Docker Compose orchestration for full DERIVA stack
- Services: Apache HTTPD, PostgreSQL, RabbitMQ, Traefik reverse proxy, Keycloak IDP
- Optional: JupyterHub, Grafana/Prometheus monitoring
- Provides local `localhost` catalog environment for development

**Key commands:**
```bash
cd ~/projects/deriva-docker/deriva
docker compose --env-file ~/.deriva-docker/env/localhost.env up    # Start stack
docker compose --env-file ~/.deriva-docker/env/localhost.env stop  # Stop stack
```

**Default test credentials:** deriva-admin/deriva-admin, deriva/deriva

### ~/projects/deriva-ml
**Purpose:** DerivaML Python library (core library)

**What it does:**
- Python library for reproducible ML workflows backed by Deriva catalogs
- Captures code provenance, configuration, outputs
- Provides APIs for dataset management, execution tracking, feature management
- Includes CLI tools: deriva-globus-auth-utils, deriva-ml-run-notebook, deriva-ml-install-kernel

**Usage:** Installed as dependency in model projects via `uv add deriva-ml`

### ~/projects/deriva-ml-model-template
**Purpose:** Template for creating ML models with DerivaML (THIS REPO)

**What it does:**
- Template repository for new ML model projects
- Pre-configured with Hydra/hydra-zen for experiment management
- Includes example CIFAR-10 CNN model and loader
- Script entrypoint (deriva_run.py) and Jupyter notebook support
- GitHub Actions for automated versioning

**Key features:**
- Configuration modules: deriva.py, datasets.py, assets.py, model configs
- Model runner framework connecting to DerivaML
- Version management with bump-version script

## Today's Progress (2026-01-22)
- Created new catalog `ml_demo_cc_0122` on localhost (catalog ID: 6)
- Located bdbag tool at `.venv/bin/bdbag`
- Materialized dataset_3-JQMG.zip to ~/projects/facebase-snapshots/ (2.4GB, 249 files)
- Created new repo `ml-demo-cc` from template at ~/projects/ml-demo-cc
- Created `load-facebase` loader script for FaceBase bdbag data
- Uploaded 125 brain images (.mnc) + 124 landmarks (.tag) to catalog 6
- Created FaceBase dataset (RID: BD6) in catalog
- Updated configs to use catalog 6 and FaceBase dataset as defaults

## New Repo: ~/projects/ml-demo-cc
- Cloned from deriva-ml-model-template
- Customized for FaceBase brain imaging workflow
- Added `load-facebase` CLI command (src/scripts/load_facebase.py)
- Configured for catalog 6 (ml_demo_cc_0122)

## Key URLs & References
- **Catalog 6 main:** https://localhost/chaise/recordset/#6/deriva-ml:Dataset
- **FaceBase dataset BD6:** https://localhost/chaise/record/#6/deriva-ml:Dataset/RID=BD6
- **Brain images:** https://localhost/chaise/recordset/#6/ml_demo_cc_0122:Brain_Image
- **Landmarks:** https://localhost/chaise/recordset/#6/ml_demo_cc_0122:Landmark
- **Bearer token:** `~/.deriva/credential.json` (for curl API access)

## Data Source
- **Origin:** FaceBase.org
- **Dataset:** "Ap2: A standardized mouse morphology dataset for MusMorph"
- **Original RID:** 3-JQMG
- **Content:** Mouse craniofacial micro-CT scans (.mnc) with anatomical landmarks (.tag)

## Repro Steps
```bash
# Create catalog via deriva-ml MCP
# Returns catalog_id: 6
# Chaise URL: https://localhost/chaise/recordset/#6/deriva-ml:Dataset

# Materialize dataset from zip file
mkdir -p ~/projects/facebase-snapshots
.venv/bin/bdbag --materialize --output-path ~/projects/facebase-snapshots ~/Downloads/dataset_3-JQMG.zip

# Create new repo from template
cd ~/projects
git clone https://github.com/informatics-isi-edu/deriva-ml-model-template.git ml-demo-cc
cd ml-demo-cc
rm -rf .git && git init
git config user.email "xuedaliu@usc.edu"
uv sync

# Load FaceBase data into catalog
uv run load-facebase --hostname localhost --catalog-id 6 --bag-path ~/projects/facebase-snapshots/dataset_3-JQMG
# Result: Dataset RID: BD6, Execution RID: ADP
```

## What Worked
- MCP tool `create_catalog` created catalog with all ML schema tables
- Bdbag materialize downloaded all 249 remote files, validated checksums
- FaceBase loader successfully created Brain_Image and Landmark asset tables
- Uploaded 125 brain images + 124 landmarks to catalog 6
- Created Dataset (RID: BD6) and linked all assets
- Configs updated: default_deriva points to catalog 6, default_dataset points to BD6

## What Failed + Fixes
- `ml.create_asset_table()` doesn't exist → Use `ml.create_asset()`
- `ExecutionConfiguration(description=...)` missing workflow → Must create workflow first with `ml.create_workflow()`
- `list_assets()` returns dicts not objects → Access RID with `a["RID"]` not `a.asset_rid`
- `ml.lookup_dataset()` doesn't exist → Use `ml.add_dataset_members(dataset_rid, members)` directly
- Dataset type "Complete" not found → Add `setup_dataset_types()` to create vocab terms first

## Open Questions
- What model to create for .mnc brain image files?
- Are there labels for brain images (for supervised learning)?
- How to read/process .mnc format (need nibabel or similar)?

## Progress (2026-01-27) - Domain Schema Enhancement

### Task: Add FaceBase Domain Schema
Per professor's instructions: "Look at the schema in the catalog and the structure of the downloaded bag and add a schema creation function to the load script."

### What Was Done

1. **Analyzed bdbag structure** at `~/projects/facebase-snapshots/dataset_3-JQMG/data/`:
   - `project.json` - Research project metadata
   - `experiment.json` - Experiments with types
   - `biosample.json` - Biological samples (125 records)
   - `file.json` - File records linking to biosamples

2. **Created `load_facebase.py`** script with `create_domain_schema()` function:
   - Location: `src/scripts/load_facebase.py`
   - CLI entry point: `uv run load-facebase`

3. **Domain tables created:**
   | Table | Type | Description |
   |-------|------|-------------|
   | Project | Regular | Research project metadata |
   | Experiment | Regular | Experiments with type/protocol |
   | Biosample | Regular | Samples with species, genotype, stage, anatomy |
   | Scan | Asset | Micro-CT scan files (.mnc) |
   | Landmark | Asset | Anatomical landmark files (.tag) |

4. **Vocabularies created:**
   - Species (Mus musculus)
   - Developmental_Stage (E10.5, E11.5, E14.5, E15.5, E18.5, Adult)
   - Anatomy (head, face)
   - Genotype
   - File_Format (MINC, MNI_TAG)

5. **Applied to Catalog 6** (ml_demo_cc_0122 schema):
   - Added new tables alongside existing Brain_Image/Landmark
   - Loaded: 1 project, 6 experiments, 125 biosamples
   - Created Dataset D9J with 374 members (biosamples + existing images)
   - URL: https://localhost/chaise/recordset/#6/deriva-ml:Dataset

6. **Created Catalog 7** (facebase schema):
   - Fresh catalog with "facebase" as domain schema name
   - All domain tables created from scratch
   - Loaded: 1 project, 6 experiments, 125 biosamples
   - Created Dataset 532 with 125 biosample members
   - URL: https://localhost/chaise/recordset/#7/deriva-ml:Dataset

### Key URLs - Catalog 7 (facebase)
- **Main:** https://localhost/chaise/recordset/#7/deriva-ml:Dataset
- **Project:** https://localhost/chaise/recordset/#7/facebase:Project
- **Experiment:** https://localhost/chaise/recordset/#7/facebase:Experiment
- **Biosample:** https://localhost/chaise/recordset/#7/facebase:Biosample
- **Dataset 532:** https://localhost/chaise/record/#7/deriva-ml:Dataset/RID=532

### API Learnings
- `DerivaML.create_table()` requires `TableDefinition` object, not kwargs
- Use `TableDefinition(name=..., column_defs=[...])` with `ColumnDefinition` objects
- `BuiltinTypes.text`, `BuiltinTypes.markdown`, etc. for column types
- No `ml.insert_records()` - use `ml.pathBuilder.schemas[schema].tables[table].insert(records)`
- No `ml.update_record()` - use `table.filter(table.RID == rid).update([updates])`
- `apply_catalog_annotations()` rebuilds navbar dropdown menus

### CLI Usage
```bash
# Create schema only (no data)
uv run load-facebase --hostname localhost --create-catalog facebase_test

# Load into existing catalog
uv run load-facebase --hostname localhost --catalog-id 6 --bag-path ~/projects/facebase-snapshots/dataset_3-JQMG

# Create new catalog and load data
uv run load-facebase --hostname localhost --create-catalog facebase --bag-path ~/projects/facebase-snapshots/dataset_3-JQMG
```

### Sharing Catalogs
To share localhost catalogs with others on different networks:
1. **Clone to shared server** (e.g., ml.derivacloud.org)
2. **Use ngrok tunnel:** `ngrok http https://localhost:443`
3. **Export as bdbag** and share the file
4. **Deploy Deriva on cloud VM** with public IP

## Next Steps
- Upload actual .mnc and .tag files to Catalog 7's Scan/Landmark tables
- Link Biosample records to their corresponding Scan/Landmark files
- Consider ML tasks: shape analysis, genotype classification, landmark prediction
- Add nibabel for .mnc file processing

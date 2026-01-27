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

## Next Session Plan (Task 4: Adapt model for FaceBase data)
- Decide on ML task for FaceBase data (classification, segmentation, shape analysis, etc.)
- Add nibabel or similar library to read .mnc MINC format files
- Create model in src/models/ adapted for brain imaging data
- Create model configs in src/configs/
- Test full workflow: load data → train → track results in catalog

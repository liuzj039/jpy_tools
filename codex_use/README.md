# Codex Notes for `jpy_tools`

Last audited: 2026-06-15
Repo path: `/datapool/home/zhijian/zhijian/github/jpy_tools`
Remote: `git@github.com:liuzj039/jpy_tools.git`
Default local branch observed: `master`

This file is written for future Codex sessions working with Zhijian. Treat it as an orientation and triage note, not as proof that every pipeline runs end to end.

## First Rules for Future Codex

1. Do not clean or overwrite user changes automatically. On 2026-06-15 the repo already had a dirty worktree before this note was added:
   - Modified: `jModule/jpy_tools/parseSnake2.py`
   - Modified: `pipeline/basecallByGuppy/snakemake/split.ipynb`
   - Untracked: `jModule/jpy_tools/__pycache__/`
   - Untracked: `jModule/jpy_tools/singleCellTools/__pycache__/`
   - Untracked: `jModule/jpy_tools/singleCellTools_v2/__pycache__/`
   - Untracked: `pipeline/bgiC4scRna/`
   - Untracked: `pipeline/mulocdeep/`
   - Untracked: `pipeline/scumiatac/.snakemake/`
2. Before recreating tools, genome references, or indexes, check Zhijian's shared resources:
   - Tools root: `/datapool/home/zhijian/tools`
   - Genome root: `/datapool/data/Users/zhijian/genome`
   - Naming guide: `/datapool/data/Users/zhijian/genome/CODEX_RESOURCE_LOCATIONS.md`
3. Prefer documenting and parameterizing historical paths instead of deleting them. Many scripts encode old HPC paths from `/public/home/liuzj`, `/public1/software/liuzj`, `/data/Zhaijx/liuzj`, and newer `/datapool/...` locations.
4. Do not assume pipeline configs are portable. Most config files are project snapshots and must be edited before reuse.
5. Be cautious around scripts that call `rm`, submit jobs, kill jobs, or run `os.system`; review command construction before running them.

## Repository Shape

- `jModule/`: installable-ish Python package plus CLI helpers. This is probably the most reusable part of the repo.
- `jModule/jpy_tools/`: reusable Python modules for single-cell analysis, R interop, Snakemake generation, genomic features, plotting, bulk RNA-seq helpers, nanopore helpers, pickle utilities, and multiprocessing helpers.
- `jModule/jpy_tools/singleCellTools/`: older but broad single-cell toolkit. Useful, but imports heavy dependencies at package import time.
- `jModule/jpy_tools/singleCellTools_v2/`: cleaner newer single-cell utilities and annotation code. Has tests under `tests/`; likely the best target for future cleanup.
- `tools/`: standalone scripts and bundled binaries. Not packaged as a Python module.
- `pipeline/`: many Snakemake/notebook pipelines, mostly historical project templates.
- `codex_use/`: notes for future Codex sessions. This directory should stay lightweight.

## Most Useful Assets

### Python package and helpers

- `jModule/jpy_tools/parseSnake2.py`: a local DSL/helper for generating Snakemake rules and recording parameter dataframes in the generated Snakefile. The generated style appears in `pipeline/scumiatac/snakefile`.
- `jModule/jpy_tools/singleCellTools_v2/utils.py`: useful AnnData helpers such as `ad2df`, `initLayer`, `getOverlap`, `splitAdata`, `testAllCountIsInt`, and layer-subsetting utilities.
- `jModule/jpy_tools/singleCellTools_v2/annotation.py`: label transfer / annotation workflows, but depends on scanpy, scvi, rpy2, and R packages.
- `jModule/jpy_tools/rTools.py`: R bridge helpers for rpy2/Seurat-style workflows.
- `jModule/jpy_tools/otherTools.py`: common utility functions such as pickle helpers exported in package `__init__.py`.
- `jModule/jsub.py`: PBS/LSF job submission wrapper. Useful on matching HPC systems, but hardcoded around old notification tools and cluster conventions.
- `jModule/jpy_runSnakemake.py`: wrapper for Snakemake execution/submission through `jsub.py`. Useful as a pattern, but not safe to run blindly.
- `jModule/jpy_qdel.py`: PBS/LSF job deletion helper. Highly environment-specific.

### Standalone tools

- `tools/geneAnnoTransfer/`: contains UCSC-style binaries `gtfToGenePred`, `genePredToBed`, `bedToBigBed`, plus `gtfToBed12_addGeneName.py`. Useful for genome annotation conversion, but the binaries make the repo heavy.
- `tools/singleCell/`: standalone scripts for nanopore/Illumina integration, BAM tagging, polyA/intron/gene annotation, SNP analysis, expression matrix generation, and plotting from AnnData.
- `tools/orthologDetect/`: wrappers/parsers for DIAMOND and OrthoFinder-style outputs.
- `tools/nanopore/`: gene count matrix helpers for bedtools/bambu outputs.

### Pipelines

- `pipeline/ChIP_seq_pipeline/`: best documented pipeline. Has README, config, sample table, main Snakefile, and split smk files. Good candidate for modernization into a reusable template.
- `pipeline/scumiatac/`: generated-style Snakemake pipeline for scUMI-ATAC. Useful as an example of `parseSnake2.py` output. Review carefully because shell commands remove intermediate FASTQ files.
- `pipeline/basecallByGuppy/`: nanopore Guppy basecalling/splitting/merging workflow. Has logs and notebook state; not fully clean as a template.
- `pipeline/10xAnalysis*`, `pipeline/dropseqAnalysisByStarsolo/`, `pipeline/bgiC4scRna/`: single-cell preprocessing templates around Cell Ranger / STARsolo / platform-specific processing. Many hardcoded historical data paths.
- `pipeline/sicelore/`: bundled Sicelore scripts, jars, example data, and outputs. Useful as an archived working environment, but very heavy and not ideal as normal Git content.
- `pipeline/polyACallerJb/` and `pipeline/polyACallerBasedOnWp/`: polyA calling examples with committed data/results; useful for reproducing old analysis, not clean as a lightweight template.
- `pipeline/mulocdeep/`: untracked as of this audit, apparently a newer MulocDeep Snakemake workflow. Do not assume it is committed or stable.
- `pipeline/calcIrRatioNanopore/` and `pipeline/analyzeNGSData/`: intron retention / bulk analysis helpers; likely useful, but need project-specific config review.

## Static Checks Performed

Commands were read-only except creating this note.

- Python AST parse: checked 94 `.py` files excluding `.git`, `__pycache__`, and `.snakemake`.
  - Result: 1 true syntax error.
- YAML parse: checked 20 `.yaml`/`.yml` files.
  - Result: 0 YAML parse errors.
- Shell syntax: checked the two `.sh` files with `bash -n`.
  - Result: no shell syntax errors detected.
- Tool availability in current Codex environment:
  - `snakemake`: not installed / not on PATH.
  - `pytest`: not installed / not on PATH.
  - `shellcheck`: not installed / not on PATH.
- Runtime validation was not performed. Do not claim pipelines are runnable until dependencies and configs are checked.

## Known Errors and Risks

### Definite syntax error

- `jModule/jpy_tools/singleCellTools/deprecated.py:141`: `unindent does not match any outer indentation level`.
  - This prevents that module from being imported or parsed.
  - It is in a deprecated module, but still worth fixing because syntax scans and package import tooling will fail on it.

### Packaging issues

- `jModule/setup.py` hardcodes `packages=['jpy_tools', 'jpy_tools.singleCellTools']`.
  - `jpy_tools.singleCellTools_v2` is not packaged even though it exists and has tests.
  - Other subpackages/tests may also be missed.
  - Prefer `setuptools.find_packages()` in a future cleanup.
- `setup.py` reads `requirements_pip.txt` relative to the current working directory.
  - Running `python /path/to/setup.py ...` from outside `jModule/` may fail.
  - Use `Path(__file__).parent / 'requirements_pip.txt'` in a future cleanup.
- Package import is heavy:
  - `jpy_tools/singleCellTools/__init__.py` imports scanpy, pandas, numpy, plotting, normalization, annotation, and R helpers at import time.
  - This makes light utility imports fragile on machines without full single-cell/R dependencies.

### Dependency mismatches / missing bundled tools

- `jModule/jpy_qdel.py` imports `from cool import F`, but `cool` is commented out in requirement files.
- `jsub.py` and `jpy_runSnakemake.py` call `jpy_sendMessage.py`, but that script was not found in this repo during audit.
- `pipeline/ChIP_seq_pipeline/Snakefile` calls `send.py`, but `send.py` was not found in this repo during audit.
- Many workflows depend on HPC commands/tools that may not exist in a fresh environment: `qsub`, `bsub`, `qstat`, `bjobs`, `samtools`, `STAR`, `cellranger`, `fastp`, `umi_tools`, `bwa`, `picard`, `bedtools`, Guppy, Singularity, R/Seurat, and others.

### Warnings worth cleaning later

Python parse emitted multiple `SyntaxWarning: invalid escape sequence` warnings, mostly in regex/R-string contexts such as `"\|"`, `"\("`, `"\s+"`, and mathtext `"\it"`.

Representative files:

- `jModule/jsub.py`
- `pipeline/calcIrRatioNanopore/scripts/step11_parseBedtoolsOutput.py`
- `jModule/jpy_tools/otherTools.py`
- `jModule/jpy_tools/genomeFeatureTools.py`
- `jModule/jpy_tools/bulkTools.py`
- `jModule/jpy_tools/rTools.py`
- `jModule/jpy_tools/parseSnake2.py`
- `jModule/jpy_tools/nanoporeTools.py`
- `jModule/jpy_tools/ReadProcess.py`
- `jModule/jpy_tools/singleCellTools_v2/utils.py`
- `jModule/jpy_tools/singleCellTools/plotting.py`
- `jModule/jpy_tools/singleCellTools/others.py`
- `jModule/jpy_tools/singleCellTools/geneEnrichInfo.py`

Use raw strings for regexes where appropriate, but do not blindly change R code strings without testing.

### Debug traps

- `jModule/jpy_tools/singleCellTools/normalize.py` contains an active `pdb.set_trace()` inside `if debug:`.
  - This is not a normal-run failure if `debug=False`, but future automation should not call it with `debug=True` unless interactive debugging is intended.

### Hardcoded paths

The repo contains many hardcoded absolute paths. Common examples:

- `/public/home/liuzj/...`
- `/public1/software/liuzj/...`
- `/data/Zhaijx/liuzj/...`
- `/public/apps/...`
- `/datapool/data/Users/zhijian/...`
- `/datapool/home/zhijian/...`

Future work should move these into config files, environment variables, or resource-location docs. For genomes/indexes, first check `/datapool/data/Users/zhijian/genome/CODEX_RESOURCE_LOCATIONS.md`.

### Git bloat / committed run products

Observed sizes on 2026-06-15:

- Whole repo: about `947M`
- `.git`: about `278M`
- `jModule`: about `4.6M`
- `tools`: about `82M`
- `pipeline`: about `583M`

Large committed or present files include:

- `pipeline/polyACallerJb/data/workspace/nanopore.fast5.gz` ~54 MB
- `pipeline/sicelore/scripts/Data/chr4.fa.gz` ~50 MB
- `pipeline/sicelore/scripts/Data/190c.clta.nanopore.reads.fq` ~29 MB
- Multiple `Sicelore-1.0*.jar` files ~19 MB each, with duplicates/old versions
- `tools/geneAnnoTransfer/genePredToBed` ~38 MB
- `tools/geneAnnoTransfer/gtfToGenePred` ~37 MB
- `tools/geneAnnoTransfer/bedToBigBed` ~9 MB
- Example BAM/BAI/log/result files under several pipelines

Do not delete these without user approval. For future cleanup, consider Git LFS, release assets, external storage, or a separate archive branch.

### Generated / runtime files

There are notebooks, logs, `.ipynb_checkpoints`, `__pycache__`, and `.snakemake` state directories. Future `.gitignore` cleanup would be useful, but do not remove current untracked files unless the user asks.

## Suggested Future Cleanup Order

1. Add or update `.gitignore` for `__pycache__/`, `.snakemake/`, `.ipynb_checkpoints/`, logs, and common large runtime outputs. Ask before removing already-tracked files.
2. Fix `singleCellTools/deprecated.py` indentation syntax error.
3. Modernize `jModule/setup.py`:
   - use `find_packages()`;
   - make requirement-file paths relative to `setup.py`;
   - consider moving to `pyproject.toml` later.
4. Split heavy imports in `singleCellTools/__init__.py` so basic utilities can import without full scanpy/R/scvi stack.
5. Decide whether `singleCellTools_v2` should become the main supported API; if yes, package it and run/add tests.
6. Parameterize old absolute paths in pipelines; write example configs with placeholders.
7. For `jsub.py`, `jpy_runSnakemake.py`, and `jpy_qdel.py`, add dry-run mode and safer command construction before using on a new cluster.
8. Convert the best-maintained pipelines into clean templates, starting with `pipeline/ChIP_seq_pipeline/`.
9. Move large example data and duplicated jars out of normal Git history only with explicit user approval.

## Useful Commands for Future Audits

Run from anywhere:

```bash
/usr/bin/git -C /datapool/home/zhijian/zhijian/github/jpy_tools status --short --branch
/usr/bin/git -C /datapool/home/zhijian/zhijian/github/jpy_tools remote -v
find /datapool/home/zhijian/zhijian/github/jpy_tools -path '*/.git' -prune -o -path '*/__pycache__' -prune -o -path '*/.snakemake' -prune -o -name '*.py' -print
find /datapool/home/zhijian/zhijian/github/jpy_tools/pipeline -maxdepth 3 -type f \( -name 'snakefile' -o -name 'Snakefile' -o -name '*.smk' -o -name 'config.yaml' -o -name 'config.yml' \) -print
du -sh /datapool/home/zhijian/zhijian/github/jpy_tools /datapool/home/zhijian/zhijian/github/jpy_tools/.git /datapool/home/zhijian/zhijian/github/jpy_tools/jModule /datapool/home/zhijian/zhijian/github/jpy_tools/tools /datapool/home/zhijian/zhijian/github/jpy_tools/pipeline
```

Python syntax check without creating `__pycache__`:

```bash
python3 - <<'PY'
from pathlib import Path
import ast
root = Path('/datapool/home/zhijian/zhijian/github/jpy_tools')
skip = {'.git', '__pycache__', '.snakemake'}
errors = []
count = 0
for path in root.rglob('*.py'):
    if any(part in skip for part in path.parts):
        continue
    count += 1
    text = path.read_text(encoding='utf-8')
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        errors.append((path, exc.lineno, exc.offset, exc.msg))
print(f'checked_py_files={count}')
print(f'syntax_errors={len(errors)}')
for path, line, col, msg in errors:
    print(f'{path}:{line}:{col}: {msg}')
PY
```

YAML parse check:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
root = Path('/datapool/home/zhijian/zhijian/github/jpy_tools')
skip = {'.git', '__pycache__', '.snakemake'}
errors = []
count = 0
for path in list(root.rglob('*.yaml')) + list(root.rglob('*.yml')):
    if any(part in skip for part in path.parts):
        continue
    count += 1
    try:
        yaml.safe_load(path.read_text())
    except Exception as exc:
        errors.append((path, type(exc).__name__, str(exc).splitlines()[0]))
print(f'checked_yaml_files={count}')
print(f'yaml_errors={len(errors)}')
for path, typ, msg in errors:
    print(f'{path}: {typ}: {msg}')
PY
```

## How to Approach New User Requests in This Repo

- If the user asks for a quick reusable script, look first in `jModule/jpy_tools/` and `tools/` before writing from scratch.
- If the user asks for a new pipeline, inspect `parseSnake2.py` and `pipeline/ChIP_seq_pipeline/` first.
- If the user asks for single-cell AnnData helpers, inspect `singleCellTools_v2/utils.py` first, then older `singleCellTools/` only if needed.
- If the user asks about HPC submission, inspect `jsub.py` and `jpy_runSnakemake.py`, but treat them as environment-specific and risky until paths/notification tools are updated.
- If the user asks for genome/index resources, check shared genome docs before building new indexes.
- If the user asks to clean the repository, separate three categories: generated caches, untracked user work, and tracked historical artifacts. Ask before deleting tracked or untracked analysis results.

# Image Trace Core

This repository owns the reusable local processing package `image_trace_core`:
image decoding/fingerprints, stateless descriptors, document image extraction and
pure NumPy matrices. The four Python files at the repository root are packaged
under that namespace by `pyproject.toml`. Keep one implementation of each
algorithm and reuse the existing functions and types.

The source is maintained at [W-World-Labs/image-trace-core](https://github.com/W-World-Labs/image-trace-core).
Keep changes small. Do not copy application services, authentication, telemetry,
deployment configuration, private dependencies, credentials, logs, databases,
customer/research files, models, fixtures or other unreviewed material here.
Examples use caller-supplied local paths; manual inputs must be synthetic and
remain outside the repository.

The MIT license covers only the W-owned code and documentation. Preserve its
original notice. Users explicitly install original external packages through the
documented extras; never vendor or patch upstream source or run an installer or
model/data download from library code. Update `README.md` and `DEPENDENCIES.json`
when imports or dependency declarations change. PyMuPDF/MuPDF remains subject to
its AGPL/commercial terms; separately installing a package does not waive its
license or confer rights in native codecs, input data or extracted output.
Use official `PyMuPDF`, not the unrelated PyPI package named `fitz`.

Use Python 3.11+ and the existing packaging commands:

```sh
python -m pip install '.[vision,pdf,heif,raw]'
python scripts/check-source.py
python -m compileall -q __init__.py image_processor.py document_parser.py feature_matrix.py scripts
python -m build --no-isolation
python scripts/check-artifacts.py dist
```

Do not add unit or integration tests. Reuse existing application checks when
working in its source workspace; use the compile, archive-closure, dependency
audit and secret-scan gates here. Keep input/output limits explicit; native
decoders and large matrices still need caller-owned process, memory and storage
controls. Do not claim runtime deployment, format/accuracy certification or
commercial clearance from a successful build.

For a release, keep the package, version records, source/artifact boundary and
manifest consistent. Use a `codex/` branch and reviewed PR. Scan current source
and full public history, build only this package, and wait for exact-head CI to
pass before tagging. Retain immutable old tags and release assets; publish a new
version for later changes. New paths must be unique after NFC normalization and
case folding. A public snapshot must never include nested private repositories
or material outside the reviewed source allowlist.

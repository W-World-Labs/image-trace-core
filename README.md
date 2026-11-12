# Image Trace Core

Version 0.1.0 is the reusable **local Python core** of Image Trace. It contains
W-owned image decoding/fingerprint/grouping helpers, stateless OpenCV descriptor
helpers, PDF/DOCX/PPTX image extraction and pure NumPy similarity matrices.
It is a library, not the Image Trace web service or desktop application.
Application database caches, tenant authorization, accounts, telemetry, deployment
configuration, UI and stored files are outside this distribution.

The application imports this same core; its old parser import is a compatibility
path. The extracted algorithms have one implementation. No upstream library source
is copied or patched. This source, wheel and sdist contain no third-party wheel,
model, image, document, research fixture, dataset or native binary. Native codecs
are supplied only by the external packages users explicitly choose to install.

## Install the original dependencies explicitly

Use Python 3.11+ and a separate virtual environment. From a reviewed checkout:

```sh
python3.11 -m venv /absolute/path/to/new-core-env
/absolute/path/to/new-core-env/bin/python -m pip install .
```

Base installation includes Pillow 13.0.0, ImageHash 4.3.2 and a compatible NumPy
2.x release. Optional backends are separate user choices:

```sh
# Choose only the features you need; these install the original PyPI packages.
/absolute/path/to/new-core-env/bin/python -m pip install '.[vision]'
/absolute/path/to/new-core-env/bin/python -m pip install '.[pdf]'
/absolute/path/to/new-core-env/bin/python -m pip install '.[heif,raw]'
```

The fixed optional packages are OpenCV headless 4.14.0.94, PyMuPDF 1.27.2.3,
pillow-heif 1.8.1 and rawpy 0.28.0. AVIF uses Pillow's built-in support; no duplicate
AVIF plugin is installed or imported. Do not install multiple
OpenCV wheel variants into one environment: they share the `cv2` namespace.
**`fitz` is not the PyPI dependency name.** Install the official `PyMuPDF` package;
the unrelated PyPI package named `fitz` must not be used. PyMuPDF's preferred
import is `pymupdf`; the compatibility import name in older official releases was
`fitz`. PDF support is optional, and Office-image extraction works without it.

These commands install software only. The library never runs pip, downloads a
model, fetches input data, contacts a service or installs a system driver at
runtime. Optional image plugins register their normal public decoding APIs if
already installed; their package code remains unchanged.

## Use local files and caller-provided vectors

```python
from image_trace_core.image_processor import compute_image_features, calculate_similarity

a = compute_image_features("/absolute/path/to/first.png")
b = compute_image_features("/absolute/path/to/second.png")
score = calculate_similarity(a["phash"], b["phash"])
```

```python
from image_trace_core.document_parser import DocumentParser

parser = DocumentParser("/absolute/path/to/uploads", "/absolute/path/to/new-extracted")
result = parser.process_document("/absolute/path/to/your-document.docx")
```

`image_processor` exposes format-aware RGB decoding, perceptual hashes,
`find_similar_images`, `group_similar_images`, optional ORB/BRISK/SIFT/AKAZE/KAZE
descriptor computation and descriptor matching scores, and explicit image resizing.
Actual format support depends on the selected Pillow build and installed original
codecs; recognizing an extension does not guarantee that a decoder is available.
`resize_image_if_needed` preserves its existing overwrite behavior when no
`output_path` is supplied; supply an output path to retain the original file.
MD5 is used only as a content fingerprint, not a security or integrity primitive.
Descriptor scores and perceptual hashes are heuristics, not identity proofs.

`feature_matrix` exposes `hash_str_to_bits`, `compute_hash_similarity_matrix`
and `compute_cosine_similarity_matrix`. Vectors are caller-supplied dictionaries
`{image_id: {variant_index: numpy_array}}`; all requested images and variants
must have complete, equal-sized vectors. Rotation mode compares eight orientations
and requires variants 0 through 7. The module does not load a database or infer
missing vectors. Limit batch counts and vector dimensions before constructing
the N-by-N matrix. Application SSIM, disk/SQL caches and cached hybrid scoring
are not part of this release.

`DocumentParser` extracts embedded PDF images with soft masks, renders PDF pages
only when embedded extraction produces no images, and reads decodable image
members from DOCX/PPTX ZIP media folders. Rendered fallback is a full-page image,
not extraction of a semantic figure. PDF rendering uses the original PyMuPDF API;
Office extraction uses Python's ZIP API and Pillow, not PyMuPDF Office. Office
video/audio entries are skipped. It does not extract text, perform OCR, recognize
objects, train models or claim document-layout accuracy.

## Input and process boundaries

This is an in-process library for files the caller is authorized to process.
It does not supply HTTP authorization, tenancy, a native-code sandbox or a task
timeout. Applications accepting untrusted documents must run native decoders in
isolated, deadline- and memory-bounded workers and enforce their own upload,
output-directory, storage-quota and file-ownership rules.

Public extraction checks a 200 MiB input limit, 1000 PDF pages/media images,
256 MiB extracted bytes per operation, and 40 million pixels for Pillow images
and rendered PDF pages. Prefixes accept only 1–64 ASCII letters/digits/`_`/`-`;
generated output names use a per-call token and exclusive creation. Office media
sizes are checked before decompression. These checks do not bound every native
decoder's temporary allocation, RAW processing, PDF parser complexity or matrix
memory. Keep OS-level worker limits. Do not share extraction directories among
mutually untrusted callers. `process_document` removes only that call's partial
outputs on a reported extraction error.

## Source, licenses and external material

The [MIT license](LICENSE), including its original copyright notice, covers only
this W-owned core and documentation. This is an extraction of Image Trace's
existing processing code, with private application imports removed, optional
backend imports clarified, and explicit input/output checks added on 2026-11-12.
No third-party algorithm implementation was copied into this repository.
The implementations of perceptual hashing, FFT/wavelets, native descriptors,
image codecs and PDF parsing remain in their original external packages.

| External software | Role and source | Original licensing/rights boundary |
| --- | --- | --- |
| [Pillow](https://github.com/python-pillow/Pillow) | Image decoding and resizing; [license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | MIT-CMU/Pillow notices; bundled codec libraries have their own notices. |
| [ImageHash](https://github.com/JohannesBuchner/imagehash) | Perceptual/hash features; [license](https://github.com/JohannesBuchner/imagehash/blob/master/LICENSE) | BSD-2-Clause; its NumPy/SciPy/PyWavelets dependencies retain separate licenses. |
| [NumPy](https://github.com/numpy/numpy) | Pixel arrays and matrix computation; [license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause and bundled native-library notices; wheel contents can add license terms. |
| [opencv-python](https://github.com/opencv/opencv-python) / [OpenCV](https://opencv.org/license/) | Optional original headless `cv2` wheel | Wrapper/build notices and OpenCV Apache-2.0 (4.5+); separately bundled native components retain their licenses. SURF/nonfree implementations are not provided. |
| [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) / MuPDF | Optional original PDF backend; [licensing](https://pymupdf.io/licensing) | AGPL-3.0 or an applicable Artifex commercial agreement. Neither this core's MIT license nor user-side pip installation removes those obligations or grants closed-source commercial rights. Assess the combined application and service; obtain the appropriate rights if AGPL requirements cannot be met. Preserve upstream producer/copyright notices. |
| [pillow-heif](https://github.com/bigcat88/pillow_heif) / [libheif](https://github.com/strukturag/libheif) | Optional HEIF codec registration | Wrapper BSD-3-Clause; libheif is LGPL-3.0, and configured codecs/binary dependencies can add their own terms. |
| [rawpy](https://github.com/letmaik/rawpy) / [LibRaw](https://www.libraw.org/download) | Optional camera RAW decoding | Wrapper MIT; LibRaw and enabled demosaicing/native components retain their original LGPL/CDDL or other notices. Inspect the selected wheel/build, not only the wrapper's MIT metadata. |

`DEPENDENCIES.json` records the direct runtime/optional/build declarations and
observed upstream metadata. It is an attribution record, not a complete resolver
lock or a legal clearance for every native wheel. Release `BUILD-DEPENDENCIES.txt`
records the actual CI environment; inspect installed distributions' license files
and upstream notices before redistributing their binaries. No external wheel or
native library is included in this project's artifacts.

There are **no bundled models, training/evaluation data, research images, sample
documents or user uploads**. Code licenses confer no rights in the files users
supply or the images/text they extract. Verify copyright, privacy, dataset terms
and commercial permissions for both inputs and outputs. This library makes no
accuracy, medical/research-validation, model-performance or business-license claim.

CI uses commit-pinned [GitHub checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python),
[upload-artifact](https://github.com/actions/upload-artifact) and
[download-artifact](https://github.com/actions/download-artifact) actions, plus
the official SHA-256-pinned [Gitleaks](https://github.com/gitleaks/gitleaks) release.
Their upstream licenses apply. Build/audit tools are public packages whose own
licenses and transitive dependency notices remain applicable.

#!/usr/bin/env python3
"""Check the reviewed source boundary, attribution and portable file names."""
import ast
import json
from pathlib import Path
import subprocess
import tomllib
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    '.github/workflows/ci.yml', '.gitignore', 'DEPENDENCIES.json', 'LICENSE',
    'MANIFEST.in', 'README.md', 'UPSTREAM.json', '__init__.py',
    'document_parser.py', 'feature_matrix.py', 'image_processor.py',
    'pyproject.toml', 'scripts/check-source.py', 'scripts/check-artifacts.py',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    entries = {}
    normalized = {}
    output = subprocess.check_output(['git', 'ls-files', '--stage', '-z', '--', '.'], cwd=ROOT).decode()
    for row in filter(None, output.split('\0')):
        metadata, name = row.split('\t', 1)
        mode, _, stage = metadata.split()
        require(mode == '100644' and stage == '0', 'Unexpected file type or merge stage: ' + name)
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), 'Only regular files are permitted: ' + name)
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe path')
        for i in range(1, len(Path(name).parts) + 1):
            prefix = '/'.join(Path(name).parts[:i])
            key = unicodedata.normalize('NFC', prefix).casefold()
            require(key not in normalized or normalized[key] == prefix, 'NFC/case path collision')
            normalized[key] = prefix
        entries[name] = mode
    require(set(entries) == FILES, 'Source inventory differs from the reviewed core-only boundary')
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    require(project['project']['name'] == 'image-trace-core' and project['project']['version'] == '0.1.0', 'Unexpected package identity')
    require(project['project']['license'] == 'MIT', 'Own core license must be explicit')
    require(project['tool']['setuptools']['packages'] == ['image_trace_core'], 'Unexpected packaged implementation')
    require(project['tool']['setuptools']['package-dir'] == {'image_trace_core': '.'}, 'Unexpected package directory')
    require((ROOT / 'LICENSE').read_text().startswith('MIT License\n'), 'Original MIT license is required')
    provenance = json.loads((ROOT / 'UPSTREAM.json').read_text())
    require(provenance['third_party_source_vendored'] is False and provenance['third_party_source_modified'] is False, 'Upstream boundary must be explicit')
    require(provenance['external_models_included'] is False and provenance['external_data_included'] is False, 'No model or data payload is permitted')
    declared = {('runtime', x) for x in project['project']['dependencies']}
    declared |= {('build', x) for x in project['build-system']['requires']}
    declared |= {(group, x) for group, values in project['project']['optional-dependencies'].items() for x in values}
    attributed = {(row['group'], row['requirement']) for pkg in json.loads((ROOT / 'DEPENDENCIES.json').read_text())['packages'] for row in pkg['declarations']}
    require(declared == attributed, 'Dependency attribution declarations are stale')
    for name in ['__init__.py', 'document_parser.py', 'feature_matrix.py', 'image_processor.py']:
        tree = ast.parse((ROOT / name).read_text())
        approved = {'hashlib', 'logging', 'os', 'io', 'uuid', 'zipfile', 'typing', 'pathlib', 'importlib', 're', 'PIL', 'numpy', 'imagehash', 'cv2', 'rawpy', 'pillow_heif', 'pymupdf', 'fitz'}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                require((node.level == 1 and node.module == 'image_processor') or (node.level == 0 and (node.module or '').split('.')[0] in approved), 'Unreviewed core import')
            if isinstance(node, ast.Import):
                require(all(x.name.split('.')[0] in approved for x in node.names), 'Unreviewed core import')
    print('Verified core-only source, attribution, original license and NFC/case paths:', len(FILES))


if __name__ == '__main__':
    main()

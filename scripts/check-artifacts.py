#!/usr/bin/env python3
"""Verify own-source closure and exclude unreviewed payloads from release archives."""
import argparse
import email
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = {'__init__.py', 'document_parser.py', 'feature_matrix.py', 'image_processor.py'}
DATA = {'README.md', 'AGENTS.md', 'UPSTREAM.json', 'DEPENDENCIES.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe archive path')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    wheels = list(args.directory.glob('*.whl'))
    sources = list(args.directory.glob('*.tar.gz'))
    require(len(wheels) == len(sources) == 1, 'Exactly one own wheel and sdist are required')
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate wheel paths')
        expected_modules = {'image_trace_core/' + name for name in MODULES}
        require({x for x in names if x.startswith('image_trace_core/')} == expected_modules, 'Unexpected package payload')
        for name in MODULES:
            require(archive.read('image_trace_core/' + name) == (ROOT / name).read_bytes(), 'Wheel/source bytes differ: ' + name)
        for name in names:
            safe_name(name)
            require(name in expected_modules or name.startswith('image_trace_core-0.1.1.dist-info/') or name.startswith('image_trace_core-0.1.1.data/data/share/image-trace-core/'), 'Unexpected wheel path: ' + name)
            require(not name.endswith(('.whl', '.pt', '.onnx', '.so', '.dll', '.jpg', '.png', '.pdf', '.docx', '.pptx', '.csv', '.db', '.pem', '.key')), 'External/data/binary payload in own wheel')
        for name in DATA:
            path = 'image_trace_core-0.1.1.data/data/share/image-trace-core/' + name
            require(archive.read(path) == (ROOT / name).read_bytes(), 'Provenance bytes differ')
        require(archive.read('image_trace_core-0.1.1.dist-info/licenses/LICENSE') == (ROOT / 'LICENSE').read_bytes(), 'Original license differs')
        metadata = email.message_from_bytes(archive.read('image_trace_core-0.1.1.dist-info/METADATA'))
        require(metadata['Name'] == 'image-trace-core' and metadata['Version'] == '0.1.1', 'Unexpected wheel identity')
    own_files = set(filter(None, subprocess.check_output(['git', 'ls-files', '-z', '--', '.'], cwd=ROOT).decode().split('\0')))
    generated = {'PKG-INFO', 'setup.cfg', 'image_trace_core.egg-info/PKG-INFO', 'image_trace_core.egg-info/SOURCES.txt', 'image_trace_core.egg-info/dependency_links.txt', 'image_trace_core.egg-info/requires.txt', 'image_trace_core.egg-info/top_level.txt'}
    with tarfile.open(sources[0]) as archive:
        files = {}
        for member in archive.getmembers():
            safe_name(member.name)
            require(not member.issym() and not member.islnk(), 'Archive links are forbidden')
            if member.isdir() and member.name == 'image_trace_core-0.1.1':
                continue
            require(member.name.startswith('image_trace_core-0.1.1/'), 'Unexpected sdist prefix')
            if member.isfile():
                relative = str(PurePosixPath(member.name.split('/', 1)[1]))
                require(relative not in files, 'Duplicate sdist file')
                require(relative in own_files | generated, 'Unexpected sdist payload: ' + relative)
                files[relative] = member
        require(own_files <= files.keys(), 'Own source/provenance missing from sdist')
        for name in own_files:
            require(archive.extractfile(files[name]).read() == (ROOT / name).read_bytes(), 'Sdist/source bytes differ: ' + name)
    print('Verified wheel/sdist source closure, original MIT license and absence of third-party/data/native payloads')


if __name__ == '__main__':
    main()

"""Resolve the inherited listener and add verified heads without replacing it.

The experiment card is read-only. Each assembled bundle occupies a fresh empty
directory; all inherited files, including auxiliary graphs and external data,
are checked before and after copying. No shipped-listener fallback is allowed.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


ASSETS = (
    'duration-negatives.json',
    'duration-negatives-features.onnx',
    'duration-negatives-matched.json',
    'duration-negatives-random.json',
)
DESCRIPTOR = 'duration-negatives-assets.json'


def fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            size += len(block)
            digest.update(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def primary_manifest(model):
    manifest = json.loads((model / 'manifest.json').read_text())
    file = Path(manifest['file'])
    if file.is_absolute() or '..' in file.parts:
        raise ValueError('Primary file must be inside its inherited bundle')
    if fingerprint(model / file)['sha256'] != manifest['sha256']:
        raise ValueError('Inherited primary checksum mismatch')
    return manifest


def resolve_listener(card, models):
    listener = json.loads(Path(card).read_text()).get('listener')
    if not isinstance(listener, str) or not listener or listener in ('.', '..') or Path(listener).name != listener:
        raise ValueError('Card must name one inherited listener bundle')
    model = Path(models) / listener
    if not model.is_dir():
        raise FileNotFoundError(f'Inherited listener bundle is unavailable: {model}')
    primary_manifest(model)
    return model


def snapshot(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Bundle assets must be regular files, not symlinks')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = fingerprint(path)
    return result


def assemble(model, fitted, out):
    model, fitted, out = Path(model), Path(fitted), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output bundle must be fresh and empty')
    inherited = snapshot(model)
    manifest = primary_manifest(model)
    additions = (*ASSETS, DESCRIPTOR)
    if any(file in inherited for file in additions):
        raise ValueError('Duration-negative assets would overwrite inherited assets')
    metadata = json.loads((fitted / ASSETS[0]).read_text())
    extractor = json.loads((fitted / 'duration-negatives-features-metadata.json').read_text())
    for parameters in (metadata, extractor):
        if (parameters.get('primarySha256') != manifest['sha256'] or
                parameters.get('tokens') != manifest['tokens'] or
                parameters.get('inventory') != manifest['inventory']):
            raise ValueError('Fitted features/head differ from the inherited primary')
    declared = json.loads((fitted / DESCRIPTOR).read_text())
    fitted_files = {file: fingerprint(fitted / file) for file in additions}
    for file in ASSETS:
        if fitted_files[file]['bytes'] <= 0 or declared.get(file) != fitted_files[file]:
            raise ValueError(f'Fitted asset checksum mismatch: {file}')
    # Validate everything before writing; never merge into a previous bundle.
    shutil.copytree(model, out, dirs_exist_ok=True)
    for file in additions:
        shutil.copyfile(fitted / file, out / file)
    expected = {**inherited, **fitted_files}
    if snapshot(model) != inherited or snapshot(out) != expected:
        raise ValueError('Bundle assembly changed or lost inherited/fitted assets')
    return dict(model=str(model), bundle=str(out), primarySha256=manifest['sha256'],
                inherited=inherited, added=fitted_files)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest='command', required=True)
    resolve = commands.add_parser('resolve')
    resolve.add_argument('--card', required=True)
    resolve.add_argument('--models', required=True)
    bundle = commands.add_parser('assemble')
    bundle.add_argument('--model', required=True)
    bundle.add_argument('--fitted', required=True)
    bundle.add_argument('--out', required=True)
    bundle.add_argument('--report', required=True)
    args = p.parse_args()
    if args.command == 'resolve':
        print(resolve_listener(args.card, args.models))
    else:
        report = assemble(args.model, args.fitted, args.out)
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(bundle=report['bundle'], inheritedFiles=len(report['inherited']))))

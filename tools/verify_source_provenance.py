"""Check maintained compatibility-host inputs separately from declared import hashes.

This verifies local source integrity, not authorship or the unverified import's
identity. --refresh updates maintained-input digests without rewriting import data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('third_party/compatibility_bridge/import_manifest.json')


def hashes(data):
    return dict(sha256=hashlib.sha256(data).hexdigest(),
                normalized_lf_sha256=hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest())


def source_path(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Maintained source path must use repository-relative forward slashes')
    path = Path(relative)
    if path.is_absolute() or path.drive or '..' in path.parts:
        raise ValueError('Maintained source path escapes repository')
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Maintained source path escapes repository')
    if not target.is_file():
        raise ValueError(f'Missing maintained source: {relative}')
    return target


def verify_manifest(root=ROOT, *, refresh=False):
    path = root / MANIFEST
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict) or document.get('schema') != 'phoenix-source-import/2':
        raise ValueError('Unsupported compatibility source manifest schema')
    entries = document.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Missing maintained source inputs')
    seen = set()
    updates = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Source input must be a record object')
        relative = entry.get('path')
        target = source_path(root, relative)
        if relative in seen:
            raise ValueError('Duplicate maintained source path')
        seen.add(relative)
        if not isinstance(entry.get('source'), str) or not entry['source']:
            raise ValueError('Missing declared import source path')
        for field in ('import_sha256', 'sha256', 'normalized_lf_sha256'):
            if not isinstance(entry.get(field), str) or not re.fullmatch('[0-9a-f]{64}', entry[field]):
                raise ValueError(f'Invalid source digest: {field}')
        actual = hashes(target.read_bytes())
        if refresh:
            updates.append((entry, actual))
        elif entry['sha256'] != actual['sha256'] and entry['normalized_lf_sha256'] != actual['normalized_lf_sha256']:
            raise ValueError(f'Maintained source hash mismatch: {relative}')
    # Validate the whole input before performing a requested refresh.
    if refresh:
        for entry, actual in updates:
            entry.update(actual)
        path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8', newline='\n')
    return len(entries)


def verify_import_snapshot(snapshot_root, root=ROOT):
    """Verify declared import bytes against a separately supplied source tree.

    Never compare adapted destinations to the import digest, refresh a declared
    digest, or imply that a local byte match authenticates historical authorship.
    """
    verify_manifest(root)
    document = json.loads((root / MANIFEST).read_text(encoding='utf-8-sig'))
    for entry in document['files']:
        relative = entry['source'].replace('\\', '/')
        imported = source_path(snapshot_root, relative)
        if hashes(imported.read_bytes())['sha256'] != entry['import_sha256']:
            raise ValueError(f'Declared import hash mismatch: {relative}')
    return len(document['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Explicitly refresh current hashes; retain import hashes')
    parser.add_argument('--import-root', type=Path,
                        help='Read-only source tree to check against preserved import digests')
    args = parser.parse_args()
    if args.refresh and args.import_root:
        parser.error('--refresh and --import-root are separate operations')
    count = (verify_import_snapshot(args.import_root.resolve()) if args.import_root
             else verify_manifest(refresh=args.refresh))
    print(json.dumps(dict(passed=True, files=count, refreshed=args.refresh,
                          declared_import_bytes_verified=bool(args.import_root),
                          authenticates_upstream=False)))


if __name__ == '__main__':
    main()

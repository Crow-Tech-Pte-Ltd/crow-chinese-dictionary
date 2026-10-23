"""Verify or explicitly download a checksum-pinned official Kaikki snapshot.

No keys, inference API or provider account is used. Existing valid snapshots are
reused without a network request. The rolling publisher URL must still provide
exactly the reviewed bytes; changed upstream bytes require a newly reviewed
provenance record rather than silent acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .common import data_path

MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
BUCKETS = ('Chinese', 'Mandarin', 'Cantonese')


def _official_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (parsed.scheme == 'https' and parsed.hostname in {'kaikki.org', 'www.kaikki.org'}
            and parsed.port in {None, 443} and not parsed.username and not parsed.password
            and parsed.path.startswith('/dictionary/') and parsed.path.endswith('.jsonl.gz')
            and not parsed.query and not parsed.fragment)


class OfficialOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _official_url(newurl):
            raise ValueError('Snapshot redirected outside approved official Kaikki downloads')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_source(provenance: Path, root: Path, *, bucket: str = 'Chinese', download: bool = False) -> dict:
    record = json.loads(Path(provenance).read_text(encoding='utf-8'))
    found = [item for item in record['downloads'] if item['languageBucket'] == bucket]
    if len(found) != 1 or bucket not in BUCKETS:
        raise ValueError('Expected exactly one reviewed snapshot for the requested bucket')
    snapshot = found[0]
    url = snapshot['url']
    expected_bytes = snapshot['compressedBytes']
    expected_hash = snapshot['sha256']
    if (not _official_url(url) or not isinstance(expected_bytes, int) or
            not 0 < expected_bytes <= MAX_DOWNLOAD_BYTES or
            len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash)):
        raise ValueError('Invalid or oversized official snapshot provenance')
    relative = '/data/kaikki.org-dictionary-' + bucket + '.jsonl.gz'
    destination = data_path(Path(root), relative)
    result = dict(languageBucket=bucket, file=relative, compressedBytes=expected_bytes,
                  sha256=expected_hash, verified=False, downloadedBytesThisRun=0)
    if destination.exists():
        if destination.stat().st_size != expected_bytes or _digest(destination) != expected_hash:
            raise ValueError('Existing snapshot differs from reviewed size/checksum; refusing to overwrite it')
        return result | {'verified': True}
    if not download:
        raise FileNotFoundError('Snapshot is absent; use --download to request the reviewed official source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + '.pending')
    opener = urllib.request.build_opener(OfficialOnlyRedirect)
    request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'crow-chinese-dictionary-source-fetch/0.1'})
    received = 0
    digest = hashlib.sha256()
    with opener.open(request, timeout=60) as response:
        length = response.headers.get('Content-Length')
        if length is not None and int(length) != expected_bytes:
            raise ValueError('Publisher size changed; review a new snapshot instead of accepting it silently')
        with pending.open('wb') as target:
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > expected_bytes or received > MAX_DOWNLOAD_BYTES:
                    raise ValueError('Official download exceeded reviewed size or 500 MiB cap')
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    if received != expected_bytes or digest.hexdigest() != expected_hash:
        raise ValueError('Downloaded source did not match the reviewed checksum; incomplete file was not promoted')
    pending.replace(destination)
    return result | {'verified': True, 'downloadedBytesThisRun': received}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, default=Path('reports/upstream-source-provenance.json'))
    parser.add_argument('--root', type=Path, default=Path('work/upstream'))
    parser.add_argument('--bucket', choices=BUCKETS, default='Chinese')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--download', action='store_true', help='download the reviewed snapshot only if absent')
    action.add_argument('--verify-only', action='store_true', help='verify retained source; never make a network request (default)')
    args = parser.parse_args(argv)
    print(json.dumps(fetch_source(args.provenance, args.root, bucket=args.bucket, download=args.download), indent=2))


if __name__ == '__main__':
    main()

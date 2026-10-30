"""Publish a prepared add-on release using existing Git Credential Manager auth.

No credentials are written to files, command arguments, logs or release assets.
The caller must create an audited publication ref/tag first. Existing releases
are never replaced by this command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request

from audit_release import ROOT
from prepare_addon_release import VERSION, CURRENT

REPO = 'RazvanManolache/dlss5-phoenix-addon'
API = 'https://api.github.com/repos/'+REPO
PUBLIC_NAME = 'RazvanManolache'
PUBLIC_EMAIL = PUBLIC_NAME+'@users.noreply.github.com'


def git(*args, capture=False):
    command = ['git', '-c', 'credential.https://github.com.helper=',
               '-c', 'credential.helper=manager', *args]
    return subprocess.check_output(command, cwd=ROOT, env={**os.environ,
        'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'}) if capture else subprocess.run(
            command, cwd=ROOT, check=True, env={**os.environ,
            'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'})


def credential():
    result = subprocess.run(['git', '-c', 'credential.https://github.com.helper=',
        '-c', 'credential.helper=manager', 'credential', 'fill'], cwd=ROOT,
        input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'}, timeout=30)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode or not fields.get('password'):
        raise RuntimeError('Existing GitHub credentials are unavailable; authenticate first.')
    return fields['password']


def request(url, token=None, method='GET', payload=None, raw=None, content_type=None):
    headers = {'User-Agent': 'Phoenix-addon-release', 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28'}
    if token:
        headers['Authorization'] = 'Bearer '+token
    if payload is not None:
        raw = json.dumps(payload).encode()
        content_type = 'application/json'
    if content_type:
        headers['Content-Type'] = content_type
    operation = urllib.request.Request(url, data=raw, headers=headers, method=method)
    with urllib.request.urlopen(operation, timeout=180) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', required=True, help='Audited publication commit/ref')
    parser.add_argument('--title', help='Public release title')
    parser.add_argument('--prerelease', action='store_true', help='Publish as a GitHub prerelease; beta/alpha/rc/preview versions imply this automatically')
    parser.add_argument('--publish', action='store_true', help='Push the ref/tag and publish a new release')
    args = parser.parse_args()
    prerelease = args.prerelease or bool(re.search(r'(?i)(?:alpha|beta|preview|rc)', VERSION))
    output = CURRENT/f'release-{VERSION}'
    receipt = json.loads((output/'VALIDATION.json').read_text())
    commit = git('rev-parse', args.ref+'^{commit}', capture=True).decode().strip()
    if receipt['commit'] != commit:
        raise ValueError('Prepared assets do not match publication commit')
    for line in (output/'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if Path(name).name != name or hashlib.sha256((output/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared asset changed: '+name)
    git('fetch', 'origin', '--prune')
    parent = git('rev-parse', 'origin/main', capture=True).decode().strip()
    if git('rev-list', '--parents', '-n', '1', commit, capture=True).decode().split() != [commit, parent]:
        raise ValueError('Publication must be one clean commit directly above current origin/main')
    identities = git('show', '-s', '--format=%an%n%ae%n%cn%n%ce', commit, capture=True).decode().splitlines()
    if identities != [PUBLIC_NAME, PUBLIC_EMAIL, PUBLIC_NAME, PUBLIC_EMAIL]:
        raise ValueError('Publication identity must use the public GitHub noreply identity')
    if not args.publish:
        print(json.dumps(dict(ready=True, commit=commit, version=VERSION, network_writes=False)))
        return
    token = credential()
    repo = json.loads(request(API, token))
    if not repo.get('permissions', {}).get('push'):
        raise RuntimeError('Existing credential does not grant push access')
    releases = json.loads(request(API+'/releases?per_page=100', token))
    if any(row['tag_name'] == VERSION for row in releases):
        raise ValueError('Release already exists; inspect it before any replacement')
    tag = subprocess.run(['git', 'show-ref', '--verify', '--quiet', 'refs/tags/'+VERSION], cwd=ROOT)
    if tag.returncode == 0:
        if git('rev-parse', VERSION+'^{commit}', capture=True).decode().strip() != commit:
            raise ValueError('Existing tag points elsewhere')
    else:
        subprocess.run(['git', '-c', 'user.name='+PUBLIC_NAME, '-c', 'user.email='+PUBLIC_EMAIL,
            'tag', '-a', VERSION, commit, '-m', 'DLSS5 Phoenix add-on '+VERSION], cwd=ROOT, check=True)
    git('push', '--atomic', 'origin', commit+':refs/heads/main', 'refs/tags/'+VERSION)
    body = (output/'RELEASE_NOTES.md').read_text(encoding='utf-8')
    base = 'https://github.com/'+REPO+'/blob/'+VERSION+'/docs/RELEASE_'+VERSION+'.md'
    body = re.sub(r'\]\((?!https?://|#)([^)]+)\)',
                  lambda match: ']('+urllib.parse.urljoin(base, match[1])+')', body)
    release = json.loads(request(API+'/releases', token, 'POST', dict(tag_name=VERSION,
        target_commitish=commit, name=args.title or 'Phoenix DLSS5 '+VERSION,
        body=body, draft=True, prerelease=prerelease)))
    for name in [line.split('  ', 1)[1] for line in (output/'SHA256SUMS.txt').read_text().splitlines()]+['SHA256SUMS.txt']:
        upload = release['upload_url'].split('{', 1)[0]+'?name='+urllib.parse.quote(name)
        asset = json.loads(request(upload, token, 'POST', raw=(output/name).read_bytes(),
            content_type='application/zip' if name.endswith('.zip') else 'application/octet-stream'))
        if asset['state'] != 'uploaded' or asset['size'] != (output/name).stat().st_size:
            raise ValueError('Asset upload incomplete: '+name)
        print('Uploaded '+name, flush=True)
    release = json.loads(request(API+'/releases/'+str(release['id']), token, 'PATCH',
                                 dict(draft=False, make_latest='false' if prerelease else 'true')))
    public = json.loads(request(API+'/releases/tags/'+VERSION, token))
    verified = []
    for asset in public['assets']:
        downloaded = request(asset['browser_download_url'])
        if hashlib.sha256(downloaded).hexdigest() != hashlib.sha256((output/asset['name']).read_bytes()).hexdigest():
            raise ValueError('Public download mismatch: '+asset['name'])
        verified.append(asset['name'])
        print('Verified public download '+asset['name'], flush=True)
    if public.get('prerelease') != prerelease or public.get('draft'):
        raise ValueError('Published release visibility/channel differs from the requested channel')
    proof = dict(published=True, version=VERSION, commit=commit, url=release['html_url'], prerelease=prerelease,
                 public_assets_sha256_verified=verified)
    (output/'PUBLISHED.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit('GitHub operation failed with HTTP '+str(error.code)) from None

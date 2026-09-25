"""Stream Kaggle output atomically, checking HTTP status and byte counts."""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from kaggle import api
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
import requests


def download_file(url,target,previous=None):
    partial=target.with_name(target.name+'.downloading')
    marker=target.with_name(target.name+'.downloading.meta.json')
    response=requests.get(url,stream=True,timeout=(30,120),headers={'Accept-Encoding':'identity'})
    response.raise_for_status()
    # Kaggle's proxy omits ETag/hash headers. Its /kf/<version>/ prefix pins
    # immutable completed-version output; the next path segment is a secret
    # access token and must never be persisted or logged.
    version_match=re.match(r'^/kf/(\d+)/',urlsplit(url).path)
    identity=dict(etag=response.headers.get('ETag'),length=response.headers.get('Content-Length'),
                  version=version_match.group(1) if version_match else None,
                  md5=next((x.strip()[4:] for x in response.headers.get('x-goog-hash','').split(',') if x.strip().startswith('md5=')),None))
    expected=int(identity['length']) if identity['length'] is not None else None
    def hashes(path):
        sha,md5=hashlib.sha256(),hashlib.md5()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):sha.update(block);md5.update(block)
        return sha.hexdigest(),base64.b64encode(md5.digest()).decode()
    if target.exists() and target.stat().st_size==expected:
        sha,md5=hashes(target)
        same_source=previous and ((identity['etag'] and previous.get('etag')==identity['etag']) or (identity['version'] and previous.get('version')==identity['version']))
        if (identity['md5'] and md5==identity['md5']) or (same_source and previous.get('sha256')==sha):
            response.close()
            return dict(bytes=expected,sha256=sha,etag=identity['etag'],version=identity['version'])
    offset=0
    if partial.exists() and marker.exists() and (identity['etag'] or identity['md5'] or identity['version']) and json.loads(marker.read_text())==identity:
        offset=partial.stat().st_size
        if expected is None or offset>=expected:offset=0
    marker.write_text(json.dumps(identity))
    if offset:
        response.close()
        headers={'Accept-Encoding':'identity','Range':f'bytes={offset}-'}
        if identity['etag']:headers['If-Range']=identity['etag']
        response=requests.get(url,stream=True,timeout=(30,120),headers=headers)
        response.raise_for_status()
        if response.status_code==200:offset=0
        elif response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
            response.close();raise RuntimeError('Invalid resumed download range')
        if identity['etag'] and response.headers.get('ETag')!=identity['etag']:
            response.close();raise RuntimeError('Remote output changed during download')
    with response,partial.open('ab' if offset else 'wb') as handle:
        for chunk in response.iter_content(1024*1024):
            if chunk:handle.write(chunk)
    size=partial.stat().st_size
    if expected is not None and size!=expected:raise RuntimeError(f'Truncated {target.name}: {size}/{expected}')
    if size==0 and target.suffix in ('.pt','.json','.csv'):raise RuntimeError(f'Empty artifact: {target.name}')
    sha,md5=hashes(partial)
    if identity['md5'] and md5!=identity['md5']:raise RuntimeError(f'Checksum mismatch: {target.name}')
    partial.replace(target);marker.unlink(missing_ok=True)
    return dict(bytes=size,sha256=sha,etag=identity['etag'],version=identity['version'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slug')
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    owner, slug = args.slug.split('/')
    manifest, token, log = {}, None, ''
    progress=root/'.download-progress.json'
    previous=json.loads(progress.read_text()) if progress.exists() else {}
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request,100,token)
            result = client.kernels.kernels_api_client.list_kernel_session_output(request)
            log = result.log or log
            # A completed training run is actionable as soon as its checkpoint
            # and small metrics files arrive.  Kaggle lists auxiliary assets
            # first, including downloaded foundation weights, which can block a
            # flaky connection before the trained checkpoint is reached.
            def priority(item):
                name = item.file_name.replace('\\', '/')
                if name.endswith('/weights/best.pt'):
                    return (0, name)
                if name.endswith('/weights/last.pt') or name.endswith('/results.csv'):
                    return (1, name)
                if name.endswith('/args.yaml') or name.endswith('/resolved_experiment.json'):
                    return (2, name)
                return (3, name)

            for item in sorted(result.files or [], key=priority):
                target = (root/item.file_name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError('Output path leaves target directory')
                target.parent.mkdir(parents=True,exist_ok=True)
                # The Kaggle output proxy can close a large response cleanly
                # before its advertised Content-Length.  The partial file is
                # valid and `download_file` resumes it with Range, so continue
                # in-process rather than waiting a full monitor interval.
                for retry in range(32):
                    try:
                        manifest[item.file_name] = download_file(
                            item.url, target, previous.get(item.file_name)
                        )
                        break
                    except RuntimeError as error:
                        if not str(error).startswith('Truncated ') or retry == 31:
                            raise
                        print(f'Resuming {item.file_name} after partial response '
                              f'({retry + 1}/32)', flush=True)
                previous[item.file_name]=manifest[item.file_name]
                temporary=progress.with_suffix('.tmp');temporary.write_text(json.dumps(previous));temporary.replace(progress)
                print(f'Verified {item.file_name}: {manifest[item.file_name]["bytes"]} bytes',flush=True)
            token = result.next_page_token
            if not token:
                break
    (root/(slug+'.log')).write_text(log,encoding='utf-8')
    (root/'.download-complete.json').write_text(json.dumps(dict(slug=args.slug,files=manifest),indent=2),encoding='utf-8')


if __name__ == '__main__':
    try:
        main()
    except requests.RequestException as error:
        # Do not put temporary signed download URLs into diagnostic logs.
        raise SystemExit(f'Download transport failed ({type(error).__name__}); retry with a fresh URL')

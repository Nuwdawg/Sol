"""Observe one existing Kaggle kernel and download terminal output; never launch."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time

from kaggle import api
from scripts.monitor_training import TERMINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slug')
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    previous = None
    while True:
        try:
            current = str(api.kernels_status(args.slug).status).rsplit('.',1)[-1].upper()
        except Exception as error:
            current = 'UNKNOWN'
            print(f'Status observation failed: {type(error).__name__}',flush=True)
        state = dict(pid=os.getpid(),slug=args.slug,status=current,
                     checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),downloaded=False)
        if current in TERMINAL:
            # The downloader has per-request timeouts and resumable transfers.
            # An overall600s kill restarted large downloads and could orphan
            # Python launcher children on Windows.
            code = subprocess.call([sys.executable,'-X','utf8','-m','scripts.download_kernel',
                                    args.slug,str(args.output)])
            state['downloaded'] = code == 0
        (args.output/'watch-state.json').write_text(json.dumps(state,indent=2))
        if current != previous or state['downloaded']:
            print(json.dumps(state),flush=True)
            previous = current
        if state['downloaded']:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()

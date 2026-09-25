"""Persist live Kaggle execution logs without launching or replacing any job."""
import argparse
import json
from pathlib import Path
import time

from kaggle import api


def force_utf8_events(response, parser):
    # SSE is UTF-8. Requests can otherwise infer Latin-1 from text/*, turning
    # progress glyph bytes into control characters that split valid JSON lines.
    response.encoding = 'utf-8'
    yield from parser(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slug')
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    original_parser = api._iter_sse_events
    api._iter_sse_events = lambda response: force_utf8_events(response, original_parser)
    seen = set(args.output.read_text(encoding='utf-8').splitlines()) if args.output.exists() else set()
    with args.output.open('a',encoding='utf-8') as output:
        while True:
            try:
                state = str(api.kernels_status(args.slug).status).upper()
                if 'QUEUED' in state:
                    time.sleep(60)
                    continue
                for event in api.kernels_logs_stream(args.slug):
                    row = json.dumps(event,sort_keys=True)
                    if row not in seen:
                        seen.add(row); output.write(row+'\n'); output.flush()
                state = str(api.kernels_status(args.slug).status).upper()
                if any(value in state for value in ('COMPLETE','ERROR','CANCEL')):
                    return
            except Exception as error:
                print(f'Log stream retry: {type(error).__name__}: {error}',flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()

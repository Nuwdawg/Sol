#!/bin/bash
SP=${SP:?set SP to a scratch directory containing papers/ and work/}
cd $SP && ls papers/97*_qp_*.pdf | xargs -P 8 -n 6 python3 -I scripts/segment.py work/seg > work/seg_log.txt 2>&1
echo done

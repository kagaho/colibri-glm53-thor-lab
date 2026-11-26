#!/usr/bin/env python3
"""Record the same real Rust prompt using this study's 32 GiB chat helper."""
import importlib.util
from pathlib import Path
import sys

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('baseline_recorder',ROOT/'scripts/record_demo.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT=STUDY
if __name__=='__main__':
    module.main()

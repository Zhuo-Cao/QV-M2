#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
submission_path=results/xxx/xxx.jsonl
gt_path=data/QV-M2/val.jsonl
save_path=xxx.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}

tfl_config_path=$1
ckpt_path=$2
eval_split_name=$3
eval_path=data/highlight_${eval_split_name}_release.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python FlashMMR/inference.py \
${tfl_config_path} \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:4}

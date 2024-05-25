# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method2, inference from local model

# for more input type, please ref to readme.md
workspace="/root/FunASR"
input_dir="${workspace}/funasr/bin/cbt1_testset_wo_prelabel"
input="${input_dir}/wav.scp"
output_dir="${workspace}/funasr/bin/outputs_zh-tw_balance_paraformer_offline_v204_cbt1_testset_wo_prelabel"


# download model
model_path_root=/project/asr_game_test/01_asr/04_funasr_latest/FunASR-1.0.14-v2/examples/industrial_data_pretraining/modelscope_models
# mkdir -p ${model_path_root}
model_path=${model_path_root}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
# git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ${model_path}
model_revision="v2.0.4"

device="cuda:0" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

tokens="${model_path}/tokens.json"
cmvn_file="${model_path}/am.mvn"

config="config.yaml"
init_param="/project/users/zhangqing/asr_paraformer_models_20240520/l22_asr_model/model.pt"

python ${workspace}/funasr/bin/inference.py \
++model="${model_path}" \
++model_revision=${model_revision} \
++init_param="${init_param}" \
++tokenizer_conf.token_list="${tokens}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++input="${input}" \
++output_dir="${output_dir}" \
++device="${device}"

python ${workspace}/runtime/python/utils/compute_wer.py ${input_dir}/text ${output_dir}/1best_recog/token ${output_dir}/wer

tail -3 ${output_dir}/wer








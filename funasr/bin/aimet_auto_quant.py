from typing import Optional
import random
import json
import logging
import sys
import os
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from funasr.auto.auto_model import AutoModel, prepare_data_iterator
from funasr.models.sanm.encoder import SANMEncoderExport
from funasr.models.paraformer.cif_predictor import CifPredictorV2Export
from funasr.models.paraformer.decoder import ParaformerSANMDecoderExport
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.paraformer.search import Hypothesis
from compute_wer import Wer

from aimet_common.quantsim_config import quantsim_config

quantsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True

from aimet_common.defs import QuantizationDataType

from aimet_torch.pro.model_preparer import prepare_model as prepare_model_pro
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.adaround.adaround_weight import AdaroundParameters
from aimet_torch.pro.auto_quant_v2 import AutoQuant
from aimet_torch.onnx_utils import OnnxExportApiArgs



logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(module)-16.16s L%(lineno)-.4d %(levelname)-5.5s| %(message)s")

log_path = os.path.join(os.path.dirname(__file__), "quant.log")
handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
handler.setLevel(logging.DEBUG)
logging_format = logging.Formatter("%(asctime)s %(module)-16.16s L%(lineno)-.4d %(levelname)-5.5s| %(message)s")
handler.setFormatter(logging_format)

LOG = logging.getLogger()
LOG.addHandler(handler)


def predictor_forward(encoder_out: torch.Tensor, predictor_model: nn.Module):
    pre_acoustic_embeds, pre_token_length = predictor_model(encoder_out)
    # 将pre_acoustic_embeds填充至(1, 68, 512)
    padding_size = 68 - pre_acoustic_embeds.size(1)
    if padding_size > 0:
        pre_acoustic_embeds = nn.functional.pad(pre_acoustic_embeds, (0, 0, 0, padding_size))
    # 将masks前pre_token_length个位置置为1
    masks = torch.zeros(1, 68).cuda()
    masks[0, :pre_token_length] = 1
    # torch.save(pre_acoustic_embeds, 'pre_acoustic_embeds.pt')
    # torch.save(masks, 'masks.pt')
    return pre_acoustic_embeds, pre_token_length, masks


class AsrDataset(Dataset):
    def __init__(self, input, automodel_kwargs, encoder_model=None, predictor_model=None, padding_seconds=4.0):
        self.key_list, self.data_list = prepare_data_iterator(input)
        self.kwargs = automodel_kwargs
        self.frontend = self.kwargs["frontend"]
        self.tokenizer = self.kwargs["tokenizer"]
        self.padding_seconds = padding_seconds
        # self.input_list_writer = open("input_list_encoder.txt", "w")
        # self.wav_filtered_writer = open("wav_filtered.scp", "w")
        self.encoder_model = encoder_model
        self.predictor_model = predictor_model

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, key = self.data_list[idx], self.key_list[idx]
        audio_fs = self.kwargs.get("fs", 16000)
        audio_sample = load_audio_text_image_video(data, fs=self.frontend.fs, audio_fs=audio_fs, data_type=self.kwargs.get("data_type", "sound"), tokenizer=self.tokenizer)
        # pad audio samples to the same length
        if audio_sample.size(0) > self.padding_seconds * audio_fs:
            # audio_sample = audio_sample[: int(self.padding_seconds * audio_fs)]
            return torch.tensor([]), ""
        else:
            audio_sample = nn.functional.pad(audio_sample, (0, int(self.padding_seconds * audio_fs) - audio_sample.size(0)))
        speech, _ = extract_fbank(audio_sample, data_type=self.kwargs.get("data_type", "sound"), frontend=self.frontend)
        # input_file_name = "./quant_data_encoder/speech_{:0>4d}".format(idx)
        # speech.numpy().astype('float32').tofile(input_file_name)
        # self.input_list_writer.write(input_file_name + '\n')
        # self.input_list_writer.flush()
        # self.wav_filtered_writer.write(key + " " + data + '\n')
        # self.wav_filtered_writer.flush()

        if self.encoder_model is None or self.predictor_model is None:
            # encoder dataset
            return speech.squeeze(dim=0), key

        # decoder dataset
        inputs_cuda = speech.cuda()
        encoder_out = self.encoder_model(inputs_cuda)
        pre_acoustic_embeds, pre_token_length, masks = predictor_forward(encoder_out, self.predictor_model)
        return (encoder_out.squeeze(dim=0).cpu(), pre_acoustic_embeds.squeeze(dim=0).cpu(), masks.squeeze(dim=0).cpu()), key


class ParaformerEncoder(nn.Module):
    def __init__(self, original_model):
        super(ParaformerEncoder, self).__init__()
        self.encoder = SANMEncoderExport(original_model.encoder, onnx=True)

    def forward(self, speech: torch.Tensor):
        return self.encoder(speech)


class ParaformerPredictor(nn.Module):
    def __init__(self, original_model):
        super(ParaformerPredictor, self).__init__()
        
        self.predictor = CifPredictorV2Export(original_model.predictor, onnx=True)
        self.register_buffer("mask", torch.ones(1, 1, 67))

    def forward(self, encoder_out: torch.Tensor):
        # mask = self.make_pad_mask(enc_len)[:, None, :]
        pre_acoustic_embeds, pre_token_length, _, _ = self.predictor(encoder_out, self.mask)
        # pre_token_length = pre_token_length.floor().type(torch.int32)
        return pre_acoustic_embeds, pre_token_length


class ParaformerDecoder(nn.Module):
    def __init__(self, original_model):
        super(ParaformerDecoder, self).__init__()
        
        self.decoder = ParaformerSANMDecoderExport(original_model.decoder, onnx=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_out: torch.Tensor, pre_acoustic_embeds: torch.Tensor, masks: torch.Tensor):
        decoder_out = self.decoder(encoder_out, pre_acoustic_embeds, masks)
        decoder_out = self.log_softmax(decoder_out)
        return decoder_out



# ==================================================================================
# Step 1. Define constants and helper functions
QUANT_TARGET = "encoder"  # encoder or decoder
# EVAL_DATASET_SIZE = 1571
EVAL_DATASET_SIZE = 1114
CALIBRATION_DATASET_SIZE = 1000
# CALIBRATION_DATASET_SIZE = 100
BATCH_SIZE = 1

_subset_samplers = {}

def _create_sampled_data_loader(dataset, num_samples):
    if num_samples not in _subset_samplers:
        indices = random.sample(range(len(dataset)), num_samples)
        _subset_samplers[num_samples] = SubsetRandomSampler(indices=indices)
    return DataLoader(dataset,
                      sampler=_subset_samplers[num_samples],
                      batch_size=BATCH_SIZE)

# Step 2. Prepare model and dataset
init_model_args = {
	"model": "/project/asr_game_test/01_asr/04_funasr_latest/FunASR-1.0.14-v2/examples/industrial_data_pretraining/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
	"model_revision": "v2.0.4",
	# "init_param": "/project/users/zhangqing/asr_paraformer_models_20240520/l22_asr_model/model.pt",
    "init_param": "/project/users/zhangqing/asr_paraformer_models_20240608/model.pt",
	"tokenizer_conf": {
		"token_list": "/project/asr_game_test/01_asr/04_funasr_latest/FunASR-1.0.14-v2/examples/industrial_data_pretraining/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json"
	},
	"frontend_conf": {
		"cmvn_file": "/project/asr_game_test/01_asr/04_funasr_latest/FunASR-1.0.14-v2/examples/industrial_data_pretraining/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn"
	},
	# "input": "/root/FunASR/funasr/bin/cbt1_testset_wo_prelabel/wav_1.scp",
    "input": "/root/FunASR/funasr/bin/cbt1_testset_wo_prelabel/wav_filtered.scp",
	"output_dir": "/root/FunASR/funasr/bin/outputs_zh-tw_balance_paraformer_offline_v204_cbt1_testset_wo_prelabel",
	"device": "cuda:0"
}

asr_model = AutoModel(**init_model_args)
automodel_kwargs = asr_model.kwargs
tokenizer = automodel_kwargs["tokenizer"]

LOG.info(f"CUDA status: {torch.cuda.is_available()}")

paraformer_model = asr_model.model
encoder_model = ParaformerEncoder(paraformer_model).cuda().eval()
predictor_model = ParaformerPredictor(paraformer_model).cuda().eval()
decoder_model = ParaformerDecoder(paraformer_model).cuda().eval()


# Step 3. Prepare unlabeled dataset
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.
class UnlabeledDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        input, _ = self._dataset[index]
        return input

# Step 4. Prepare eval callback
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.


def post_process(keys, decoder_out, pre_token_length, wer):
    batch_size = decoder_out.size(0)
    for i in range(batch_size):
        am_scores = decoder_out[i, :pre_token_length, :]

        yseq = am_scores.argmax(dim=-1)
        score = am_scores.max(dim=-1)[0]
        score = torch.sum(score, dim=-1)
        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor(
            [paraformer_model.sos] + yseq.tolist() + [paraformer_model.eos], device=yseq.device
        )
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        if isinstance(hyp.yseq, list):
            token_int = hyp.yseq[1:last_pos]
        else:
            token_int = hyp.yseq[1:last_pos].tolist()
            
        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != paraformer_model.eos and x != paraformer_model.sos and x != paraformer_model.blank_id, token_int))
        
        # Change integer-ids to tokens
        token = tokenizer.ids2tokens(token_int)
        wer.compute_wer(keys[i], token)  


if QUANT_TARGET == "encoder":
    quant_model = encoder_model
    dummy_input = torch.load("encoder_inputs.pt").cuda()
    input_names = ["speech"]
    output_names = ["xs_pad"]
    eval_dataset = AsrDataset(init_model_args["input"], automodel_kwargs)
    calibration_dataset = AsrDataset("/root/FunASR/funasr/bin/asr_train_dataset/train_wav_filtered.scp", automodel_kwargs)
else:
    quant_model = decoder_model
    dummy_input = torch.load("encoder_out.pt").cuda(), torch.load("pre_acoustic_embeds.pt").cuda(), torch.load("masks.pt").cuda()
    input_names = ["encoder_out", "pre_acoustic_embeds", "masks"]
    output_names = ["logits"]
    eval_dataset = AsrDataset(init_model_args["input"], automodel_kwargs, encoder_model, predictor_model)
    calibration_dataset = AsrDataset("/root/FunASR/funasr/bin/asr_train_dataset/train_wav_filtered.scp", automodel_kwargs, encoder_model, predictor_model)


def eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:
    if num_samples is None:
        num_samples = len(eval_dataset)
        # num_samples = 10

    eval_data_loader = _create_sampled_data_loader(eval_dataset, num_samples)
    wer = Wer("/root/FunASR/funasr/bin/cbt1_testset_wo_prelabel/text")
    # wer = Wer("/root/FunASR/funasr/bin/asr_train_dataset/text")
    pbar = tqdm(colour="blue", total=num_samples, dynamic_ncols=True)

    with torch.no_grad():
        for inputs, keys in eval_data_loader:
            if isinstance(inputs, torch.Tensor) and inputs.numel() == 0:
                continue
            # FIXME: batch_size should be 1, due to fixed shape in models
            if QUANT_TARGET == "encoder":
                inputs = inputs.cuda()
                # torch.save(inputs, 'encoder_inputs.pt')
                encoder_out = model(inputs)
                # torch.save(encoder_out, 'encoder_out.pt')
                pre_acoustic_embeds, pre_token_length, masks = predictor_forward(encoder_out, predictor_model)
                decoder_out = decoder_model(encoder_out, pre_acoustic_embeds, masks)
                post_process(keys, decoder_out, pre_token_length, wer)
                pbar.update(1)
            else:
                encoder_out = inputs[0].cuda()
                pre_acoustic_embeds = inputs[1].cuda()
                masks = inputs[2].cuda()
                pre_token_length = torch.sum(masks == 1, dim=1).item()
                decoder_out = model(encoder_out, pre_acoustic_embeds, masks)
                post_process(keys, decoder_out, pre_token_length, wer)
                pbar.update(1)


    return wer.summary()

# for _ in range(1):
#     eval_result = eval_callback(quant_model)
#     LOG.info(f"eval result: {eval_result}, WER: {1 - eval_result:.4f}")

# valid = ModelValidator.validate_model(quant_model, model_input=dummy_input)
# LOG.info(f"Model validation result: {valid}")

# prepared_model = prepare_model_pro(quant_model, dummy_input)

# valid = ModelValidator.validate_model(prepared_model, model_input=dummy_input)
# LOG.info(f"Model validation result after prepare: {valid}")

# predictor_dummy_input = torch.load('encoder_out.pt')

# with torch.no_grad():
#     torch.onnx.export(
#         predictor_model.cpu(),
#         predictor_dummy_input.cpu(), 
#         "predictor.onnx",
#         verbose=False,
#         opset_version=13,
#         do_constant_folding=True,
#         input_names=['encoder_out'],
#         output_names=["pre_acoustic_embeds", "pre_token_length"]
#         # dynamic_axes=model.export_dynamic_axes()
#     )
# LOG.info(f"predictor model is saved as 'predictor.onnx'")

# unlabeled_dataset = UnlabeledDatasetWrapper(eval_dataset)
unlabeled_dataset = UnlabeledDatasetWrapper(calibration_dataset)
unlabeled_data_loader = _create_sampled_data_loader(unlabeled_dataset, CALIBRATION_DATASET_SIZE)

# workaround for module of aimet op is None
# quant_model = prepare_model_pro(quant_model, dummy_input)

# workaround for model deepcopy issue:
# https://stackoverflow.com/questions/56590886/how-to-solve-the-run-time-error-only-tensors-created-explicitly-by-the-user-gr
with torch.no_grad():
    # Step 5. Create AutoQuant object
    auto_quant = AutoQuant(quant_model,
                        dummy_input,
                        unlabeled_data_loader,
                        eval_callback,
                        param_bw=8,
                        output_bw=16,
                        config_file="backend_aware_htp_quantsim_config_v75_all_quant.json")

    # auto_quant.set_model_preparer_params(module_classes_to_exclude=[])
    auto_quant.set_export_params(onnx_export_args=OnnxExportApiArgs(
        # set opset_version, to align with prepare_model_pro
        opset_version=12,
        # do_constant_folding=True,
        input_names=input_names,
        output_names=output_names
    ))

    # Step 6. (Optional) Set adaround params
    ADAROUND_DATASET_SIZE = 1000
    # ADAROUND_DATASET_SIZE = 100
    adaround_data_loader = _create_sampled_data_loader(unlabeled_dataset, ADAROUND_DATASET_SIZE)
    adaround_params = AdaroundParameters(adaround_data_loader, num_batches=len(adaround_data_loader))
    auto_quant.set_adaround_params(adaround_params)


    auto_quant.set_mixed_precision_params(
        candidates=[
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),      # W8A16
            # ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),       # W8A8
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)), # FP16
        ]
    )

    # Step 7. Run AutoQuant
    # sim, initial_accuracy = auto_quant.run_inference()
    # LOG.info(f"- Quantized Accuracy (before optimization): {initial_accuracy:.4f}")

    optimized_model, optimized_accuracy, encoding_path, pareto_front = auto_quant.optimize(allowed_accuracy_drop=0.01)

    if optimized_model is not None:
        LOG.info(f"- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}, WER: {1 - optimized_accuracy:.4f}")
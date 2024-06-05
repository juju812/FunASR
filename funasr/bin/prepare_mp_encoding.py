import json

import onnx

ONNX_MODEL = "/root/asr_w8a16/w8a16_asr_encoder.onnx"
ENCODING_JSON = "/root/asr_w8a16/w8a16_asr_encoder.encodings"
OUTPUT_ENCODING_JSON = "/root/asr_w8a16/w8a16_asr_encoder_mp.encodings"

QUANT_SIM_CONFIG = "backend_aware_htp_quantsim_config_v75.json"

def get_fp32_output_ops():
    with open(QUANT_SIM_CONFIG, "r") as config_fd:
        config_data = json.load(config_fd)
    fp32_output_ops = []
    for k, v in config_data["op_type"].items():
        if isinstance(v, dict) and v.get("is_output_quantized") == "False":
            fp32_output_ops.append(k)
    
    return fp32_output_ops

FP32_OUTPUT_OPS = get_fp32_output_ops()

model = onnx.load(ONNX_MODEL)

with open(ENCODING_JSON, "r") as encoding_fd:
    encoding_data = json.load(encoding_fd)

updated_outputs = []

for node in model.graph.node:
    if node.op_type not in FP32_OUTPUT_OPS:
        continue
    for output in node.output:
        # overwrite the encoding data
        encoding_data["activation_encodings"][output] = [
            {
                "bitwidth": 32,
                "dtype": "float"
            }
        ]
        updated_outputs.append(output)

with open(OUTPUT_ENCODING_JSON, "w") as output_fd:
    json.dump(encoding_data, output_fd, indent=4)

print(f"Updated {len(updated_outputs)} outputs in {OUTPUT_ENCODING_JSON}: ")

for output in updated_outputs:
    print(f"  {output}")






        

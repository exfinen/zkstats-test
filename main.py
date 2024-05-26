import torch

from zkstats.computation import State, computation_to_model
from zkstats.core import generate_data_commitment, create_dummy, prover_gen_settings, verifier_define_calculation, setup, prover_gen_proof, verifier_verify

def user_computation(s: State, data: list[torch.Tensor]) -> torch.Tensor:
    median1 = s.median(data[0])
    median2 = s.median(data[1])
    return s.mean(torch.cat((median1.unsqueeze(0), median2.unsqueeze(0))).reshape(1,-1,1))

# paths
data_commitment_path = './data_commitments.json'
data_path = './data.json'
dummy_data_path = './dummy_data.json'
precal_witness_arr_path = './precal_witness_arr.json'
proof_path = './test.pf'
prover_compiled_model_path = './prover.compiled'
prover_model_path = './prover_model.onnx'
pk_path = 'test.pk'
settings_path = './settings.json'
sel_data_path = './sel_data.json'
sel_dummy_data_path = './sel_dummy_data.json'
verifier_compiled_model_path = './verifier.compiled'
verifier_model_path = './verifier.onnx'
vk_path = 'test.vk'
witness_path = './witness.json'

scales = [2]
error = 0.01
selected_columns = ['x', 'y']

# Step 1
## Create dummy data associated with their dataset for verifier to use for generating onnx model.
create_dummy(data_path, dummy_data_path)

## Generate data commitment for their dataset.
generate_data_commitment(data_path, scales, data_commitment_path)

# Step 2
## Prover derives a model from user_computation
_, prover_model = computation_to_model(user_computation, precal_witness_arr_path, True, error)

## Prover exports onnx file, precal_witness_arr.json for verifier, and the settings for ezkl calibration
prover_gen_settings(data_path, selected_columns, sel_data_path, prover_model, prover_model_path, scales, "resources", settings_path)

# Step 3
## Verifier derives the same model using precal_witness_arr.json
_, verifier_model = computation_to_model(user_computation, precal_witness_arr_path, False, error)

verifier_define_calculation(dummy_data_path, selected_columns, sel_dummy_data_path, verifier_model, verifier_model_path)

# Step 4
## Verifier sets up vk, pk
## Prover generates proof with prover_gen_proof
setup(verifier_model_path, verifier_compiled_model_path, settings_path, vk_path, pk_path)

prover_gen_proof(prover_model_path, sel_data_path, witness_path, prover_compiled_model_path, settings_path, proof_path, pk_path)

# Step 5
## Verifier verifies
result = verifier_verify(proof_path, settings_path, vk_path, selected_columns, data_commitment_path)

print(f'result {result}')

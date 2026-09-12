from pathlib import Path
import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
import finn.core.onnx_exec as oxe


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "03_finn" / "model_streamlined.onnx"

model = ModelWrapper(str(MODEL))

input_name = model.graph.input[0].name
output_name = model.graph.output[0].name

mul_node = model.get_nodes_by_op_type("Mul")[0]

raw_tensor = None

for inp in mul_node.input:
    if model.get_initializer(inp) is None:
        raw_tensor = inp


rng = np.random.default_rng(42)

NUM_TESTES = 10000
erros = 0


for _ in range(NUM_TESTES):

    x = rng.integers(
        0,
        2,
        size=(1, 16)
    ).astype(np.float32)

    ctx = oxe.execute_onnx(
        model,
        {input_name: x},
        return_full_exec_context=True
    )

    raw = ctx[raw_tensor][0]

    final = ctx[output_name][0]

    # Classe original, após Mul + Add
    classe_original = np.argmax(final)

    r0 = int(round(raw[0]))
    r1 = int(round(raw[1]))

    # Decisão usando apenas saída INT32 do FPGA
    classe_integer = 0 if (r0 - r1) >= -20 else 1

    if classe_original != classe_integer:
        erros += 1


print("=== VALIDACAO DA DECISAO INTEGER ===")
print("Testes:", NUM_TESTES)
print("Divergencias:", erros)

if erros == 0:
    print("Decisao integer equivalente ao pos-processamento.")
else:
    print("Ainda existem divergencias.")
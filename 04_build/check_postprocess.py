from pathlib import Path
import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
import finn.core.onnx_exec as oxe


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "03_finn" / "model_streamlined.onnx"

model = ModelWrapper(str(MODEL))


print("=== POS-PROCESSAMENTO ===")

for node in model.graph.node:

    if node.op_type in ["Mul", "Add"]:

        print()
        print(node.name, node.op_type)

        for inp in node.input:

            init = model.get_initializer(inp)

            if init is not None:
                print("Constante:")
                print(init)


# ==========================================
# Descobrir tensor antes do Mul
# ==========================================

mul_node = model.get_nodes_by_op_type("Mul")[0]

data_input = None

for inp in mul_node.input:
    if model.get_initializer(inp) is None:
        data_input = inp


print()
print("Tensor antes do Mul:")
print(data_input)


# ==========================================
# Testar se argmax muda
# ==========================================

rng = np.random.default_rng(42)

input_name = model.graph.input[0].name
output_name = model.graph.output[0].name

mudancas = 0
NUM_TESTES = 1000


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

    antes = ctx[data_input]
    depois = ctx[output_name]

    classe_antes = np.argmax(antes, axis=1)
    classe_depois = np.argmax(depois, axis=1)

    if not np.array_equal(
        classe_antes,
        classe_depois
    ):
        mudancas += 1


print()
print("=== TESTE ARGMAX ===")
print("Amostras:", NUM_TESTES)
print("Classes alteradas:", mudancas)

if mudancas == 0:
    print("Argmax permaneceu igual em todos os testes.")
else:
    print("O pos-processamento pode alterar a classe.")
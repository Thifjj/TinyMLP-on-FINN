import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_shapes import InferShapes

import finn.core.onnx_exec as oxe
from finn.transformation.streamline import Streamline


# =========================
# Carregar FINN-ONNX
# =========================

model = ModelWrapper("model_finn.onnx")


# =========================
# Tidy-up
# =========================

model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveUniqueParameterTensors())
model = model.transform(GiveReadableTensorNames())
model = model.transform(RemoveStaticGraphInputs())

model.save("model_tidy.onnx")

print("=== APOS TIDY-UP ===")
for node in model.graph.node:
    print(node.op_type)


# =========================
# Entrada de teste
# =========================

x = np.array(
    [[0, 1, 1, 0,
      1, 0, 1, 1,
      0, 1, 0, 1,
      1, 0, 0, 1]],
    dtype=np.float32
)

input_name = model.graph.input[0].name
output_name = model.graph.output[0].name

out_before = oxe.execute_onnx(
    model,
    {input_name: x}
)[output_name]


# =========================
# Streamline
# =========================

model = model.transform(Streamline())
model = model.transform(RemoveUnusedTensors())

model.save("model_streamlined.onnx")

print()
print("=== APOS STREAMLINE ===")

for node in model.graph.node:
    print(node.op_type)


# =========================
# Verificacao
# =========================

input_name = model.graph.input[0].name
output_name = model.graph.output[0].name

out_after = oxe.execute_onnx(
    model,
    {input_name: x}
)[output_name]

print()
print("Saida antes:")
print(out_before)

print("\nSaida depois:")
print(out_after)

print("\nDiferenca maxima:")
print(np.max(np.abs(out_before - out_after)))

print(
    "\nMesmo resultado:",
    np.allclose(out_before, out_after, atol=1e-5)
)
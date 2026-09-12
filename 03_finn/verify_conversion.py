import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx


qonnx_model = ModelWrapper("model_clean.onnx")
finn_model = ModelWrapper("model_finn.onnx")

# Entrada simples de teste
x = np.array(
    [[0, 1, 1, 0,
      1, 0, 1, 1,
      0, 1, 0, 1,
      1, 0, 0, 1]],
    dtype=np.float32
)

# Nome das entradas
qonnx_input = qonnx_model.graph.input[0].name
finn_input = finn_model.graph.input[0].name

# Executa
qonnx_out = execute_onnx(
    qonnx_model,
    {qonnx_input: x}
)

finn_out = execute_onnx(
    finn_model,
    {finn_input: x}
)

# Pega saída principal
qonnx_output_name = qonnx_model.graph.output[0].name
finn_output_name = finn_model.graph.output[0].name

y_qonnx = qonnx_out[qonnx_output_name]
y_finn = finn_out[finn_output_name]

print("QONNX:")
print(y_qonnx)

print("\nFINN:")
print(y_finn)

print("\nDiferenca maxima:")
print(np.max(np.abs(y_qonnx - y_finn)))

print("\nMesmo resultado:",
      np.allclose(y_qonnx, y_finn, atol=1e-5))
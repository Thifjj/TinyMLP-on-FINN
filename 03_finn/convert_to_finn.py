from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

# Carrega o QONNX
model = ModelWrapper("model_clean.onnx")

print("=== ANTES ===")
print("Nos:", len(model.graph.node))

for node in model.graph.node:
    print(node.op_type)

# QONNX -> FINN-ONNX
model = model.transform(ConvertQONNXtoFINN())

# Salva
model.save("model_finn.onnx")

print()
print("=== DEPOIS ===")
print("Nos:", len(model.graph.node))

for node in model.graph.node:
    print(node.op_type)

print()
print("Modelo salvo em model_finn.onnx")
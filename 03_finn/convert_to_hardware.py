from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


model = ModelWrapper("model_streamlined.onnx")

print("=== ANTES ===")
for node in model.graph.node:
    print(node.op_type)

# MatMul quantizado -> bloco Matrix-Vector do FINN
model = model.transform(
    to_hw.InferQuantizedMatrixVectorActivation()
)

model = model.transform(GiveUniqueNodeNames())

model.save("model_hw.onnx")

print("\n=== APOS CONVERSAO PARA HW ===")
for node in model.graph.node:
    print(node.op_type, node.name)


# Especializa para a FPGA da ZCU104
model = model.transform(
    SpecializeLayers("xczu7ev-ffvc1156-2-e")
)

model = model.transform(GiveUniqueNodeNames())
for node in model.graph.node:
    if node.op_type == "MVAU_rtl":
        print("Aplicando TMR em:", node.name)
        node.op_type = "MVAU_TMR_rtl"
model.save("model_hw_specialized_tmr.onnx")

print("\n=== APOS SPECIALIZE LAYERS ===")
for node in model.graph.node:
    print(node.op_type, node.name)
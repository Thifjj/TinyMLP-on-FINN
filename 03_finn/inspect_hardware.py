from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from onnx import helper

model = ModelWrapper("model_hw_specialized.onnx")

print("=== HARDWARE FINN ===")

for node in model.graph.node:

    if node.op_type.startswith("MVAU"):
        print()
        print("=" * 50)
        print("Nome:", node.name)
        print("Tipo:", node.op_type)

        inst = getCustomOp(node)

        atributos = [
            "MW",
            "MH",
            "SIMD",
            "PE",
            "InputDataType",
            "WeightDataType",
            "OutputDataType",
            "noActivation",
            "mem_mode",
        ]

        for attr in atributos:
            try:
                valor = inst.get_nodeattr(attr)
                print(f"{attr}: {valor}")
            except Exception:
                pass

        print("\nAtributos ONNX:")

        for attr in node.attribute:
            try:
                valor = helper.get_attribute_value(attr)
                print(f"{attr.name}: {valor}")
            except Exception:
                pass
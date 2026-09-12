from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

model = ModelWrapper("model_hw_specialized_tmr.onnx")

print("=== CONFIGURACAO ORIGINAL ===")

for node in model.graph.node:
    if node.op_type.startswith("MVAU"):
        inst = getCustomOp(node)

        print(
            node.name,
            "SIMD =", inst.get_nodeattr("SIMD"),
            "PE =", inst.get_nodeattr("PE"),
            "cycles =", inst.get_exp_cycles()
        )


# ==================================================
# Configurar folding
# ==================================================

for node in model.graph.node:

    if node.name == "MVAU_hls_0":
        inst = getCustomOp(node)

        inst.set_nodeattr("SIMD", 4)
        inst.set_nodeattr("PE", 2)

    elif node.name == "MVAU_rtl_0":
        inst = getCustomOp(node)

        inst.set_nodeattr("SIMD", 1)
        inst.set_nodeattr("PE", 1)


print()
print("=== NOVA CONFIGURACAO ===")

for node in model.graph.node:
    if node.op_type.startswith("MVAU"):
        inst = getCustomOp(node)

        print(
            node.name,
            "SIMD =", inst.get_nodeattr("SIMD"),
            "PE =", inst.get_nodeattr("PE"),
            "cycles =", inst.get_exp_cycles()
        )


model.save("model_folded_tmr.onnx")

print()
print("Salvo em model_folded_tmr.onnx")
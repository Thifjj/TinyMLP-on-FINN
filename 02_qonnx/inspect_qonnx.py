import onnx

modelo = onnx.load("02_qonnx/model.onnx")

print("=== NOS DO GRAFO ===")

for i, node in enumerate(modelo.graph.node):
    print(
        f"{i}: "
        f"op={node.op_type} | "
        f"nome={node.name}"
    )
from qonnx.util.cleanup import cleanup

entrada = "02_qonnx/model.onnx"
saida = "02_qonnx/model_clean.onnx"

cleanup(
    entrada,
    out_file=saida
)

print("Modelo QONNX limpo criado:")
print(saida)
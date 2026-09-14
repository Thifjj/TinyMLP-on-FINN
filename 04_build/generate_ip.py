from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "04_build" / "partitions_tmr" / "partition_0.onnx"

PART = "xczu7ev-ffvc1156-2-e"
CLOCK_NS = 10.0


model = ModelWrapper(str(MODEL))

model = model.transform(GiveUniqueNodeNames())

print("=== GERANDO CODIGO DOS IPs ===")

model = model.transform(
    PrepareIP(
        PART,
        CLOCK_NS
    )
)

print("\n=== SINTETIZANDO NOS HLS ===")

model = model.transform(
    HLSSynthIP(
        fpgapart=PART
    )
)

saida = ROOT / "04_build" / "partition_ipgen_tmr.onnx"

model.save(str(saida))


print("\n=== RESULTADO ===")

for node in model.graph.node:

    inst = getCustomOp(node)

    print()
    print(node.name)
    print("Tipo:", node.op_type)

    try:
        print(
            "code_gen_dir_ipgen:",
            inst.get_nodeattr("code_gen_dir_ipgen")
        )
    except Exception:
        pass

    try:
        print(
            "ipgen_path:",
            inst.get_nodeattr("ipgen_path")
        )
    except Exception:
        pass


print()
print("Modelo salvo em:")
print(saida)
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw

from finn.transformation.fpgadataflow.specialize_layers import (
    SpecializeLayers
)

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition
)


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "03_finn" / "model_folded_tmr.onnx"

PART = "xczu7ev-ffvc1156-2-e"

PARTITION_DIR = ROOT / "04_build" / "partitions_tmr"
PARTITION_DIR.mkdir(exist_ok=True)


model = ModelWrapper(str(MODEL))


# ==========================================
# MultiThreshold restante -> HW
# ==========================================

model = model.transform(
    to_hw.InferThresholdingLayer()
)

model = model.transform(
    SpecializeLayers(PART)
)

model = model.transform(GiveUniqueNodeNames())
model = model.transform(InferShapes())
model = model.transform(InferDataTypes())


print("=== GRAFO ANTES DA PARTICAO ===")

for node in model.graph.node:
    print(node.op_type, node.name)


model.save(
    str(ROOT / "04_build" / "model_before_partition_tmr.onnx")
)


# ==========================================
# Criar dataflow partition
# ==========================================

parent = model.transform(
    CreateDataflowPartition(
        partition_model_dir=str(PARTITION_DIR)
    )
)

parent.save(
    str(ROOT / "04_build" / "model_parent_tmr.onnx")
)


# ==========================================
# Encontrar modelo da particao
# ==========================================

partitions = parent.get_nodes_by_op_type(
    "StreamingDataflowPartition"
)

print()
print("Numero de particoes:", len(partitions))

assert len(partitions) == 1

partition_node = getCustomOp(partitions[0])

child_path = partition_node.get_nodeattr("model")

print("Modelo dataflow:")
print(child_path)


# ==========================================
# Mostrar conteudo
# ==========================================

child = ModelWrapper(child_path)

print()
print("=== DENTRO DO ACELERADOR ===")

for node in child.graph.node:
    print(node.op_type, node.name)
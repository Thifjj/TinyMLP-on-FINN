from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths
)


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "04_build" / "partition_ipgen.onnx"

PART = "xczu7ev-ffvc1156-2-e"
CLOCK_NS = 10.0


model = ModelWrapper(str(MODEL))


# ==========================
# Corrigir larguras AXI Stream
# ==========================

model = model.transform(InsertDWC())


# ==========================
# Inserir FIFOs
# ==========================

model = model.transform(
    InsertFIFO(create_shallow_fifos=True)
)


# ==========================
# Especializar novos nós
# ==========================

model = model.transform(
    SpecializeLayers(PART)
)

model = model.transform(
    GiveUniqueNodeNames()
)


print("=== GRAFO PREPARADO ===")

for node in model.graph.node:
    print(node.op_type, node.name)


# ==========================
# Gerar IP dos novos nós
# ==========================

model = model.transform(
    PrepareIP(PART, CLOCK_NS)
)

model = model.transform(
    HLSSynthIP(fpgapart=PART)
)

model = model.transform(
    ReplaceVerilogRelPaths()
)


saida = ROOT / "04_build" / "partition_ready_stitch.onnx"

model.save(str(saida))

print()
print("Salvo em:")
print(saida)
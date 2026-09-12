from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "03_finn" / "model_folded.onnx"

FPGA_PART = "xczu7ev-ffvc1156-2-e"

model = ModelWrapper(str(MODEL))


# =============================
# Ciclos
# =============================

cycles = exp_cycles_per_layer(model)

print("=== CICLOS ESTIMADOS ===")

for node, value in cycles.items():
    print(f"{node}: {value} ciclos")


# =============================
# Recursos
# =============================

resources = res_estimation(
    model,
    FPGA_PART
)

print("\n=== RECURSOS ESTIMADOS ===")

for node, values in resources.items():

    print(f"\n{node}")

    for resource, value in values.items():
        print(f"  {resource}: {value}")
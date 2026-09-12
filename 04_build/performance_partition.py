from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "04_build" / "partitions" / "partition_0.onnx"

CLOCK_NS = 10.0  # 100 MHz


model = ModelWrapper(str(MODEL))

# Adiciona estimativa de ciclos em cada nó
model = model.transform(AnnotateCycles())

# Analisa a partição dataflow completa
perf = model.analysis(dataflow_performance)


print("=== PERFORMANCE DA PARTICAO FPGA ===")

for nome, valor in perf.items():
    print(f"{nome}: {valor}")


max_cycles = perf["max_cycles"]
critical_path = perf["critical_path_cycles"]

clock_hz = 1e9 / CLOCK_NS

throughput = clock_hz / max_cycles
latency_ns = critical_path * CLOCK_NS


print()
print("=== CLOCK ===")
print(f"Periodo: {CLOCK_NS} ns")
print(f"Frequencia: {clock_hz / 1e6:.2f} MHz")

print()
print("=== THROUGHPUT TEORICO ===")
print(f"Gargalo: {max_cycles} ciclos/amostra")
print(f"Throughput: {throughput:.2f} amostras/s")

print()
print("=== LATENCIA ESTIMADA ===")
print(f"Critical path: {critical_path} ciclos")
print(f"Latencia: {latency_ns:.2f} ns")
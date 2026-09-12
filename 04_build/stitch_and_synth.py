from pathlib import Path
import json
import shutil

from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.fpgadataflow.create_stitched_ip import (
    CreateStitchedIP
)

from finn.util.vivado import parse_ooc_synth_results


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "04_build" / "partition_ready_stitch.onnx"

PART = "xczu7ev-ffvc1156-2-e"
CLOCK_NS = 10.0


# =========================
# Carregar acelerador
# =========================

model = ModelWrapper(str(MODEL))


print("=== CRIANDO STITCHED IP ===")


# =========================
# Stitch + síntese + P&R
# =========================

model = model.transform(
    CreateStitchedIP(
        PART,
        CLOCK_NS,
        ip_name="tiny_mlp_finn",
        run_pnr=True
    )
)


# =========================
# Salvar ONNX
# =========================

saida = ROOT / "04_build" / "partition_stitched.onnx"

model.save(str(saida))


# =========================
# Projeto Vivado
# =========================

vivado_proj = model.get_metadata_prop(
    "vivado_stitch_proj"
)

print()
print("=== PROJETO VIVADO ===")
print(vivado_proj)


# =========================
# Ler resultados OOC
# =========================

results = parse_ooc_synth_results(
    vivado_proj
)


print()
print("=== RESULTADOS REAIS ===")

if results is None:
    print("Nao foi possivel interpretar os relatorios.")
else:
    for key, value in results.items():
        print(f"{key}: {value}")


# =========================
# Salvar resultados
# =========================

report_dir = ROOT / "04_build" / "ooc_reports"
report_dir.mkdir(exist_ok=True)

if results is not None:
    with open(
        report_dir / "ooc_results.json",
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )


# Copiar os relatórios importantes
for nome in [
    "ooc_utilization.rpt",
    "ooc_timing.rpt",
    "ooc_power.rpt",
    "ooc_metadata.txt"
]:
    origem = Path(vivado_proj) / nome

    if origem.exists():
        shutil.copy(
            origem,
            report_dir / nome
        )


print()
print("ONNX salvo em:")
print(saida)

print()
print("Relatorios salvos em:")
print(report_dir)
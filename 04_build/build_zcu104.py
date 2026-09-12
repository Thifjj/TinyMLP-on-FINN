from pathlib import Path
import shutil

from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "04_build" / "partitions" / "partition_0.onnx"

OUTPUT = ROOT / "04_build" / "zcu104_deploy"
OUTPUT.mkdir(exist_ok=True)

PARTITION_DIR = ROOT / "04_build" / "zcu104_partitions"
PARTITION_DIR.mkdir(exist_ok=True)


model = ModelWrapper(str(MODEL))


print("=== BUILD ZCU104 ===")

model = model.transform(
    ZynqBuild(
        platform="ZCU104",
        period_ns=10.0,
        partition_model_dir=str(PARTITION_DIR)
    )
)


# Salvar modelo final
model_path = OUTPUT / "tiny_mlp_zcu104.onnx"
model.save(str(model_path))


# Arquivos gerados pelo FINN
bitfile = model.get_metadata_prop("bitfile")
hwhfile = model.get_metadata_prop("hw_handoff")
vivado_project = model.get_metadata_prop("vivado_pynq_proj")


print()
print("=== RESULTADO ===")
print("Bitstream:", bitfile)
print("HWH:", hwhfile)
print("Projeto Vivado:", vivado_project)


# Copiar para fora de /tmp
bit_dest = OUTPUT / "tiny_mlp.bit"
hwh_dest = OUTPUT / "tiny_mlp.hwh"

shutil.copy(bitfile, bit_dest)
shutil.copy(hwhfile, hwh_dest)


print()
print("=== DEPLOY ===")
print("BIT:", bit_dest)
print("HWH:", hwh_dest)
print("ONNX:", model_path)
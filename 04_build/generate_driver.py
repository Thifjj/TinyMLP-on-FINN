from pathlib import Path
import shutil

from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.make_driver import MakePYNQDriver


ROOT = Path(__file__).resolve().parents[1]

MODEL = (
    ROOT
    / "04_build"
    / "zcu104_deploy"
    / "tiny_mlp_zcu104.onnx"
)

DEPLOY = ROOT / "04_build" / "zcu104_deploy"
DRIVER_OUT = DEPLOY / "driver"


model = ModelWrapper(str(MODEL))

print("=== GERANDO DRIVER PYNQ ===")

model = model.transform(
    MakePYNQDriver("zynq-iodma")
)

driver_tmp = Path(
    model.get_metadata_prop("pynq_driver_dir")
)

print()
print("Driver temporario:")
print(driver_tmp)


# copiar para pasta persistente
shutil.copytree(
    driver_tmp,
    DRIVER_OUT,
    dirs_exist_ok=True
)

# salvar ONNX com metadata do driver
model.save(
    str(DEPLOY / "tiny_mlp_zcu104_driver.onnx")
)

print()
print("=== DRIVER SALVO ===")
print(DRIVER_OUT)

print()
print("Conteudo:")

for item in DRIVER_OUT.iterdir():
    print(item.name)
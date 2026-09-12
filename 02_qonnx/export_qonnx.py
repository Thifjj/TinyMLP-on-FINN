from pathlib import Path
import sys
import torch

from brevitas.export import export_qonnx

# Raiz do projeto
ROOT = Path(__file__).resolve().parents[1]

# Permite importar model.py de 01_train
sys.path.insert(0, str(ROOT / "01_train"))

from model import TinyMLP


# =========================
# Carregar modelo
# =========================

model = TinyMLP()

pesos = torch.load(
    ROOT / "01_train" / "tiny_mlp.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(pesos)
model.eval()


# =========================
# Entrada de exemplo
# =========================

dummy_input = torch.zeros(1, 16)


# =========================
# Exportar QONNX
# =========================

saida = ROOT / "02_qonnx" / "model.onnx"

export_qonnx(
    model,
    export_path=str(saida),
    input_t=dummy_input
)

print(f"QONNX exportado para: {saida}")
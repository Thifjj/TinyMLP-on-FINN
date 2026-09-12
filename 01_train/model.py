import torch
import torch.nn as nn
import brevitas.nn as qnn


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.quant_in = qnn.QuantIdentity(
            bit_width=4,
            return_quant_tensor=True
        )

        self.fc1 = qnn.QuantLinear(
            in_features=16,
            out_features=8,
            bias=True,
            weight_bit_width=4
        )

        self.relu1 = qnn.QuantReLU(
            bit_width=4,
            return_quant_tensor=True
        )

        self.fc2 = qnn.QuantLinear(
            in_features=8,
            out_features=2,
            bias=True,
            weight_bit_width=4
        )

    def forward(self, x):
        x = self.quant_in(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x
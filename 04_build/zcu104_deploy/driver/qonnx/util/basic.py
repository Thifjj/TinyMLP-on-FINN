# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This is a minimal version of qonnx.util.basic, containing only the subset of
# functions required by the generated PYNQ driver. It is trimmed down to
# keep the runtime dependencies on the deployment board lightweight
# (avoiding heavy imports such as onnx / bitstring). Refer to the full
# qonnx.util.basic in the original source tree for the complete implementation.

import numpy as np
from typing import cast

from qonnx.core.datatype import BaseDataType, DataType, FixedPointType


def roundup_to_integer_multiple(x: int, factor: int) -> int:
    """Round up integer x to the nearest integer multiple of integer factor.
    Returns x if factor is set to -1. Both x and factor must otherwise be
    positive."""
    # ensure integers
    assert int(x) == x, "The input x is not an integer."
    assert int(factor) == factor, "The input factor is not an integer."
    # use -1 to indicate no padding needed
    if factor == -1:
        return x
    # ensure positive values
    assert factor > 0 and x > 0, "Factor and x are <= 0."
    if x < factor:
        return factor
    else:
        if x % factor == 0:
            return x
        else:
            return x + (factor - (x % factor))



def gen_finn_dt_tensor(finn_dt: BaseDataType, tensor_shape: tuple[int, ...] | list[int]) -> np.ndarray:
    """Generates random tensor in given shape and with given QONNX DataType."""
    if type(tensor_shape) is list:
        tensor_shape = tuple(tensor_shape)
    if finn_dt == DataType["BIPOLAR"]:
        tensor_values = np.random.randint(2, size=tensor_shape)
        tensor_values = 2 * tensor_values - 1
    elif finn_dt == DataType["BINARY"]:
        tensor_values = np.random.randint(2, size=tensor_shape)
    elif "INT" in finn_dt.name or finn_dt == DataType["TERNARY"]:
        tensor_values = np.random.randint(
            int(finn_dt.min()), high=int(finn_dt.max()) + 1, size=tensor_shape, dtype=finn_dt.to_numpy_dt()
        )
    elif "FIXED" in finn_dt.name:
        int_dt = DataType["INT" + str(finn_dt.bitwidth())]
        tensor_values = np.random.randint(int(int_dt.min()), high=int(int_dt.max()) + 1, size=tensor_shape, dtype=int_dt.to_numpy_dt())
        tensor_values = tensor_values * cast("FixedPointType",finn_dt).scale_factor()
    elif finn_dt in [DataType["FLOAT32"], DataType["FLOAT16"]]:
        tensor_values = np.random.randn(*tensor_shape)
    else:
        raise ValueError("Datatype {} is not supported, no tensor could be generated".format(finn_dt))
    # always use float type as container
    if finn_dt == DataType["FLOAT16"]:
        return tensor_values.astype(np.float16)
    else:
        return tensor_values.astype(np.float32)

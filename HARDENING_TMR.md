# Modificações para a MVAU com TMR

Este documento registra as alterações feitas no FINN e no projeto TinyMLP para adicionar uma nova camada `MVAU_TMR_rtl`, mantendo a `MVAU_rtl` original disponível.

## Objetivo

A nova camada triplica o núcleo RTL `mvu` e aplica votação majoritária bit a bit nas saídas `vld` e `p`. Uma divergência entre as réplicas ativa `fault_detected` até o próximo reset.

## Alterações no FINN

### `finn-rtllib/mvu_tmr/`

Foi criada como cópia de `finn-rtllib/mvu/`, preservando os arquivos auxiliares da implementação original.

#### `mvu_tmr.sv`

Novo wrapper RTL que:

- instancia três núcleos `mvu`: `replica0`, `replica1` e `replica2`;
- usa `KEEP_HIERARCHY` e `DONT_TOUCH` nas três instâncias;
- calcula `p` e `vld` por votação majoritária;
- detecta divergência entre as réplicas;
- mantém `fault_detected` ativo até `rst`.

O núcleo usado pelas três réplicas continua sendo o original:

```text
finn-rtllib/mvu/mvu.sv
```

#### `mvu_vvu_axi.sv`

Somente a cópia dentro de `mvu_tmr/` foi modificada. A instanciação:

```systemverilog
mvu #(...)
```

foi substituída por:

```systemverilog
mvu_tmr #(...)
```

Também foi adicionada a conexão:

```systemverilog
.fault_detected()
```

O sinal está atualmente sem conexão externa; portanto, a correção por votação funciona, mas o software não recebe a indicação da falha.

### `src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_tmr_rtl.py`

Foi criada a classe:

```python
class MVAU_TMR_rtl(MVAU_rtl):
```

Ela herda toda a implementação da `MVAU_rtl` original e altera apenas o necessário:

- estima três vezes o número de DSPs;
- usa `finn-rtllib/mvu_tmr/mvu_vvu_axi_wrapper.v`;
- troca o `mvu_vvu_axi.sv` original pela versão TMR;
- inclui simultaneamente o núcleo original `mvu.sv` e o wrapper `mvu_tmr.sv`;
- rejeita `TH > 1`, que ainda não foi implementado para esta versão TMR.

### `src/finn/custom_op/fpgadataflow/rtl/__init__.py`

A nova classe foi importada e registrada:

```python
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_tmr_rtl import MVAU_TMR_rtl

custom_op["MVAU_TMR_rtl"] = MVAU_TMR_rtl
```

Isso permite que `getCustomOp()` reconheça nós ONNX com tipo `MVAU_TMR_rtl`.

### `src/finn/custom_op/fpgadataflow/hwcustomop.py`

`MVAU_TMR_rtl` foi adicionada às listas usadas por:

- `generate_hdl_memstream()`;
- `generate_hdl_fetch_weights()`.

Assim, a nova camada reutiliza a infraestrutura de armazenamento e fornecimento de pesos da MVAU original.

## Alterações no projeto TinyMLP

### `03_finn/convert_to_hardware.py`

Depois de `SpecializeLayers`, os nós originalmente escolhidos como `MVAU_rtl` são convertidos explicitamente para:

```python
node.op_type = "MVAU_TMR_rtl"
```

O resultado é salvo separadamente em:

```text
03_finn/model_hw_specialized_tmr.onnx
```

A `MVAU_hls` da primeira camada permanece normal; somente a MVAU RTL recebe TMR.

### `03_finn/set_folding.py`

A entrada foi alterada para:

```text
model_hw_specialized_tmr.onnx
```

O modelo configurado é salvo como:

```text
model_folded_tmr.onnx
```

O folding atual permanece:

```text
MVAU_hls_0:     SIMD=4, PE=2
MVAU_TMR_rtl_0: SIMD=1, PE=1
```

O nome do nó mudou para `MVAU_TMR_rtl_0` durante a criação da partição.

### `04_build/create_partition.py`

O script agora lê:

```text
03_finn/model_folded_tmr.onnx
```

e usa saídas separadas:

```text
04_build/model_before_partition_tmr.onnx
04_build/model_parent_tmr.onnx
04_build/partitions_tmr/partition_0.onnx
```

Isso impede que o fluxo TMR sobrescreva ou reutilize acidentalmente os modelos normais.

### `04_build/generate_ip.py`

A entrada foi alterada para:

```text
04_build/partitions_tmr/partition_0.onnx
```

## Estado verificado

As etapas abaixo foram executadas com sucesso:

```bash
cd /TinyMLP_FINN/03_finn
python convert_to_hardware.py
python set_folding.py

cd /TinyMLP_FINN/04_build
python create_partition.py
```

A partição resultante contém:

```text
Thresholding_rtl
MVAU_hls
MVAU_TMR_rtl
```

A geração do IP ainda está pendente:

```bash
cd /TinyMLP_FINN/04_build
python generate_ip.py
```

Depois dela, deve-se confirmar nos arquivos e relatórios gerados a presença de `replica0`, `replica1` e `replica2`.

## Limitações atuais

- Apenas a segunda MVAU, implementada em RTL, está protegida.
- A `MVAU_hls` e os demais blocos continuam sem TMR.
- Entradas, pesos, clock, reset e o votador são pontos comuns de falha.
- `fault_detected` ainda não está exposto por AXI/GPIO.
- Não há suporte TMR para `TH > 1`.
- Ainda faltam simulação com injeção de falhas, síntese e confirmação de que o Vivado preservou fisicamente as três réplicas.
- TMR lógico sozinho não torna a ZCU104 qualificada para uso espacial.

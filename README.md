# TinyMLP FINN

Implementação de uma rede neural quantizada pequena, treinada com PyTorch/Brevitas e convertida pelo FINN em um acelerador FPGA para a placa AMD-Xilinx ZCU104.

O exemplo resolve um problema sintético: recebe 16 bits e prevê a classe `1` quando há mais de oito valores iguais a `1`; caso contrário, prevê a classe `0`.

## Visão geral

O projeto percorre quatro etapas:

```text
PyTorch + Brevitas      QONNX                 FINN-ONNX              FPGA ZCU104
01_train/          ->   02_qonnx/        ->   03_finn/         ->   04_build/
treino quantizado       exportação/limpeza    dataflow/folding       IP, bitstream e driver
```

O modelo tem 154 parâmetros treináveis e a seguinte arquitetura:

```text
entrada (16 valores)
  -> QuantIdentity, ativação de 4 bits
  -> QuantLinear 16 x 8, pesos de 4 bits, com bias
  -> QuantReLU, ativação de 4 bits
  -> QuantLinear 8 x 2, pesos de 4 bits, com bias
  -> dois scores de classe
```

A classe prevista pelo modelo de software é o índice do maior dos dois scores (`argmax`).

## Pré-requisitos

- Linux com Docker;
- repositório FINN na pasta irmã `../finn`;
- Vivado/Vitis acessíveis ao contêiner para síntese e geração do bitstream;
- placa ZCU104 com uma imagem PYNQ compatível para executar o driver;
- espaço em disco e tempo para as etapas de HLS, síntese e place-and-route.

O repositório FINN presente ao lado deste projeto está identificado como `v1.0.0-alpha`. Como o FINN possui uma cadeia extensa de dependências, as etapas de compilação devem ser executadas no contêiner fornecido por ele.

## Entrando no contêiner FINN

No host:

```bash
cd ~/Documents/Laboratorio_LEDS/Estudo_FINN/finn

FINN_DOCKER_EXTRA="-v /home/thiago/Documents/Laboratorio_LEDS/Estudo_FINN/TinyMLP_FINN:/TinyMLP_FINN" ./run-docker.sh
```

Esse comando monta o projeto no caminho `/TinyMLP_FINN` dentro do contêiner. Se o projeto estiver em outro lugar, ajuste apenas o caminho à esquerda de `:`.

Para as etapas que usam Vivado, o `run-docker.sh` também espera que `FINN_XILINX_PATH` e `FINN_XILINX_VERSION` estejam configurados no host de acordo com a instalação local. Sem essas variáveis ainda é possível executar transformações de software, mas não gerar os IPs e o bitstream.

Depois que o shell do contêiner abrir:

```bash
cd /TinyMLP_FINN
```

## Execução completa

Os comandos abaixo respeitam os diretórios de trabalho esperados pelos scripts. Os arquivos `.onnx`, relatórios e produtos de deploy já presentes são resultados de uma execução anterior e serão sobrescritos ao repetir o fluxo.

### 1. Treinar a rede quantizada

```bash
cd /TinyMLP_FINN/01_train
python train.py
```

`train.py` cria 5.000 amostras binárias pseudoaleatórias com semente 42, usa 80% para treino e 20% para teste, treina por 100 épocas com Adam (`lr=0.01`) e `CrossEntropyLoss`, imprime a acurácia a cada dez épocas e grava `tiny_mlp.pth`.

Como a divisão é feita depois de gerar o conjunto com uma semente fixa, a execução é reproduzível. O treinamento usa o lote de treino inteiro em cada época.

### 2. Exportar e limpar o QONNX

```bash
cd /TinyMLP_FINN
python 02_qonnx/export_qonnx.py
python 02_qonnx/inspect_qonnx.py
python 02_qonnx/cleanup_qonnx.py
```

- `export_qonnx.py` recria `TinyMLP`, carrega `tiny_mlp.pth` e exporta `model.onnx` com uma entrada de exemplo de formato `[1, 16]`;
- `inspect_qonnx.py` lista os nós do grafo exportado;
- `cleanup_qonnx.py` aplica a limpeza do QONNX e cria `model_clean.onnx` com apenas uma entrada pública (`global_in`).

Copie o modelo limpo para a etapa FINN:

```bash
cp 02_qonnx/model_clean.onnx 03_finn/model_clean.onnx
```

### 3. Converter para FINN e preparar as camadas de hardware

Os scripts desta etapa usam nomes relativos e devem ser executados dentro de `03_finn`:

```bash
cd /TinyMLP_FINN/03_finn
python convert_to_finn.py
python verify_conversion.py
python streamline_model.py
python convert_to_hardware.py
python inspect_hardware.py
python set_folding.py
```

O que cada comando faz:

1. `convert_to_finn.py` converte os operadores quantizados do QONNX em operadores FINN-ONNX e cria `model_finn.onnx`.
2. `verify_conversion.py` executa QONNX e FINN-ONNX para a mesma entrada e verifica igualdade numérica com tolerância `1e-5`.
3. `streamline_model.py` infere formas, dobra constantes, normaliza nomes e aplica `Streamline`; gera `model_tidy.onnx` e `model_streamlined.onnx`, comparando as saídas antes e depois.
4. `convert_to_hardware.py` transforma as multiplicações de matriz quantizadas em unidades MVAU, especializa os nós para `xczu7ev-ffvc1156-2-e` e gera `model_hw.onnx` e `model_hw_specialized.onnx`.
5. `inspect_hardware.py` mostra dimensões, paralelismo, tipos de dados e demais atributos das MVAUs.
6. `set_folding.py` define o paralelismo e salva `model_folded.onnx`: `SIMD=4`, `PE=2` para `MVAU_hls_0`; `SIMD=1`, `PE=1` para `MVAU_rtl_0`.

O grafo simplificado tem duas MVAUs, uma camada de limiarização e um pós-processamento `Mul + Add`. Folding controla a reutilização dos recursos: mais `SIMD`/`PE` tende a aumentar desempenho e consumo de FPGA; menos paralelismo economiza recursos e aumenta o número de ciclos.

### 4. Analisar e particionar o acelerador

```bash
cd /TinyMLP_FINN
python 04_build/estimate.py
python 04_build/check_postprocess.py
python 04_build/check_integer_decision.py
python 04_build/create_partition.py
python 04_build/performance_partition.py
```

- `estimate.py` estima ciclos e recursos para a ZCU104 antes da síntese;
- `check_postprocess.py` testa em 1.000 entradas se o `Mul + Add` final altera o `argmax`;
- `check_integer_decision.py` testa em 10.000 entradas a regra inteira usada para interpretar a saída crua do FPGA;
- `create_partition.py` converte o `MultiThreshold` restante em hardware e separa os nós dataflow em `partitions/partition_0.onnx`; o modelo pai mantém `Mul + Add` fora do acelerador;
- `performance_partition.py` estima gargalo, throughput e latência da partição para clock de 100 MHz (`10 ns`).

A regra inteira validada pelo projeto é:

```python
classe = 0 if (r0 - r1) >= -20 else 1
```

Ela incorpora o efeito decisório do pós-processamento que ficou fora da partição. Deve ser revalidada se o modelo for treinado novamente, se a quantização mudar ou se o grafo for modificado.

## Builds disponíveis

Há dois caminhos complementares depois da criação da partição.

### Síntese out-of-context do acelerador

Use este caminho para gerar os IPs, montar o bloco dataflow e obter relatórios de utilização, timing e potência sem construir o sistema Zynq completo:

```bash
cd /TinyMLP_FINN
python 04_build/generate_ip.py
python 04_build/prepare_stitch.py
python 04_build/stitch_and_synth.py
```

- `generate_ip.py`: gera código e sintetiza os nós HLS da partição;
- `prepare_stitch.py`: insere conversores de largura AXI Stream e FIFOs rasas, especializa e sintetiza os novos nós;
- `stitch_and_synth.py`: conecta os IPs, executa síntese e place-and-route OOC e copia os relatórios para `04_build/ooc_reports/`.

Resultados existentes, obtidos com Vivado 2024.2, alvo `xczu7ev-ffvc1156-2-e` e período solicitado de 10 ns:

| Recurso/métrica | Resultado | Disponível na FPGA | Uso |
|---|---:|---:|---:|
| LUT | 4.271 | 230.400 | 1,85% |
| Flip-flops | 4.945 | 460.800 | 1,07% |
| DSP48E2 | 17 | 1.728 | 0,98% |
| BRAM 36K / 18K | 0 / 0 | — | 0% |
| URAM | 0 | — | 0% |
| WNS | 5,218 ns | — | timing atendido |
| Fmax estimada | 209,12 MHz | — | — |
| Potência total estimada | 0,615 W | — | — |

Esses números descrevem o IP OOC, não necessariamente o sistema completo na placa. Os arquivos-fonte da medição são `ooc_results.json`, `ooc_utilization.rpt`, `ooc_timing.rpt`, `ooc_power.rpt` e `ooc_metadata.txt`.

### Bitstream completo para a ZCU104

Use este caminho para construir o projeto Zynq, copiar o bitstream e gerar o driver PYNQ:

```bash
cd /TinyMLP_FINN
python 04_build/build_zcu104.py
python 04_build/generate_driver.py
```

`build_zcu104.py` chama `ZynqBuild` para a plataforma `ZCU104` a 100 MHz, salva o modelo com metadados de hardware e copia os produtos temporários do FINN para uma pasta persistente. `generate_driver.py` cria o driver `zynq-iodma` e o associa ao modelo final.

O pacote pronto está em `04_build/zcu104_deploy.zip` e contém:

- `tiny_mlp.bit`: bitstream da FPGA;
- `tiny_mlp.hwh`: descrição do hardware usada pelo PYNQ;
- `tiny_mlp_zcu104.onnx`: modelo gerado pelo build Zynq;
- `tiny_mlp_zcu104_driver.onnx`: modelo com metadados do driver;
- `driver/`: driver Python e utilitários mínimos do FINN/QONNX.

## Executar na ZCU104

Copie e descompacte `04_build/zcu104_deploy.zip` na placa. O arquivo `.hwh` deve ficar ao lado do `.bit` e ter o mesmo nome-base, como já ocorre com `tiny_mlp.bit` e `tiny_mlp.hwh`.

Crie uma entrada NumPy. Cada linha representa uma amostra com 16 valores binários:

```bash
cd zcu104_deploy/driver
python - <<'PY'
import numpy as np

x = np.array([[0, 1, 1, 0, 1, 0, 1, 1,
               0, 1, 0, 1, 1, 0, 0, 1]], dtype=np.float32)
np.save("input.npy", x)
PY
```

Execute o acelerador:

```bash
python driver.py \
  --exec_mode execute \
  --platform zynq-iodma \
  --batchsize 1 \
  --bitfile ../tiny_mlp.bit \
  --inputfile input.npy \
  --outputfile output.npy
```

Leia os dois inteiros produzidos e aplique a decisão validada no projeto:

```bash
python - <<'PY'
import numpy as np

r0, r1 = np.load("output.npy")[0]
print("saída crua:", r0, r1)
print("classe:", 0 if int(round(r0)) - int(round(r1)) >= -20 else 1)
PY
```

Para medir o caminho de execução do acelerador com o driver:

```bash
python driver.py \
  --exec_mode throughput_test \
  --platform zynq-iodma \
  --batchsize 1000 \
  --bitfile ../tiny_mlp.bit
```

As métricas são gravadas em `nw_metrics.txt`. O `validate.py` gerado automaticamente pelo FINN foi feito para MNIST/CIFAR-10 e não representa o conjunto sintético deste projeto; use entradas `.npy` com `driver.py` ou escreva uma validação específica para a regra dos 16 bits.

## Estrutura dos arquivos

```text
TinyMLP_FINN/
├── 01_train/
│   ├── model.py                 # definição da TinyMLP quantizada
│   ├── train.py                 # geração dos dados, treino, teste e checkpoint
│   └── tiny_mlp.pth             # pesos treinados
├── 02_qonnx/
│   ├── export_qonnx.py          # exportação Brevitas -> QONNX
│   ├── inspect_qonnx.py         # inspeção simples dos nós
│   ├── cleanup_qonnx.py         # normalização/limpeza do grafo
│   ├── model.onnx               # exportação original
│   ├── model.onnx.data          # dados externos da exportação
│   └── model_clean.onnx         # QONNX limpo
├── 03_finn/
│   ├── convert_to_finn.py       # QONNX -> FINN-ONNX
│   ├── verify_conversion.py     # equivalência QONNX/FINN
│   ├── streamline_model.py      # tidy-up e streamline
│   ├── convert_to_hardware.py   # inferência e especialização das MVAUs
│   ├── inspect_hardware.py      # inspeção dos atributos de hardware
│   ├── set_folding.py           # definição de SIMD e PE
│   └── model_*.onnx             # modelos intermediários da etapa
└── 04_build/
    ├── estimate.py              # estimativas de ciclos e recursos
    ├── check_postprocess.py     # efeito do pós-processamento no argmax
    ├── check_integer_decision.py# validação da decisão sobre INT32
    ├── create_partition.py      # criação da partição dataflow
    ├── performance_partition.py # desempenho teórico da partição
    ├── generate_ip.py           # geração e síntese HLS dos IPs
    ├── prepare_stitch.py        # DWC, FIFOs e preparação para conexão
    ├── stitch_and_synth.py      # IP conectado e síntese OOC
    ├── build_zcu104.py          # projeto Zynq e bitstream
    ├── generate_driver.py       # driver PYNQ
    ├── partitions/              # partição dataflow original
    ├── zcu104_partitions/       # partições intermediárias do ZynqBuild
    ├── ooc_reports/             # resultados reais do build OOC
    └── zcu104_deploy.zip        # pacote de deploy da placa
```

## Relação entre os modelos ONNX

| Arquivo | Função |
|---|---|
| `02_qonnx/model.onnx` | exportação direta do Brevitas; usa dados externos |
| `02_qonnx/model_clean.onnx` | grafo QONNX normalizado |
| `03_finn/model_finn.onnx` | operadores quantizados convertidos para FINN |
| `03_finn/model_tidy.onnx` | nomes, formas e constantes normalizados |
| `03_finn/model_streamlined.onnx` | aritmética simplificada pelo FINN |
| `03_finn/model_hw.onnx` | multiplicações convertidas em MVAUs genéricas |
| `03_finn/model_hw_specialized.onnx` | MVAUs especializadas para o dispositivo alvo |
| `03_finn/model_folded.onnx` | paralelismo SIMD/PE configurado |
| `04_build/model_before_partition.onnx` | todas as camadas aceleráveis convertidas para hardware |
| `04_build/model_parent.onnx` | partição dataflow encapsulada; pós-processamento fora dela |
| `04_build/partition_ipgen.onnx` | nós da partição com IP gerado |
| `04_build/partition_ready_stitch.onnx` | partição com FIFOs e conversores AXI Stream |
| `04_build/partition_stitched.onnx` | partição conectada e sintetizada OOC |
| `04_build/zcu104_deploy/tiny_mlp_zcu104.onnx` | resultado do build completo para a ZCU104 |

## Observações e limitações

- A entrada aceita pelo modelo tem forma fixa `[1, 16]` durante a exportação; o driver adapta o batch para a execução na placa.
- Os scripts não formam um único comando idempotente: cada etapa pressupõe que a anterior gerou o arquivo esperado.
- Alguns scripts usam caminhos relativos ao diretório atual; siga os `cd` mostrados neste README.
- `model.onnx` depende de `model.onnx.data`; mantenha os dois juntos. O modelo limpo é autocontido.
- Os números de desempenho de `performance_partition.py` são estimativas. Meça `throughput_test` na placa para obter desempenho do sistema real.
- Retreinar altera os pesos e pode invalidar modelos ONNX, IPs, bitstream, relatórios e a regra inteira do pós-processamento; nesse caso, regenere o pipeline a partir da etapa 2.
- A pasta `.venv` local não é necessária para o fluxo FINN via Docker e não faz parte do pacote de deploy.
# TinyMLP-on-FINN
# TinyMLP-on-FINN

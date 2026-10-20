# llama-hive (experimental)

Este protótipo trata cada trabalhador como uma sequência lógica (`seq_id`) dentro de um único `llama_context`. Ele não cria 200 contextos físicos e não reduz arbitrariamente a dimensão interna do transformer.

Variantes experimentais:

- `llama-hive-v1`: baseline com batch completo; use `-ctk q8_0 -ctv q8_0` para quantizar o KV.
- `llama-hive-v2`: ordena cada microbatch por um sketch auxiliar de 128 dimensões, usando três âncoras e distância euclidiana. É um scheduler; não é o estado oculto real do Gemma e não remove tokens da atenção.
- `llama-hive-v2.5`: pontua blocos de 16 tokens por similaridade com o sketch de 128 dimensões, preserva 8 blocos por trabalhador e remove os blocos antigos de menor pontuação via `llama_memory_seq_rm`.
- `llama-hive-v3`: executa um token por worker por tick, usa tarefas `planner -> critic -> synthesizer` como handoff de metadados e reserva o último worker como sintetizador final.

A estratégia é:

- carregar um único modelo, compartilhado e somente leitura;
- criar até 200 lanes lógicas no KV cache;
- avaliar o prompt em batch, com o mesmo prefixo associado às lanes;
- fazer continuous batching: um token por trabalhador em cada passo;
- manter um sampler independente por trabalhador;
- usar `n_ubatch` para controlar a granularidade física do cálculo.

## Compilar

```bash
cmake --build build --target llama-hive -j2
cmake --build build --target llama-hive-v1 llama-hive-v2 -j2
cmake --build build --target llama-hive-v2.5 -j2
cmake --build build --target llama-hive-v3 -j2
```

## Executar

```bash
./build/bin/llama-hive \
  -m build/tests/test-models/llama-dense.gguf \
  -p "Hello" -n 8 -np 200 -c 4096 -b 512 -ub 64 -t 1
```

Teste v1 com KV quantizado:

```bash
./build/bin/llama-hive-v1 \
  -m models/gemma-3-1b-it/gemma-3-1b-it-q4_k_m.gguf \
  -f models/gemma-3-1b-it/benchmark-200.txt -n 1 -np 200 \
  -c 512 -b 256 -ub 32 -t 1 -tb 2 -ctk q8_0 -ctv q8_0
```

Teste v2 com roteamento geométrico:

```bash
./build/bin/llama-hive-v2 \
  -m models/gemma-3-1b-it/gemma-3-1b-it-q4_k_m.gguf \
  -f models/gemma-3-1b-it/benchmark-200.txt -n 1 -np 200 \
  -c 512 -b 256 -ub 32 -t 1 -tb 2 -ctk q8_0 -ctv q8_0
```

Teste v2.5 com seleção de blocos KV:

```bash
./build/bin/llama-hive-v2.5 \
  -m models/gemma-3-1b-it/gemma-3-1b-it-q4_k_m.gguf \
  -f models/gemma-3-1b-it/benchmark-200.txt -n 16 -np 200 \
  -c 4096 -b 512 -ub 64 -t 1 -tb 2 -ctk q8_0 -ctv q8_0
```

A v2.5 mantém os 2 blocos mais recentes e escolhe 6 blocos antigos por similaridade geométrica. No benchmark de 200 workers e 16 tokens por worker, removeu 1.200 blocos KV.

Compare `-np 1`, `-np 8`, `-np 32`, `-np 128` e `-np 200`. O ganho esperado aparece quando há muitos workers ativos e o custo de prompt/KV é amortizado; não é garantido que 200 workers reduzam a latência de um único token.

## Sobre a dimensão 128

A dimensão oculta/head dimension pertence aos pesos do modelo. Cortá-la no runtime para 128 altera a função do modelo e, em geral, quebra as formas dos tensores. Para testar essa hipótese seria necessário treinar ou adaptar um modelo com projeções `d_model -> 128 -> d_model`, ou usar um modelo pequeno já treinado. O protótipo mantém a matemática original para permitir uma comparação válida.

Na v2, o vetor de 128 dimensões é apenas um metadado de roteamento calculado no host. Uma seleção hierárquica real exigiria estados intermediários do modelo, que não são expostos pela API pública atual; por isso a v2 não promete ganho semântico nem remove workers da atenção.

Na v2.5, `llama_memory_seq_rm` remove células lógicas do KV, mas a capacidade física do KV cache continua sendo a definida na criação do `llama_context`. Por isso a primeira implementação não reduz automaticamente o RSS nem a VRAM alocada. Para obter essa redução é necessário alterar a alocação do cache ou implementar um grafo de atenção esparsa que não leia os blocos removidos.

### v3: pipeline de tarefas e um token por tick

A v3 usa um token por trabalhador em cada rodada. Cada worker envia um handoff compacto de 128 dimensões ao próximo worker, alternando os papéis `planner`, `critic` e `synthesizer`. O worker final (`n_workers - 1`) coleta os tokens que ele próprio gerou e imprime uma frase final normalizada com pontuação.

O handoff atual é metadado de scheduling. Ele não insere automaticamente o texto produzido por um worker no prompt do seguinte. Para fazer isso, seria necessário um segundo estágio de prefill, com custo adicional e uma política explícita de truncamento. A v3 mantém essa separação para medir o custo do pipeline sem misturar comunicação textual com o decode.

Benchmark de tempo com a pergunta sobre a capital do Brasil, `n_predict=8`, KV `Q8_0/Q8_0`, `n_batch=256`, `n_ubatch=32`, 1 thread de geração e 2 threads de batch:

| Workers | Tokens gerados | Tempo | Vazão agregada |
|---:|---:|---:|---:|
| 1 | 8 | 1,533 s | 5,22 tok/s |
| 8 | 60 | 5,758 s | 10,42 tok/s |
| 32 | 205 | 19,514 s | 10,51 tok/s |
| 200 | 1.264 | 120,037 s | 10,53 tok/s |

Alguns workers terminaram antes dos 8 tokens ao emitirem um token EOG; por isso o total observado é menor que `workers × 8`. O teste mede tempo e vazão do scheduler, não a qualidade da generalização. A v3 ainda não faz síntese textual entre workers: o handoff troca metadados, e não o texto completo.

O valor 128 é um orçamento lógico por worker, não `n_ctx=128` total. Com 200 workers e 32 tokens por worker, o contexto físico precisa acomodar aproximadamente 6.400 posições, ou o cache deve usar eviction/compactação. Configurar `-c 128` não força um contexto independente de 128 tokens por lane.

Execute a v3 com:

```bash
./build/bin/llama-hive-v3 \
  -m models/gemma-3-1b-it/gemma-3-1b-it-q4_k_m.gguf \
  -p "The key to efficient parallel inference is" \
  -n 32 -np 200 -c 256 -b 256 -ub 32 -t 1 \
  -ctk q8_0 -ctv q8_0

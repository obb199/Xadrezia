<img src="https://github.com/user-attachments/assets/cbd122da-121b-4a79-96dc-478cd54241bb" alt="Logo" width="1024" heigth="256">

# Xadrezia chess engine

Xadrezia é um projeto experimental para desenvolver uma engine de xadrez baseada exclusivamente em reconhecimento de padrões em jogos dos melhores enxadristas do mundo. Ao contrário de engines tradicionais que utilizam busca em árvore (como Minimax ou Monte Carlo Tree Search ou técnicas baseadas em força bruta), Xadrezia emprega redes neurais para avaliar posições e selecionar movimentos, explorando o potencial do aprendizado de máquina no xadrez. O projeto está em fase inicial e é voltado para pesquisa, experimentação, diversão e curiosidade!

## Estrutura do Repositório

O repositório contém scripts em Python que implementam os componentes principais da engine, incluindo manipulação de dados, modelo de rede neural, geração de movimentos e inferência. Abaixo está uma descrição dos arquivos principais:

**data_loader.py:** Carrega e pré-processa dados de treinamento, convertendo posições de xadrez em tensores 8x8x7 que representam o tabuleiro para alimentar a rede neural.

**move_dict.py:** Define dicionários de mapeamento para movimentos de xadrez, convertendo notações em índices para o vetor de saída da rede neural.

**generate_training_data.py:** Gera dados de treinamento a partir de partidas de xadrez, criando pares de entrada-saída (tensores de tabuleiro e vetores de movimento).

**matches:** Contém scripts para simular jogos, usados para testar a engine ou gerar dados adicionais.

**pieces.py:** Define as peças de xadrez e suas propriedades (ex: movimentos válidos, tipos), servindo como base para a representação do tabuleiro.

**table.py:** Implementa a representação do tabuleiro de xadrez, incluindo métodos para gerar o tensor 8x8x7 com informações sobre todas as peças.

**training.py:** Loop de treino do modelo.

**inference.py:** Realiza inferência com o modelo treinado, recebendo um tensor de tabuleiro e produzindo um vetor de probabilidades para selecionar movimentos.

**model.py:** Define a arquitetura da rede neural, composta por uma convolução 2D inicial, convoluções residuais 2D e um Transformer encoder-only.

**utils.py:** Contém funções utilitárias, compartilhadas entre os outros scripts.

## Arquitetura da Rede Neural

A rede neural da Xadrezia é projetada para processar representações do tabuleiro e prever movimentos. Abaixo estão os detalhes do input, output e da arquitetura:

### Input:

O input da rede é um tensor 8x8x7, que representa o estado do tabuleiro de xadrez:

**Dimensões 8x8:** Correspondem às 64 casas do tabuleiro (8 linhas x 8 colunas).

**7 canais:** Cada canal representa um tipo de peça ou informação específica:

**6 canais para as peças:** (peão, cavalo, bispo, torre, dama, rei) de cada cor (brancas e pretas, totalizando 12 tipos de peças).

**canal adicional:** para informar a cor da peça.

Cada elemento do tensor indica a presença (ou ausência) de uma peça em uma casa específica.

### Output

**O output da rede é um vetor de probabilidades que codifica:**

O movimento selecionado - coluna e linha de onde está a peça e coluna e linha para onde vai (ex: "e2e3", "d1f4", "g6h7").

O vetor contém probabilidades para todos os movimentos possíveis, mapeados via move_dict.py.

Durante a inferência, o movimento válido de maior probabilidade é selecionado.

### Arquitetura

A rede neural combina algumas técnicas comuns no aprendizado profundo para processar o tensor de entrada e produzir o vetor de saída.

### Convolução 2D Inicial:

Aplica uma camada convolucional 2D ao tensor 8x8x7 para extrair características espaciais do tabuleiro (ex: padrões de peças, controle de casas).

Usa filtros (4x4 e 2x2) para capturar relações locais entre casas adjacentes.


### Convoluções Residuais 2D:

Uma sequência de blocos residuais (ResNet-style) refina as características extraídas.

Cada bloco contém camadas convolucionais com conexões de salto (skip connections) para evitar gradientes desaparecendo e melhorar o treinamento.

Preserva a dimensionalidade espacial (8x8) enquanto aumenta a profundidade das características na última dimensão.


### Transformer Encoder-Only:

Após as convoluções, as características extraídas alimentam um Transformer encoder-only.

O Transformer modela relações de longo alcance entre casas e peças, capturando dependências globais no tabuleiro (ex: coordenação entre peças distantes).

Usa atenção multi-cabeça (multi-head attention) para priorizar informações relevantes.

A saída do Transformer é processada por camadas densas para produzir o vetor de probabilidades, que é decodificando em um lance.

            ┌────────────────────────────┐
            │       Input (8x8x7)        │ ◄── Board representation
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │    Convolutional Layer     │
            │----------------------------│
            │      Conv2D (32, 5x5)      │ ◄── Initial feature extraction
            │ BatchNormalizaztion + ReLU │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐             
            │        Inception-SE        │
            │    Residual Convolution    │ ◄── Parallel convolution with different
            │   (4 parallel conv paths)  │       kernels followed by squeeze and
            └────────────┬───────────────┘         excitation and residual sum.
                         │
                         ▼
            ┌────────────────────────────┐
            │   Positional Encoding      │
            │----------------------------│
            │     2D PosEnc (8x8x256)    │ ◄── Adds spatial context
            │   Added to conv features   │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │    Encoder Stack [x32]     │
            │----------------------------│ 
            │     MultiHeadAttention     │
            │    LayerNorm + Residual    │  ◄── Self-attention with 16 heads
            │    FFN (ReLU + Dropout)    │
            │    LayerNorm + Residual    │
            │     Residual addition      │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │      Weighted Average      │
            │----------------------------│◄── Aggregates features with trainable
            │     Product by weights     │   weighted average for filters division.
            │      GlobalAvgPool2D       │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │         Dense Head         │
            │----------------------------│ ◄── Output: move probabilities
            │   Dense (4098, softmax)    │
            └────────────────────────────┘

![engine vs human](https://sdmntprwestus.oaiusercontent.com/files/00000000-85d8-6230-877d-bfa43a12b753/raw?se=2025-05-12T02%3A21%3A05Z&sp=r&sv=2024-08-04&sr=b&scid=00000000-0000-0000-0000-000000000000&skoid=ea1de0bc-0467-43d6-873a-9a5cf0a9f835&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-05-12T01%3A10%3A54Z&ske=2025-05-13T01%3A10%3A54Z&sks=b&skv=2024-08-04&sig=flAFi6lrlGvhgUWgj/Z128pM7AiqIO0lRa6C7AtQWe0%3D)


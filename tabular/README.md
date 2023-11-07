# Reinforcement Learning - Modelos Tabulares

Os modelos tabulares de RL utilizam tabelas para armazenar os valores que indicam qual ação o agente deve tomar em cada um dos estados.

Essas tabelas são uma representação de todos os estados possíveis cruzados com todas as ações possíveis. Geralmente, elas são inicializadas com todas as entradas zeradas e, à medida que o agente interage com o ambiente, os valores são atualizados para refletir as melhores decisões a serem tomadas.

## Modelos

Existem vários modelos tabulares para RL, cada um com suas particularidades. No entanto, optamos por utilizar dois deles: o Q-Learning e o SARSA. Embora esses modelos sejam muito semelhantes, uma pequena diferença entre eles pode afetar significativamente os resultados.

Ambos os modelos determinam a melhor ação para um estado específico procurando o valor máximo na tabela. Além disso, eles incorporam uma taxa de exploração (epsilon) que permite que o agente siga caminhos aleatórios de vez em quando, a fim de evitar ficar preso em máximos locais. No contexto do jogo da "Cobrinha", a exploração desempenha um papel fundamental na prevenção de loops infinitos.

Apesar das semelhanças, a principal diferença entre esses modelos reside na forma como eles atualizam os valores da tabela. Ambos utilizam a equação de Bellman para atualizar os valores, mas o Q-Learning calcula o máximo valor futuro, enquanto o SARSA considera a próxima ação antes de atualizar o valor do estado atual. Aqui estão as equações de Bellman para cada modelo:

#### Q-Learning
$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)]$$

#### SARSA (State, Action, Reward, State, Action)
$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$$

## Ações

Um dos primeiros passos a ser tomado é a definição do espaço de ações que o agente pode tomar. Como estamos utilizando o jogo da "Cobrinha", parece intuitivo que são 4 movimentos ($\leftarrow,\uparrow,\rightarrow,\downarrow$). Poém essa conclusão é errada, quando um a Cobra faz um movimento de subir $\uparrow$, ela não pode descer $\downarrow$ imediatamente, portanto resta apenas 3 movientos($\leftarrow,\uparrow,\rightarrow$), e o fato de não poder ir pela direção oposta a atual sempre acontece, portanto foi preferível utilizar 3 ações, sendo elas relativas a cabeça, as ações possíveis são seguir em frente( $\uparrow$ ), virar a direita($\rightarrow$) e virar a esquerda($\leftarrow$).

## Estados
## Execução

Para executar o modelo é necessário ir ao Diretório

`\Snake-RL\tradicional`


STATETAB - originalmente foi treinado com pesos:
DIED = -100
DIEDTIME = -10
ATE = 50
DEFAULT = 0

####Ele morre algumas vezes, várias vezes pelo tempo.

O novo modelo, eu tirei a taxa epsilon e troquei os pesos para
morre = -10
come = 1
### Ver qual dos dois fez a melhoria substancial.

TODOS DE SARSA FORAM TREINADOS COM epsilon = 0, morre = -10, come = 1 e tempo acaba = -20.
Tabular, não sei ao certo, só está lá. 


& C:/Users/carlo/anaconda3/envs/RL/python.exe c:/Users/carlo/Desktop/RL/Snake-RL/tabular/sarsa/run_sarsa.py

& C:/Users/carlo/anaconda3/envs/RL/python.exe c:/Users/carlo/Desktop/RL/Snake-RL/tabular/qlearning/run_q.py
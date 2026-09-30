# Easy-Transformer
트랜스포머 구조의 이해

## 트랜스포머 개념 이해 자료
- []()

## 트랜스포머 내부 구조 이해 자료
- [처음부터 자세히 알아보기(medium.com)](https://medium.com/@abineshsiva1998/transformers-explained-from-scratch-the-architecture-behind-chatgpt-google-translate-and-almost-3ac4a10f683e)
  - 초보자도 쉽게 이해할 수 있도록 수학 공식이 포함된 안내서
  - 손으로 직접 따라 할 수 있는 계산 예제 하나가 수록

- [트랜스포머의 작동 원리: 트랜스포머 구조에 대한 상세한 탐구(datacamp](https://www.datacamp.com/tutorial/how-transformers-work)
  - 내부 구조와 계산 방법을 그림으로 설명
  - 일부 잘못된 그림이 있음

- [그림으로 보는 트랜스포머(제이 알라마르](https://jalammar.github.io/illustrated-transformer/))
  - 내부 구조와 계산 방법을 그림으로 설명

- [TRANSFORMER EXPLAINER(GPT-2(samll) 인공신경망 과정 시각화)](https://poloclub.github.io/transformer-explainer/)
  > | 항목 | 원 논문 | GPT-2 |
  > |---|---|---|
  > | 층 정규화 위치 | 더한 뒤 (Post-LN) | 들어가기 전 (Pre-LN) |
  > | 마지막 층 정규화 | 없음 | 있음 |
  > | 위치 정보 | 사인·코사인으로 계산 | 학습으로 얻음 |
  > | 구조 | 인코더 + 디코더 | 디코더만 |
  
- 참고자료
  > | 단계 | 계산 | 결과 크기(예시) |
  > |---|---|---|
  > | ① Q, K, V 생성 | **X**에 가중치 행렬을 곱함 | 8×4 |
  > | ② 어텐션 점수 | **Q와 K를 내적한 뒤 √dₖ로 나눔** | 8×8 |
  > | ③ 어텐션 가중치 | 점수에 **소프트맥스**를 적용 (행마다 합이 1) | 8×8 |
  > | ④ 출력 벡터 | 가중치로 **V를 가중합** | 8×4 |

## 동영상
- [LLM 혁명의 비밀 : 트랜스포머 & 어텐션 기술(25분)](https://www.youtube.com/watch?v=DFzbYu42Li4)
  - 최소의 수식으로 쉽게 트랜스포머 & 어텐션 기술 이해와 트랜스포머의 진화
  > - KV cache
  > - MHA(Multi-Head Attention) -> GOA(Grouped Query Attention) -> MAA(Multi-Query Attention)
  > - FlashAttention(테이터를 잘게 나누어 온칩에서 연산)
  > - SwiGLU(데이터 중요도에 따라 핵심 정보만 통과시기는 지능형 활성화함수)
- [Transformer Attention 구조 완벽 설명 | Q K V, Softmax, 행렬곱까지 쉽게 이해(26분)](https://www.youtube.com/watch?v=cjLCjRNgKTU)
  - 데이터베이스의 질의, 키, 값을 예로 트랜스포머 어텐션의 이해
  > - 파이썬의 Dict, List, Set의 이해, 트랜스포머 어텐션 설명에서 차수 이해 도움
  > - <img width="1481" height="911" alt="image" src="https://github.com/user-attachments/assets/eade66b9-5076-435d-820d-f75fdb0d3986" />


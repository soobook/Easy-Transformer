# Easy-Transformer
트랜스포머 구조의 이해

## 트랜스포머 개념 이해 자료
- [트랜스포머(transformer)모델 쉽게 이해하기](https://www.hanbit.co.kr/channel/view.html?cmscode=CMS3470707425)
- [트랜스포머 모델](https://huggingface.co/learn/llm-course/ko/chapter1/1)
- [트랜스포머 모델 기본 개념과 주요 구성 요소 정리](https://velog.io/@jayginwoolee/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8-%EB%AA%A8%EB%8D%B8-%EA%B8%B0%EB%B3%B8-%EA%B0%9C%EB%85%90%EA%B3%BC-%EC%A3%BC%EC%9A%94-%EA%B5%AC%EC%84%B1-%EC%9A%94%EC%86%8C-%EC%A0%95%EB%A6%AC)

## 트랜스포머 내부 구조 이해 자료
- [처음부터 자세히 알아보기(medium.com)](https://medium.com/@abineshsiva1998/transformers-explained-from-scratch-the-architecture-behind-chatgpt-google-translate-and-almost-3ac4a10f683e)
  - 초보자도 쉽게 이해할 수 있도록 수학 공식이 포함된 안내서
  - 손으로 직접 따라 할 수 있는 계산 예제 하나가 수록

- [트랜스포머의 작동 원리: 트랜스포머 구조에 대한 상세한 탐구(datacamp)](https://www.datacamp.com/tutorial/how-transformers-work)
  - 내부 구조와 계산 방법을 그림으로 설명
  - 일부 잘못된 그림이 있음
  
- [트랜스포머: 소개와 동작 원리(추천)](https://sjh9708.tistory.com/231)
  - 내부 구조와 계산 방법을 그림으로 설명
    
- [트랜스포머 파헤치기(추천)](https://www.blossominkyung.com/deeplearning/transfomer-last)

- [2. Transformer의 구조(추천)](https://brunch.co.kr/@aideveloper/112)
  - 새로운 그림으로 자세히 설명 

- [Transformer의 큰 그림 이해](https://medium.com/@hugmanskj/transformer%EC%9D%98-%ED%81%B0-%EA%B7%B8%EB%A6%BC-%EC%9D%B4%ED%95%B4-%EA%B8%B0%EC%88%A0%EC%A0%81-%EB%B3%B5%EC%9E%A1%ED%95%A8-%EC%97%86%EC%9D%B4-%ED%95%B5%EC%8B%AC-%EC%95%84%EC%9D%B4%EB%94%94%EC%96%B4-%ED%8C%8C%EC%95%85%ED%95%98%EA%B8%B0-5e182a40459d)

- [트랜스포머(Transformer) 간단히 이해하기(추천)](https://moondol-ai.tistory.com/460)

 
- [그림으로 보는 트랜스포머(제이 알라마르](https://jalammar.github.io/illustrated-transformer/))
  - 내부 구조와 계산 방법을 그림으로 설명

- [Transformer와 GPT를 비교해 봅시다(channel AI)](https://channelai.tistory.com/10)
  
- [TRANSFORMER EXPLAINER: GPT-2(samll) 인공신경망 과정 시각화](https://poloclub.github.io/transformer-explainer/)
  - 구조도
  > <img width="1577" height="447" alt="image" src="https://github.com/user-attachments/assets/e27bcb47-7451-4eaa-bc82-7fdac5333690" />

  - 트랜스포머 논문(2017)과 GPT-2의 차이
  > | 항목 | 원 논문 | GPT-2 |
  > |---|---|---|
  > | 층 정규화 위치 | 더한 뒤 (Post-LN) | 들어가기 전 (Pre-LN) |
  > | 마지막 층 정규화 | 없음 | 있음 |
  > | 위치 정보 | 사인·코사인으로 계산 | 학습으로 얻음 |
  > | 구조 | 인코더 + 디코더 | 디코더만 |
  
  - 1. 데이터 흐름에 따른 차수
  - ① 임베딩
  > | 단계 | 연산 | 결과 차수 |
  > |---|---|---:|
  > | 입력 | 토큰 ID | n |
  > | 토큰 임베딩 | 토큰 임베딩 행렬(50,257×768)에서 해당 행을 꺼냄 | n×768 |
  > | 위치 임베딩 | 위치 임베딩 행렬(1,024×768)에서 해당 행을 꺼냄 | n×768 |
  > | 임베딩 합 X | 두 행렬을 더함 | **n×768** |
  - ② 블록: 어텐션 부분 (블록마다 반복)
  > | 단계 | 연산 | 결과 차수 |
  > |---|---|---:|
  > | 층 정규화 1 | 토큰별로 정규화 | n×768 |
  > | Q·K·V 생성 | X × W^QKV(768×2304) | n×2304 → Q, K, V 각각 n×768 |
  > | 헤드 분할 | 768 = 12헤드 × 64 | 헤드마다 Q, K, V가 n×64 |
  > | 어텐션 점수 | Q·Kᵀ ÷ √64(=8), 마스크 적용 | 헤드마다 **n×n** |
  > | 소프트맥스 → A | 행 단위 확률 | 헤드마다 n×n |
  - ③ 블록: MLP 부분 (블록마다 반복)
  > | 단계 | 연산 | 결과 차수 |
  > |---|---|---:|
  > | 층 정규화 2 | 토큰별로 정규화 | n×768 |
  > | MLP 확장 | × W₁(768×3072), GELU | **n×3072** |
  > | MLP 축소 | × W₂(3072×768) | n×768 |
  > | 잔차 더하기 | 이전 결과 + MLP 출력 | **n×768 → 다음 블록 입력** |
  - ④ 출력
  > | 단계 | 연산 | 결과 차수 |
  > |---|---|---:|
  > | 최종 층 정규화 | 토큰별로 정규화 | n×768 |
  > | 로짓 | × 토큰 임베딩 행렬ᵀ(768×50,257), 가중치 공유 | n×50,257 |
  > | 다음 토큰 | 마지막 행만 소프트맥스 | **50,257개 확률** |

  - 2. 파라미터(학습되는 가중치) 개수
  > | 구성 요소 | 계산 | 개수 |
  > |---|---|---:|
  > | 토큰 임베딩 | 50,257×768 | 38,597,376 |
  > | 위치 임베딩 | 1,024×768 | 786,432 |
  > | 블록 1개: W^QKV + 편향 | 768×2304 + 2304 | 1,771,776 |
  > | 블록 1개: W^O + 편향 | 768×768 + 768 | 590,592 |
  > | 블록 1개: W₁ + 편향 | 768×3072 + 3072 | 2,362,368 |
  > | 블록 1개: W₂ + 편향 | 3072×768 + 768 | 2,360,064 |
  > | 블록 1개: 층 정규화 2개 | 2×(768+768) | 3,072 |
  > | 블록 1개 합계 |  | 7,087,872 |
  > | 블록 12개 | ×12 | 85,054,464 |
  > | 최종 층 정규화 | 768+768 | 1,536 |
  > | **총합** |  | **124,439,808 ≈ 124M** |  

- 참고자료
  > | 단계 | 계산 | 결과 크기(예시) |
  > |---|---|---|
  > | ① Q, K, V 생성 | **X**에 가중치 행렬을 곱함 | 8×4 |
  > | ② 어텐션 점수 | **Q와 K를 내적한 뒤 √dₖ로 나눔** | 8×8 |
  > | ③ 어텐션 가중치 | 점수에 **소프트맥스**를 적용 (행마다 합이 1) | 8×8 |
  > | ④ 출력 벡터 | 가중치로 **V를 가중합** | 8×4 |

## 트랜스포머 구현 코드 자료
- [LLM의 중추, 트랜스포머 아키텍처](https://hunseop2772.tistory.com/402)

## 동영상
- [트랜스포머 구조 소개(박해선)](https://www.youtube.com/watch?v=JZUm4IAVnK4)
  
- [트랜스포머, 스텝 바이 스텝(신박AI)](https://www.youtube.com/watch?v=p216tTVxues)
  - 20분 영상으로 계산 과정을 쉽게 설명
  
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

- [핵심 머신러닝 Transformer(고려대 김성범 교수)](https://www.youtube.com/watch?v=a_-YgMO0u0E)

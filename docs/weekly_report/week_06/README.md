# 6주차 주간 업무 일지 
## 시스템 설계를 위한 시스템 분석 ver.6
✒️ 7,4 hamming code 환경 기반 Sum Product Algorithm 구현 + LLR을 사용 
> ![image](https://user-images.githubusercontent.com/45085563/170321406-d0c67d55-3e08-4383-bf96-a25cced39570.png)
> LLR을 사용하여 7,4 hamming code 상황에서 sum product algorithm을 구현 
> 그러나 **parity check bit**가 부족하여 hamming code보다 낮은 확률을 보이고 있음 

-----
❗ 보안점 
> hamming code와 같은 간단한 parity check matrix의 경우는 연산이 간단하지만, LDPC에서 사용하는 코드의 경우 압도적으로 0으로 채워져있는 parity check bit가 많다. 따라서 [torch.sparse](https://pytorch.org/docs/stable/sparse.html)를 사용하여 코드를 바꾸는 것을 교수님께서 추천 

# 교수님과 주간 미팅 2022-04-08
> LLR을 사용한 LDPC
>![image](https://user-images.githubusercontent.com/45085563/170321218-5f8c7670-bc34-4835-b5bd-0f9c222a175b.png)
#  
![image](https://user-images.githubusercontent.com/63450024/170591320-e13385fe-0b58-492e-87c2-c65ce7d48b1c.png)

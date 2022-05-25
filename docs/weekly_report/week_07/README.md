# 7주차 주간 업무 일지 
## 시스템 설계를 위한 시스템 분석 ver.7
✒️ 6주차의 내용을 sparse matrix로 구현 진행 
> 
> +  
> 그러나 **parity check bit**가 부족하여 hamming code보다 낮은 확률을 보이고 있음 

-----
❗ 보안점 
> hamming code와 같은 간단한 parity check matrix의 경우는 연산이 간단하지만, LDPC에서 사용하는 코드의 경우 압도적으로 0으로 채워져있는 parity check bit가 많다. 따라서 [torch.sparse](https://pytorch.org/docs/stable/sparse.html)를 사용하여 코드를 바꾸는 것을 교수님께서 추천 

# 교수님과 주간 미팅 2022-04-01
> LLR을 사용한 LDPC
>![image](https://user-images.githubusercontent.com/45085563/170325472-936c161b-89b4-4386-84a6-1219f55db6bd.png)
# 일정표 
> ![image](https://user-images.githubusercontent.com/45085563/170325569-6727fd30-5d66-40d2-ac01-b83aaf4451ac.png)

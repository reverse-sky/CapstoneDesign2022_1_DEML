# 3주차 주간 업무 일지 
## 시스템 설계를 위한 시스템 분석 ver.3
### (7,4) Hamming code 구현 
> + 2주차에 진행한 (7,4)hamming code를 python으로 구현       
> + 팀원 각각 Hamming code 구현 완료 이해가 둘다 Hamming code에 대해선 이해가 되었음을 인지 

-----
## 구현상의 문제점
> for문을 사용해서 구현을 하다보니 code word의 갯수가 1e+6 의 경우 상당한 시간이 소요되어짐, matrix product를 이용한 효율적인 코딩 방식 추구 
> + G matrix를 사용해 code word를 생성하고, H matrix를 이용해 Maximum Likelihood 를 진행  
> ![image](https://user-images.githubusercontent.com/45085563/170306149-f5f0ecb4-db2e-471c-a2dd-368f534efde6.png)
> + 우리가 구현한 Hamming code error_rate 
> <img width="500" alt="image" src="https://user-images.githubusercontent.com/45085563/170305781-7ebcd03a-adba-4ef6-a86e-84dee2b33575.png">
>공식 문서상의 code rate 비교 

![image](https://user-images.githubusercontent.com/45085563/170305681-4b7f1eb2-4c0b-47dd-885e-c18c15ad9a55.png)
# 교수님과 주간 미팅 2022-03-18
> + (7,4) Hamming code 구현 및 error rate 비교 
>![image](https://user-images.githubusercontent.com/45085563/170303973-e9d45bc3-d10b-40fa-9067-1b4e6ec8b4ec.png)



# 일정표 
> <img width="1736" alt="image" src="https://user-images.githubusercontent.com/45085563/170303877-46cfcf0f-acf8-4891-b0b0-35d0b8351318.png">

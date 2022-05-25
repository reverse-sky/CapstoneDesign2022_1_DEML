# 5주차 주간 업무 일지 
## 시스템 설계를 위한 시스템 분석 ver.5
✒️ 7,4 hamming code 환경 기반 Sum Product Algorithm 구현 
> + 5G LDPC 환경을 위한 Sum product Algorithm 분석 진행 
👉 + 행렬 단위로 sum or product를 진행함에 있어서 update가 편리해짐을 발견
>![image](https://user-images.githubusercontent.com/45085563/170319354-0a93719c-953a-4cde-a86c-c2e122fd3419.png)
> + variable node update 방법 
>![image](https://user-images.githubusercontent.com/45085563/170319534-a3bfda52-85a9-484d-adb1-b35bbfe417dd.png)
> + factor node update 방법 

-----
✒️ 기존 방식의 문제점  
> ![image](https://user-images.githubusercontent.com/45085563/170318659-ca88b4ed-793d-4966-ac10-be9796663979.png)
> 0 or 1일 확률을  벡터로 저장해야 하므로 data의 차원이 1차원 늘어나게 되고 차원 계산의 복잡성을 처리하는 것이 까다로웠다. 
![image](https://user-images.githubusercontent.com/45085563/170318635-716fedb9-3ff0-4f38-963f-d727df63e8e3.png)
> LLR을 사용하여 1,2 vector를 하나의 수로 만드는 것이 가능함을 구현.

# 교수님과 주간 미팅 2022-04-01
> LDPC의 핵심 알고리즘인 LLR이해 
>![image](https://user-images.githubusercontent.com/45085563/170318808-2dced803-f6d2-4470-973b-28d311342d3b.png)
![image](https://user-images.githubusercontent.com/45085563/170318971-0da01ae6-a12f-43ab-a19c-52b346b14b39.png)

# 일정표 
> <img width="1733" alt="image" src="https://user-images.githubusercontent.com/45085563/170318525-fb54ef0d-571f-4bca-98c4-54e8714c4ed5.png">

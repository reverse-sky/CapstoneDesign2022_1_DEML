# CapstoneDesign_2022_1_DEML
---------
## DEML(Development of Encoding techniques using Machine Learning)
------------
### - <a href="./docs/weekly_report">주차별 회의록</a>
 ----------------
 # Contents

> ### [1. Introduction](#1-Introduction)
> ### [2. Team](#2-Team)
> ### [3. Problem_Environment](#3-Problem_Environment)
> ### [4. Simulation](#4-Simulation)
> ### [5. Result](#5-Result)
> ### [6. Expectation plan](#6-Expectation-plan)

---------------------
## 1 Introduction
> + 본 프로젝트는 상용화된 5G통신의 LDPC Simulator를 pytorch로 구현 및 성능 검증을 목표로 한다.   

> + 기존의 채널 코드는 상용화를 위한 시간복잡도 및 계산 복잡도 등의 이유로 시뮬레이션과 딥러닝의 과정을 분리시켜 비효율적인 환경에서 연구가 진행되고 있다. 구체적으로, python으로 학습한 딥러닝 parameter들을 C++이나 matlab으로 옮겨놓은 다음 pretrained model의 방식으로 딥러닝을 적용하고 있다.   
> + 최근 두 분야를 접목시키기 위해 다양한 시도가 이루어 지고 있으며, Keras기반의 물리 계층 오픈 소스 라이브러리가 등록되었다. [Sionna: An Open-Source Library for Next-Generation Physical Layer Research](https://github.com/NVlabs/sionna) 

----
><img src="https://velog.velcdn.com/images/reversesky/post/386555e6-b974-4337-b668-da0476055565/image.png" style="width: 150vw; min-width: 140px;" />

>+ 그러나 최근 딥러닝을 사용한 framework 조사 결과 pytorch를 사용한 연구가 tensor flow를 제치고 활발하게 연구되고 있다고 조사되어진다.  

[이미지 원본 출처](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2022/)

>+ 이에 본 프로젝트는 5G통신에서 사용되는 통신 물리 계층 시뮬레이터를 pytorch 기반으로 딥러닝과 연동할 수 있도록 구현하는 것을 목표로 한다. 물리계층 시뮬레이터를 pytorch로 구현함에 따라, GPU를 사용한 병렬 연산을 진행할 수 있다. 이는 기존의 python이 가진 느린 연산속도라는 단점을 극복할 수 있게 된다. 그 과정에서 네트워크 통신의 기반인 물리계층을 공부하며 통신 네트워크 시스템의 기반을 이해하고, 최종적으로 구현한 pytorch 라이브러리를 사용하여 주요 채널 코딩 중 하나인 LDPC코드를 딥러닝과 연동하여 성능을 개선하는 것을 궁극적인 목표로 한다. 

-----
## 2 Team
> ## TEAM
|Name|Department|Contact|Github|
|---|---|---|---|
|Kim MinHyuk| Hallym-univ/Big-data| hyuk6745@gmail.com |https://github.com/heigarnik
|Yong KwonSoon| Hallym-univ/Big-data| yykks3971@gmail.com|https://github.com/reversesky
-----
## **A professor in charge**
|Name|Department|Contact|
|---|---|---|
| Lim sung hoon | Hallym Univ(Prof.) | shlim@hallym.ac.kr

## 3 Problem_Environment
### OSI 7 Layer
> 5G 통신 물리 계층 시뮬레이터를 구현하기 전 네트워크 시스템의 전반적인 필요하다. 
> ![](https://velog.velcdn.com/images/reversesky/post/049b4dc0-a635-4163-81d3-99018d1beb9a/image.png)
+ 위의 그림은 ISO에서 제시한 **OSI 7계층 모델**이다. 연결된 두 호스트는 각각 7개의 계층으로 구성된 모듈을 수행함으로써 데이터 송수신이 가능하다. 전송 데이터는 송신 호스트의 응용 계층에서 시작해 하위 계층으로 순차적으로 전달되어, 최종적으로 물리 계층에서 수신 호스트에 전달된다. 수신 호스트에는 데이터를 상위 계층으로 순차적으로 이동시켜 응용 계층까지 보내준다. 
+ 여기서 계층의 최하단의 위치한 물리 계층은 전송 매체의 물리적 인터페이스에 관한 사항을 기술한다.  즉, 전송 매체에서는 개별 정보의 **BIT** 교환 문제를 다룬다.
>
 
 













## 4 Simulation
>

## 5 Result

## 6 Expectation plan

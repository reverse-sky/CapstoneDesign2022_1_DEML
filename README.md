# CapstoneDesign_2022_1_DEML
---------
## DEML(Development of Encoding techniques using Machine Learning)
------------
### - <a href="./docs/weekly_report">주차별 회의록</a>
### - <a href="./docs/src">주차별 구현 Code</a>

 ----------------
 # Contents

> ### [1. Introduction](#1-Introduction)
> ### [2. Team](#2-Team)
> ### [3. Problem_Environment](#3-Problem_Environment)
>   -  [OSI 7 Layer](#OSI-7-Layer)
>   -  [Channel Coding](#Channel-Coding) 
>   -  [Hamming (7,4) code](#Hamming(7,4)-code)  
>   -  [LDPC](#LDPC(Low-Density-Parity-Check))
> ### [4. Simulation](#4-Simulation)
>  -  [Hamming (7,4) code](#Hamming-(7,4)-code)  
>  -  [LDPC](#LDPC)
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
> 5G 통신 물리 계층 시뮬레이터를 구현하기 전 네트워크 시스템의 전반적인 이해가 필요하다. 
> ![](https://velog.velcdn.com/images/reversesky/post/049b4dc0-a635-4163-81d3-99018d1beb9a/image.png)
+ 위의 그림은 ISO에서 제시한 **OSI 7계층 모델**이다. 연결된 두 호스트는 각각 7개의 계층으로 구성된 모듈을 수행함으로써 데이터 송수신이 가능하다. 전송 데이터는 송신 호스트의 응용 계층에서 시작해 하위 계층으로 순차적으로 전달되어, 최종적으로 물리 계층에서 수신 호스트에 전달된다. 수신 호스트에는 데이터를 상위 계층으로 순차적으로 이동시켜 응용 계층까지 보내준다. 
+ 여기서 계층의 최하단의 위치한 물리 계층은 전송 매체의 물리적 인터페이스에 관한 사항을 기술한다.  즉, 전송 매체에서는 개별 정보의 **BIT** 교환 문제를 다룬다.
---

### Channel Coding 
><img src="https://velog.velcdn.com/images/reversesky/post/fccb39a0-582f-4068-8cf5-ba6b418261a7/image.png" width="600"  alt="그림 설명" /> 
>
+  통신 시스템은 기본적으로 송신기(transmitter)와 전송매체(Channel), 수신기(Receiver)로 구성되어 있다. 
> 1. 송신기는 효과적인 메세지 전송을 위하여 기저대역 신호를 변형시켜서 채널로 보내준다.    
>2. 채널은 송신기의 출력이 수신기에 도달하기 위하여 통과하는 매체로 채널은 이중 선 전선, 동축 케이블, 광섬유 케이블과 같은 유선 링크와 공기나 바닷물 같은 무선 링크로 나누어 진다.  
>3. 수신기는 채널을 통과하면서 손상된 신호를 원래의 신호에 근사하도록 복원하는 역활을 한다. 

>보통 채널 매체를 통과하면서 신호의 크기가 감쇠하고 잡음이 더해져서 수신기에 도달한다. 보내고자 하는 데이터를 $m(t)$라고 했을 때 송신기를 통과한 신호를 $s(t)= A(t)cos[2\pi f_c t + \theta(t)]$로 나타낼 수 있다. 채널 매체를 통과한 신호를 $r(t)= s(t)+n(t)$ 로 표현하는데, 여기서 $n(t)$는 잡음이며 채널에서 수신받은 신호 $r(t)$는 수신기를 통과해 $\hat m(t)$ 출력값을 나타낸다.  
> + 수신기를 통과해서 얻은 값 $\hat m(t)$을 원래의 신호 $m(t)$로 복원하는 과정을 복조화(demodulation)이라고 한다. 반대로 원래의 신호 $m(t)$를 수신기에서 복조하기 쉽게 $s(t)$로 바꾸는 과정을 부호(modulation)라고 한다. 

>다음은 위의 과정을 파트별로 세분화 한 디지털 통신 시스템의 블록도이다. 

><img src="https://velog.velcdn.com/images/reversesky/post/8af6e3a9-5c09-4675-a641-af039a660268/image.png" width="600"  alt="그림 설명" />

>  디지털 통신 시스템은 크게 소스 코딩(source coding)과 채널 코딩(channel coding)으로 나눌 수 있다. 소스 코딩은 디지털 데이터를 압축하여 전송량을 줄이는 기법이며, 주로 한정된 대역폭 특성을 극복하기 위해 사용한다. 채널 코딩은 통신 채널을 통과하면서 발생하는 오류를 수신기에서 검출하거나 정정할 수 있도록 송신기에서 의도적으로 데이터를 추가하는 기법이다. 본 프로젝트는 현재 5G NR통신에서 상용중인  LDPC 채널 코딩 방식을 pytorch를 이용하여 구현하는 것을 목표로 하고 있다.    

><img src="https://velog.velcdn.com/images/reversesky/post/5b1fa40e-b1b4-4bd3-a992-c8babc2b0b15/image.png" width="600" alt="b" />

> 다음은 AWGN채널을 사용하는 무선 이동 통신 시스템을 나타낸 구조도 이다. AWGN채널을 사용하는 채널 코딩은 크게 ENC(Encoding channel), AWGN(Additive White Gaussian Noise), DEC(Decoding-Channel)로 나누어 진다. 이는 디지털 통신 시스템 블록도에서 소스 코딩 부분을 제외한 부분으로 Channel Coder와 Modulation부분을 합쳐서 ENC, AWGN채널을 통과해서 수신받은 신호를 demodulation하고 Channel decoder 하는 부분을 합쳐서 DEC로 정의한다. 
> + AWGN 채널은 대기중에 신호를 전송시 더해지는 잡음을 표현하는 채널로서 잡음 $Z$~ $N(0,\sigma^2)$를 확률분포로 갖는다. 




> 통신에 있어서 가장 중요한 두 가지 자원은 대역폭과 송신 전력이다. 대역폭은 왜곡 없는 전송을 위해 필요한 요소이다. 송신 전력은 수신측에서 원하는 품질을 얻기 위하여 거리에 따른 손실을 거려하여 산출한다. 잡음이 신호에 대한 영향을 나타내는 척도:신호 대 잡음비(signal to noise ration)를 SNR으로 정의하며 식은 다음과 같다. 
+ $SNR [dB] = 10 log\frac{S}{N}[dB]$   
 $S$: 평균 신호 전력 $S = P_{s(avg)}= v^2_{s(rms)}$    
 $N$: 평균 잡음 전력 $N = P_{N(avg)}=v^2_{n(rms)}$  
 [정보통신기술용어해설](http://www.ktword.co.kr/word/abbr_view.php?m_temp1=1214)
 +
 디지털 통신에서는 수신측에서 원하는 비트오율(bit error rate:BER)을 기준으로 평가한다. 그러나 메세지는 한 bit만 잘못되어도 메세지가 왜곡되어진다. 따라서 n개의 bit를 보냈을 때 하나의 bit만 잘못 예측하더라도 전송한 메세지의 예측을 실패했다고 가정하는 오류 측정 기준을 BLER(Block Error Rate)으로 정의한다. 
 디지털 통신에서는 어떤 경우에나 통신을 위한 최소 SNR이 요구되며, SNR값이 커질 수록 BER는 증가한다.   

### Hamming(7,4) code  
> + LDPC를 pytorch로 구현하기 전 가장 간단한 형태의 선형 코딩을 구현하였다. 
Hamming code는 1950년 미국 Bell연구소의 Hamming이 고안한 간단한 선형 블록 부호이다. BPSK(Binary Phase Shift keying)를  사용하는 AWGN 채널에서 Hamming code는 4개의 bit에서 표현 가능한 16가지의 이진 비트를 전송 할 수 있으며 보내는 메세지를 codeword, 전송 가능한 16가지의 모든 codeword의 집합을 codebook이라고 부른다. 
+ Hamming (7,4) code는 실제 정보가 담긴 길이 4bit의 Information bit 4bits에 길이가 3인 오류 정정, 검출을 위한 Parity check bits를 덧붙여 총 길이 7의 bits로 부호화 하는 방법이다.

>+ Hamming (7,4) code는 Identity matrix $I$와  Parity check matirx $P$를 사용하여 다음과 같이 나타낼 수 있다. $G = [I,P]$
><img src="https://velog.velcdn.com/images/reversesky/post/f1f83e94-e986-404f-837e-0b2cd8d39b8a/image.png" width="300" alt="b" />
$G$는 codeword를 만들어내는 Generator matrix이며 codeword $C=MG$로 부호화된다. 

>+ $H$ matrix는 $H = [P^T,I]$로 나타낼 수 있다. 
><img src="https://velog.velcdn.com/images/reversesky/post/e41e4a84-239c-4965-9f7a-82207668217c/image.png" width="300" alt="b" />
----


[Sum Product Algorithm](https://ieeexplore.ieee.org/document/910572)




### LDPC(Low Density Parity Check)







## 4 Simulation
### Hamming (7,4) code
### LDPC

## 5 Result

## 6 Expectation plan

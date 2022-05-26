# 프로젝트를 진행하면서 작성했던 Jupyter file을 정리
---
## Week 1
Capstone Design 목표 설정 및 일정수립

------
## Week 2
AWGN 채널, Hamming code 스터디

----
## Week 3
Hamming code 구현
Encode : Hamming (7,4) 
Decode : Maximum likelihood

----
## Week 4
Hamming code 구현
Encode : Hamming (7, 4) 
Decode : Maximum likelihood
+ Upper bound - hard coding
+ lower bound - shannon limit
----
## Week 5
SPA(Sum-product Algorithm) 스터디 및 Hamming (7, 4) code 에 적용하여 구현
Encode : Hamming (7, 4)
Decode : SPA(LLR 사용 x, coo_tensor 사용 x)

----
## Week 6
SPA를 Hamming (7, 4) code 에 적용하여 구현
Encode : Hamming (7, 4)
Decode : SPA(LLR 사용 o, coo_tensor 사용 x)

---
## Week 7
SPA를 Hamming (7, 4) code 에 적용하여 구현
Encode : Hamming (7, 4)
Decode : SPA(LLR 사용 o, coo_tensor 사용 o)

---
## Week 8
LDPC 스터디 및 5G 표준 규격 개요

---
## Week 9
통신 환경관리를 위한 패키지 구현 : Communication.py
```python
class communication():
	def get_constellation()
    def get_bitmap()
    def modulate()
    def AWGN()
    def demodulate()
    
def gen_random_bits()
def dec2bin()
def bin2dec()
def cpdf_conditional_premargin()
def cpdf_conditional()
def cpdf_marginal()
```
---
## Week 10
NR LDPC 구현 : nrLDCP.py
```python
class nrLDPC():
    def get_H():
    def nrLDPC_encode():
    def LDPC_decode():
    def parity_check_LLR():

def atanh():   
def matlabsparse_2_coo():
def decimal_to_binary_tensor():
def mod():
def multiply_reduce_dim1():
def multiply_reduce_dim0():
def sum_reduce_dim0():
def resize_sparse():
```
---
## Week 11
구현한 패키지를 사용하여 1/3, 2/3, 8/9 의 Code rate에 대한 결과값 확인 및 비교
비교를 위한 성능 기준 :
[Bae, J., Abotabl, A., Lin, H., Song, K., & Lee, J. (2019). An overview of channel coding for 5G NR cellular communications. APSIPA Transactions on Signal and Information Processing, 8, E17. doi:10.1017/ATSIP.2019.10](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/CF52C26874AF5E00883E00B6E1F907C7/S2048770319000106a.pdf/an-overview-of-channel-coding-for-5g-nr-cellular-communications.pdf)
<img src="https://velog.velcdn.com/images/heigarnik/post/bcc0ecc4-f7eb-4f3e-a284-58bab72e4058/image.png">


구현한 패키지를 사용한 성능:
<img src="https://velog.velcdn.com/images/heigarnik/post/467e321c-3f85-49dc-9dc5-f29c4fd9b2de/image.png">

---
## Week 12
패키지를 활용한 pytorch 기반의 Deep learning model 개발

---
## Week 13
nrLDPC의 성능과 비교

---

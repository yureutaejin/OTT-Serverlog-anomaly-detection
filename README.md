# analysis report
**반드시 README와 코드를 같이 봐주세요**  
<구성 설명>  
**final_result_ipynb => 최종 구현코드 및 시각화코드**  
dataset => raw dataset  
wd => RNR 배분 작업 폴더  
temp_result => 작업 중 임시 저장된 데이터프레임들  
final_result_answer => 제출된 라벨링 데이터셋  

---

# Index

## **1. introduction**

- 분석 주제
- 분석 배경
- 데이터 및 전체 파이프라인 설명

## **2. Data EDA & Preprocessing**

- 데이터 결측치 확인
- 데이터 index 변경
- 데이터 describe 및 visualization
- 상관관계 분석
- 결측치 시각화
- 선형 보간법
- 시간기준 보간법
- Train / Test split
- 다변수 대치
- KNN 보간법
- 서버별 Fail 비율 구하기 / 각 서버별 데이터 split
- Autocorrelation visualizaition
- ADF Test
- 서버별 상관계수

## **3. Modeling**

Unsupervised Learning

- K_Means
- Isolation Forest

Neural Network

- LSTM-AE

## 4**. Result**

## 5. Conclusion & Discussion

---

## **1. introduction**

### **분석주제**

실시간 OTT 서비스 이용자 수 추이를 기반으로 이상 발생 시점을 탐지

### **분석배경**

스트리밍 및 OTT 서비스가 매우 많아지고 있는 요즈음, 끊김없이 안정적으로 서비스를 제공하는 것이 기업에게도 소비자 경험에도 매우 중요하다.

실시간 제공되는 스트리밍형 서비스들의 stability를 저하시키는 요인은 대개 제공하는 서버 및 네트워크 과부하에 달려있다. 특히나 특정시간에 유저가 급격하게 몰리게되면 트래픽 또한 급격하게 증가해 서비스 제공에 차질이 생긴다. 코로나 19 장기화에 따라 네트워크 트래픽이 크게 증가했으며, 특히나 스트리밍 서비스 기업들은 크게 영향을 받고있다. 대한민국은 네트워크 인프라가 잘 구축되어있어 아직까지 문제가 없었으나, 넷플릭스 등의 해외 기업들은 기본 스트리밍 화질을 낮추고 다운로드 속도를 늦추는 등 사용자 경험 만족을 떨어뜨려서라도 네트워크 과부화 방지에 총력을 기울이는 중이다.

따라서 실시간 OTT 서비스 이용자 수 추이를 기반으로 이상 발생 시점을 탐지 및 라벨링할 것이다. 해당 분석의 결과는 패턴 등을 연구하여 급격하게 request가 증가하는 경우 해당 시간에만 스트리밍 화질을 낮추거나 다운로드 속도를 늦추는 등의 방안으로 활용할 수 있을 것이다.

### **데이터 및 전체 파이프라인 설명**

**데이터 설명**

데이터 출처: AIFactory 네트워크 지능화를 위한 인공지능 해커톤

미디어 서버 13종으로부터 수집된 5분 주기의 트랜잭션 데이터 24개월치가 제공

파일명 설명 :

INFO: 상품 가입/해지, 약관 동의, 구매, 포인트 조회를 위한 서버

LOGIN: 로그인, 본인 인증, PIN 관리를 위한 서버

MENU: 초기 메뉴, 채널 카테고리 메뉴 제공을 위한 서버

STREAM: VOD 스트리밍을 위한 서버

데이터 컬럼 설명 *서버 유형 별 제공되는 컬럼에 일부 차이가 있음을 안내드립니다 (자세한 사항은 베이스라인 코드 참조)

Timestamp: [YYYYMMDD_HHmm(a)-HHmm(b)] 형식을 가지며

수집 범위는 YYYY년 MM월 DD일 HH시 mm분(a)부터 HH시 mm분(b)

Server: 수집 서버 분류(파일명 설명 참고)

Request: 수집 범위 내 발생한 서비스 요청 수

Success: 수집 범위 내 발생한 서비스 요청 성공 수

Fail: 수집 범위 내 발생한 서비스 요청 실패 수

Session: 수집 시점의 미디어 스트리밍 세션 수

서버 중 하나라도 이상이라면 최종 이상이라 간주

**파이프라인**

1. EDA & Preprocessing

2. modeling (kmeans, isolation forest, LSTM-AE)

3. model run

4. result visualization

5. result score (check answer score by F2 score evaluation in AIFACTORY)

6. adjust hyperparameter and re-score

(unsupervised는 반복문으로는 시간이 너무 오래걸리고 ide 메모리가 초과되어 lstm-ae 결과를 차용해서 outlier_fraction을 선정했습니다.)

---

## **2. Data EDA & Preprocessing**

- 데이터 결측치 확인 => timestamp index는 연속되어 문제 없지만 컬럼마다 결측치가 존재

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled.png)

- 데이터 describe 및 visualization => 연속적인 데이터를 관찰한 결과 Request와 Fail이 크게 상승하는 지점들을 확인 가능

![output.png](analysis%20report%200f449e7733fa47c8977e957a39410796/output.png)

- 상관관계 분석 => fail 데이터의 상관계수는 상대적으로 낮음을 확인할 수 있음.

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%201.png)

- 결측치 시각화

![Info와 Loggin 제공 데이터에 결측치가 많음을 확인가능](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%202.png)

Info와 Loggin 제공 데이터에 결측치가 많음을 확인가능

![오른쪽의 스파크라인은 데이터 완전성의 일반적인 모양을 요약하고 데이터세트에서 최대 및 최소 nullity가 있는 행을 나타냄](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%203.png)

오른쪽의 스파크라인은 데이터 완전성의 일반적인 모양을 요약하고 데이터세트에서 최대 및 최소 nullity가 있는 행을 나타냄

덴드로그램을 사용하면 변수 완성의 상관 관계를 보다 완벽하게 파악할 수 있으므로 상관 관계 히트맵에서 볼 수 있는 쌍별 추세보다 더 깊은 추세를 확인할 수 있음.
덴드로그램은 계층적 클러스터링 알고리즘 ( 의 제공 scipy )을 사용하여 nullity 상관 관계(이진 거리로 측정)를 통해 변수를 서로 비닝.
트리의 각 단계에서 나머지 클러스터의 거리를 최소화하는 조합에 따라 변수가 분할됨. 단조로운 변수 집합이 많을수록 전체 거리가 0에 더 가깝고 평균 거리(y축)가 0에 더 가까움.
이 그래프를 해석하려면 하향식 관점에서 읽어야 함. 0의 거리에서 함께 연결된 클러스터 잎은 서로의 존재를 완전히 예측. 한 변수는 다른 변수가 채워질 때 항상 비어 있거나 항상 둘 다 채워지거나 둘 다 비어 있을 수 있음. 이 특정 예에서 덴드로그램은 필요하므로 모든 레코드에 존재하는 변수를 함께 붙임.

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%204.png)

- 선형 보간법

![결측치가 0으로 처리됨.](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%205.png)

결측치가 0으로 처리됨.

- 시간기준 보간법 ⇒ nan값이 연속된다면 보간되지 않음. ⇒ 시간에 따라 비례하여 값이 입력됨.

![시간기준 보간법 결과](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%206.png)

시간기준 보간법 결과

- Train / Test split ⇒ 2017년은 Train, 2018년은 Test로 사용
- 다변수 대치

IterativeImputer 클래스는 누락 된 값이있는 각 기능을 다른 기능의 함수로 모델링하고 해당 추정치를 대치에 사용. 반복된 라운드 로빈 방식으로 수행. 각 단계에서 특성 열은 출력 y 로 지정되고 다른 특성 열은 입력 X 로 처리 . 회귀 변수는 알려진 y에 대해 (X, y) 에 적합. 그런 다음 회귀 변수를 사용하여 y의 결측값을 예측. 이는 각 기능에 대해 반복적인 방식으로 수행된 다음 max_iter 대치라운드에 대해 반복되고 최종 대치라운드의 결과가 return.

![음수값이 return됨.](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%207.png)

음수값이 return됨.

- KNN 보간법

Request = Success+Fail 값이 올바르게 나옴.
K-NN(k nearest neighbours) 이란 classification에 사용되는 간단한 알고리즘. 'feature similarity'를 이용해 가장 닮은(근접한) 데이터를 K개를 찾는 방식.

![2017년 (Train)](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%208.png)

2017년 (Train)

![2018년 (Test)](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%209.png)

2018년 (Test)

- 서버별 Fail 비율 구하기

⇒ column명에 request가 있으면, request 다음행인 success를 request있는 열로 나눠 성공 비율 행을 나눠주고, request가 0인 경우에는 nan이 발생하므로 fillna(0)

- Autocorrelation visualization ⇒ 시계열자료를 다루므로 연속되는 오차항들의 상관가능성 시각화

ex)

![final_pipeline ipynb 확인 필요 ex) visualization 예시](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2010.png)

final_pipeline ipynb 확인 필요 ex) visualization 예시

- ADF Test

stationary 통계적 검정으로 stationary의 경우는 시간이 변해도 일정한 분포를 따르는 경우를 말하고, non-stationary의 경우는 시간이 변해도 일정한 분포를 따르지 않는 경우를 말함. 변수별 ADF Test 진행

결과적으로 ADF Test 잘 통과

- 서버별 상관계수

success와 request는 상관관계가 높아서 request, fail, ratio 3가지 feature를 추출

ex)

![final_pipeline ipynb 확인 필요 ex)correlation 예시](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2011.png)

final_pipeline ipynb 확인 필요 ex)correlation 예시

---

## 3. Modeling

### **K-means**

***simple concept***

clustering를 수행하고 cluster에 할당하지 않는 객체들은 이상치로 취급한다.
Anomaly Score by K-Means Clustering-based Anomaly Detection(KMC)
① 절대적 거리 : A anomaly score (a1) = B anomaly score (b1)
② 상대적 거리 : A anomaly score (a1/a2) < B anomaly score (b1/b2)

outliers_fraction → hyper_parmeter. annomaly 예상 row의 갯수가 모든 column에 동일하게 나옴, 최솟값을 threshold로 지정

정답데이터 score를 보면서 outlier_fractions를 변경해야 함.

![cluster와 point간 거리 계산](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2012.png)

cluster와 point간 거리 계산

![outliers_fraction → hyper_parmeter. annomaly 예상 row의 갯수가 모든 column에 동일하게 나옴, 최솟값을 threshold로 지정
labeling은 2개로만 나눌 것이기 때문에 n_clusters는 2로 고정](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2013.png)

outliers_fraction → hyper_parmeter. annomaly 예상 row의 갯수가 모든 column에 동일하게 나옴, 최솟값을 threshold로 지정
labeling은 2개로만 나눌 것이기 때문에 n_clusters는 2로 고정
&nbsp;
### **Isolation Forest**

***simple concept***

Tree를 이용한 anomaly detection을 위한 unsupervised algorithm.

Regression Decision Tree를 기반으로 실행됨.

Regression Tree가 재귀 이진 분할을 이용하여 영역을 나누는 개념을 이용

(지점을 분리해서 격리하는데 필요한 파티션의 수 = 루트 노드~)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2014.png)

정상 데이터 ⇒ 많은 재귀 이진분할

비정상 데이터 ⇒ 상대적으로 더 적은 분할

즉 depth가 짧을 수록 비정상 데이터에 가깝다고 판단

장점 : 클러스터링 anomaly detection algorithm에 비해 계산량이 매우 적고 Robust한 모델을 만들 수 있음.

outliers_fraction → hyper_parmeter. annomaly 예상 row의 갯수가 모든 column에 동일하게 나옴, 최솟값을 threshold로 지정

정답데이터 score를 보면서 outlier_fractions를 변경해야 함.

![clustering과 동일하게 outliers_fraction(contamination) 지정이 필요](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2015.png)

clustering과 동일하게 outliers_fraction(contamination) 지정이 필요

***result***

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2016.png)

![final_pipeline ipynb 확인 필요 ex) kmeans result 예시](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2017.png)

final_pipeline ipynb 확인 필요 ex) kmeans result 예시

![kmeans 이상치 개수](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2018.png)

kmeans 이상치 개수

![final_pipeline ipynb 확인 필요 ex) isolation forest result 예시](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2019.png)

final_pipeline ipynb 확인 필요 ex) isolation forest result 예시

![If 이상치 개수](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2020.png)

If 이상치 개수

### LSTM-AE

***논문 참고***

LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection (ICML 2016)

(Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, Gautam Shroff)

***LSTM?***

basic한 RNN(Vanila RNN)구조의 the problem of Long-Term Dependencies(장기의존성 문제)를 해결하기 위해 만들어진 모델

RNN은 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 역전파시 그래디언트가 점차 줄어 학습능력이 크게 저하됨.

RNN처럼 neural network layer 한 층 구성 대신, LSTM은 4개의 layer와 cell state로 구성하여 iteration이 증가(state가 오래 경과하더라도) 그래디언트가 비교적 잘 전파됨.

h(t) ⇒ 단기 상태용 벡터

c(t) ⇒ 장기 상태용 벡터

input_gate ⇒ cell state에 유지할 정보를 선택(1)

tanh_layer ⇒ input_gate 통과한 정보를 업데이트하는 layer

forget_gate ⇒ 버릴 정보 선택 (0)

cell state update ⇒ input_gate와 forget_gate 업데이트

output_gate ⇒ ouput 내보낼 정보 결정

![RNN, LSTM 비교](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2021.png)

RNN, LSTM 비교

![input gate, forget gate](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2022.png)

input gate, forget gate

![cell state](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2023.png)

cell state

***Auto encoder?***

Auto Encoder는 모델의 출력 값과 입력 값이 비슷해지도록 학습이 수행

1. Mapping Layer(Encoder) ⇒ Encoder에서는 Input 데이터를 Bottleneck Layer 로 보내 Input 정보를 저차원으로 압축하는 역할을 수행한다.
2. Bottleneck Layer
3. Demapping Layer(Decoder) ⇒ Decoder 에서는 압축된 형태의 Input 정보를 원래의 Input 데이터로 복원한다.
4. Output Layer

Auto-Encoder의 목표는 input 데이터와 동일한 데이터를 예측하는 것이므로 Output Layer를 통해 나오는 예측 값과 실제 값의 차이를 Loss Function으로 정의하고 학습을 수행하며, 해당 Loss Function을 Reconstruction Error라고 한다.

Reconstruction Error가 threshold를 초과하면 anomalies, 아니면 normal로 규정.

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2024.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2025.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2026.png)

***Neural Network construct***

LSTM layer 방식으로 auto encoder 구축

L1 L2 ⇒ encoder

L3 ⇒ bottle neck

L4 L5 ⇒ decoder

![L2 3 4 5를 L3 4 5 6로 잘못표시](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2027.png)

L2 3 4 5를 L3 4 5 6로 잘못표시

![hyperparameter ⇒ optimizer: adam, loss_func : MSE, activation func: relu, epoch: 20, batch size: 32](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2028.png)

hyperparameter ⇒ optimizer: adam, loss_func : MSE, activation func: relu, epoch: 20, batch size: 32

![reconstruction error를 통한 threshold 지정 ⇒ anomalies 판단](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2029.png)

reconstruction error를 통한 threshold 지정 ⇒ anomalies 판단

***result***

minmaxscaling을 한 것과 안한 것, 두 개 동시 진행

![epoch 진행에 따라 mae 감소](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2030.png)

epoch 진행에 따라 mae 감소

![prediction - minmax_scaling](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2031.png)

prediction - minmax_scaling

![epoch 진행에 따라 mae 감소](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2032.png)

epoch 진행에 따라 mae 감소

![prediction - no_scaling](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2033.png)

prediction - no_scaling

![final_pipeline ipynb 확인 필요 ex) LSTM-AE result 예시](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2034.png)

final_pipeline ipynb 확인 필요 ex) LSTM-AE result 예시

---

## **4. Result**

모델별로 Prediction 결과를 종합 ⇒ 서버들 중 한 timeline에 하나라도 anomalies 등장 시 해당 timeline 전체를 anomaly로 판단

라벨을 만들어내는 것이기 때문에 따로 결과에 대한 evaluation은 불가. 

AIFactory 경진 대회 score 채점기능을 이용.(내부 정답 라벨 데이터셋)

F2-score로 평가됨. (Precision보다 Recall에 advantage를 주는 경우)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2035.png)

lstm AE의 경우 일정 점수 이상의 성능을 얻을 수 있었음. lstm_not_scaling이 minmaxscaling을 사용한 것보다 결과가 좋았음.  
따라서 LSTM-AE에서 구해진 이상치 비율을 K-means와 Isolation Forest에 적용시켜보았더니 큰 폭의 향상이 이루어졌음.

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2036.png)

---

## 5. Conclusion & Discussion

***Visualization***

(3D 이미지가 memory를 과하게 요구 ⇒ final_result_visualization_compare.ipynb에 따로 코드만 구현)

위에서부터 LSTM,  Isolation Forest, Kmeans 결과물

info_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2037.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2038.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2039.png)

login_1_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2040.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2041.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2042.png)

login_2_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2043.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2044.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2045.png)

login_3_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2046.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2047.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2048.png)

login_4_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2049.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2050.png)

login_5_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2051.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2052.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2053.png)

menu_1_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2054.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2055.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2056.png)

menu_2_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2057.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2058.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2059.png)

menu_3_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2060.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2061.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2062.png)

menu_4_data_test

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2063.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2064.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2065.png)

Isolation Forest와 Kmeans의 결과물은 유사한 분포를 가져오는 것을 확인할 수 있습니다.  

하지만 LSTM AutoEncoder을 사용하게 되면 이상치 판별 방식이 달라 다른 분포를 그리는 것을 확인할 수 있습니다.  

***Discussion***

3가지 방법론에서 같은 결과가 나왔다면 0 or 3 , 아니라면 1, 2

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2066.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2067.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2068.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2069.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2070.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2071.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2072.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2073.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2074.png)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2075.png)

IF vs Kmeans                                          IF vs LSTM                                      Kmeans vs LSTM

![Kmeans - IF](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2076.png)

Kmeans - IF

![IF - LSTM](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2077.png)

IF - LSTM

![Kmeans - LSTM](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2078.png)

Kmeans - LSTM

 LSTM정상그래프 상으로 비슷한 분포를 보였지만 다른 결과 값을 보이는 경우가 많았습니다.

특히 가장 적은 이상치를 감지했던 LSTM과 타 알고리즘을 비교했을 때 K-means의 경우 LSTM과 다른 결과를 보여주는 경우가 K-means보다 많았습니다.

***Conclusion***

머신러닝 알고리즘을 사용할 경우 이상치 비율 설정 (하이퍼 파라미터)에 민감하다는 사실을 알 수 있었습니다. 비록 학습시간은 LSTM-AE에 비해 훨씬 짧은 시간을 보이지만 적당한 이상치 비율을 알지 못하다면 값을 찾는 것에 오랜 시간이 걸릴 수 있습니다.  

또한 Kmeans와 Isolation Forest의 이상치 비율이 같게 지정하더라도 많은 수가 다른 결과를 가져오는 것을 알 수 있습니다. (약 절반 가량이 다르게 분류)

![Untitled](analysis%20report%200f449e7733fa47c8977e957a39410796/Untitled%2079.png)

여러 이상치 탐지 알고리즘에는 장단점들이 분명합니다.  

다변량 변수, label의 유무, 연산량 등 다양한 조건에 따라 적절한 알고리즘을 선택해야 합니다.

해당 프로젝트 분석은 Neural Network의 일종인 LSTM-AE로 unsupervised로는 정하기 힘든 anomaly threshold를 학습에 따라 자동으로 정하고 이상치 라벨이 없어도 어느정도 이상치라고 판단을 할 수 있는 기준을 제시했다는 것의 의의가 있으며, 이후 추가연구로는 라벨링 결과패턴 등을 연구하여 급격하게 request와 Fail 비율이 증가하는 경우 해당 시간에만 스트리밍 화질을 낮추거나 다운로드 속도를 늦추는 등의 방안으로 활용할 수 있습니다.
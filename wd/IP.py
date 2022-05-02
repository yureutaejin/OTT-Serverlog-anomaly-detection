# ======================================================================
# 분야 #1: IP 네트워크
#
# 실시간 IP 할당 개수 추이를 기반으로 이상 발생 시점을 탐지하는 문제입니다.
# DHCP란 Dynamic Host Configuration Protocol의 약자로, 클라이언트(단말)의 요청에 따라 IP 주소를 동적으로 할당 및 관리합니다.
# DHCP 서버는 서버와 클라이언트를 중개하는 방식으로 요청 단말에게 IP를 할당합니다
# IP 네트워크 문제에서는 DHCP 장비 1종으로부터 수집된 10분 주기의 IP 세션 데이터 12개월치가 제공됩니다.
# 주어진 데이터를 활용하여, 2021년 하반기(7월 1일 - 12월 31일) IP 할당 프로세스의 이상 발생 시점을 예측하세요.
#
# 데이터 파일명: DHCP.csv
#
# 데이터 컬럼 설명:
# Timestamp: [YYYYMMDD_HHmm(a)-HHmm(b)] 형식을 가지며, 수집 범위는 YYYY년 MM월 DD일 HH시 mm분(a) 부터 HH시 mm분(b)입니다.
# Svr_detect: DHCP 프로세스에서 단위 시간 내 클라이언트인 단말들이 DHCP 서버에게 연결을 요청한 횟수입니다.
# Svr_connect: DHCP 프로세스에서 단위 시간 내 클라이언트인 단말들에게 DHCP 서버와 연결이 확립됨을 나타내는 횟수입니다.
# Ss_request: DHCP 프로세스에서 단위 시간 내 서버에 연결된 단말들이 IP 할당을 요청한 횟수입니다.
# Ss_established: IP 할당 요청을 받은 DHCP 서버가 클라이언트에게 IP가 할당됨을 나타내는 횟수입니다.
#
# * 데이터에는 일부 결측치가 존재합니다
# ======================================================================

import pandas as pd

def preprocess_data():
    # 학습 데이터 읽기. 경로 설정에 주의 하세요!
    data = pd.read_csv('data/IP/DHCP.csv')
    print(f'전체 데이터 세트. \n{data}\n')

    # TODO: 예시코드 실행을 위한 Train_set/Test_set 분할입니다. 반드시 이 형태로 학습/테스트할 필요는 없습니다.
    idx_half = data.index[data['Timestamp'] == '20210630_2350-0000'].tolist()[0]
    train_set = data[:idx_half+1]  # 1.1 - 6.30 분리
    test_set = data[idx_half+1:]   # 7.1 - 12.31 분리

    print(f'1월-6월. \n{train_set}\n')
    print(f'7월-12월. \n{test_set}\n')

    # -----------------------------------
    # TODO: 데이터 분석을 통해 다양한 전처리를 시도 해보세요!
    preprocessed_train_set = train_set





    # -----------------------------------

    return preprocessed_train_set, test_set

def train_model(train_data):
    # TODO: 정상(0)과 이상(1)을 판단하기 위한 모델을 학습하세요!
    model = (lambda x: [0] * len(x))  # 모든 상황을 정상(0)으로 판단하는 샘플 모델





    return model

def save_pred(model, test_data):
    # TODO: 모델을 활용해, 2021년 하반기 전체에 대한 예측을 수행하세요!
    pred = model(test_data)

    # 예측된 결과를 제출하기 위한 포맷팅
    answer = pd.DataFrame(pred, columns=['Prediction'])
    print(f'예측 결과. \n{answer}\n')  # TODO: 제출 전 row size "29496" 확인
    answer.to_csv('IP_answer.csv', index=False)  # 제출용 정답지 저장

# TODO: 제출 파일은 2021년 7월 1일 00시 00분-10분 부터 2021년 12월 31일 23시 50분-00분 구간의 이상 이벤트를 예측한
#  .csv 형식으로 저장해야 합니다.
#  예측 데이터프레임의 크기는 [26496 * 1]입니다.

if __name__ == '__main__':

    # 데이터 전처리
    train_data, test_data = preprocess_data()

    # 모델 학습
    model = train_model(train_data)

    # 예측 결과 저장
    save_pred(model, test_data)
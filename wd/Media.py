# ======================================================================
# 분야 #2: Media 서비스
#
# 실시간 OTT 서비스 이용자 수 추이를 기반으로 이상 발생 시점을 탐지하는 문제입니다.
# OTT란 Over-The-Top의 약자로, KT에서 서비스하는 Seezn과 같이 인터넷을 통해 제공되는 각종 미디어 콘텐츠를 의미합니다.
# Media 서비스 문제에서는 네 가지 기능별 서버 13종으로부터 수집된 5분 주기의 세션 데이터 24개월치가 제공됩니다.
# 주어진 데이터를 활용하여, 2018년 전체(1월 1일 - 12월 31일) Media 데이터의 이상 발생 시점을 예측하세요.
#
# 데이터 파일명:
# Media_INFO.csv: 상품 가입/해지, 약관 동의, 구매, 포인트 조회를 위한 서버 로그
# Media_LOGIN.csv: 로그인, 본인 인증, PIN 관리를 위한 서버 로그
# Media_MENU.csv: 초기 메뉴, 채널 카테고리 메뉴 제공을 위한 서버 로그
# Media_STREAM.csv: VOD 스트리밍을 위한 서버 로그
#
# 데이터 컬럼 설명:
# Timestamp: [YYYYMMDD_HHmm(a)-HHmm(b)] 형식을 가지며, 수집 범위는 YYYY년 MM월 DD일 HH시 mm분(a) 부터 HH시 mm분(b)입니다.
# Request: 수집 범위 내 발생한 서비스 요청 수
# Success: 수집 범위 내 발생한 서비스 요청 성공 수
# Fail: 수집 범위내 발생한 서비스 요청 실패 수
# Session: 수집 시점의 미디어 스트리밍 세션 수

# * 서버 유형 별 제공되는 컬럼에 일부 차이가 있습니다
# * Server-Prefix: 서비스를 제공하는 서버가 여러개일 경우, 각 서버의 번호가 컬럼명의 앞에 위치합니다.
# * 데이터에는 일부 결측치가 존재합니다
# ======================================================================


import pandas as pd
import os


def preprocess_data():
    # 학습 데이터 읽기. 경로 설정에 주의 하세요!
    data_path = 'data/Media'
    file_list = os.listdir(data_path)

    df = []
    for file in file_list:
        file_path = os.path.join(data_path, file)
        data = pd.read_csv(file_path)
        df.append(data)

    print(f'전체 데이터 세트. \n{df}\n')

    # TODO: 예시코드 실행을 위한 Train_set/Test_set 분할입니다. 반드시 이 형태로 학습/테스트할 필요는 없습니다.
    train_set = []
    test_set = []
    for data in df:
        end_of_year = data.index[data['Timestamp'] == '20171231_2355-0000'].tolist()[0]
        train_set.append(data[:end_of_year+1])  # 2017 1.1 - 12.31 분리
        test_set.append(data[end_of_year+1:])  # 2018 1.1 - 12.31 분리

    print(f'2017년. \n{train_set}\n')
    print(f'2018년. \n{test_set}\n')

    # -----------------------------------
    # TODO: 데이터 분석을 통해 다양한 전처리를 시도 해보세요!
    preprocessed_train_set = train_set





    # -----------------------------------

    return preprocessed_train_set, test_set


def train_model(train_data):
    # TODO: 정상(0)과 이상(1)을 판단하기 위한 모델을 학습하세요!
    model = (lambda x: [0] * len(x[0]))  # 모든 상황을 정상(0)으로 판단하는 샘플 모델

    return model


def save_pred(model, test_data):
    # TODO: 모델을 활용해, 2018년 전체에 대한 예측을 수행하세요!
    pred = model(test_data)

    # 예측된 결과를 제출하기 위한 포맷팅
    answer = pd.DataFrame(pred, columns=['Prediction'])
    print(f'예측 결과. \n{answer}\n')  # TODO: 제출 전 row size "105120" 확인
    answer.to_csv('Media_answer.csv', index=False)  # 정답을 제출하기 위해 저장


# TODO: 제출 파일은 2018년 1월 1일 00시 00분-05분 부터 2018년 12월 31일 23시 55분-00분 구간의 이상 이벤트를 예측한
#  .csv 형식으로 저장해야 합니다.
#  예측 데이터프레임의 크기는 [105120 * 1]입니다.


if __name__ == '__main__':

    # 데이터 전처리
    train_data, test_data = preprocess_data()

    # 모델 학습
    model = train_model(train_data)

    # 예측 결과 저장
    save_pred(model, test_data)


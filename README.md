# README.md

## Project Name
Resnet기반 이미지 임베딩을 통한 DAGM2007 이미지 유사도 분석


#### -- Project Status: Continue

## Project Objective
* 이미지 임베딩
* 유사도 측정 및 분석

### Methods Used
* CNN (Resnet18)
* t-NSE
* similarity measures (Euclidean distance)
* Data Visualization (scatter plot)
* etc. 

### Technologies
* Python
* Pytorch
* Pandas, Numpy
* sklearn
* etc. 

## Process
1. Data Preparation
2. Feature Vector Extraction
    - using only normal data
    - using normal and abnormal data
3. Embedding 
4. Visualization 
5. Measuring similarity

## Usage
1. 데이터 준비
  1.1 다운로드 (https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html)
  1.2 디렉토리 구조 설정 
      - 아래 경로에 있는 'DAGM2007.zip'의 디렉토리 구조와 동일하게 워킹 디렉토리 구조 설정
        (https://github.com/Ahn-Project/Anomaly_Detection/blob/dagm2007/data/DAGM2007.zip)

2. 특성 벡터 추출
    - DAGM2007의 정상 데이터만 사용할 경우,
     'train_normal.py' 실행한 다음, 저장된 weight를 load하여 'train_abnormal.py' 실행
    
    - DAGM2007의 정상, 비정상 데이터 모두 사용할 경우,
     'train_4classes.py' 실행

3. 이미지 임베딩 및 시각화 / 유사도 측정
    - DAGM2007의 정상 데이터만 사용할 경우,
      'similarity_DAGM2007.py' 실행
    
    - DAGM2007의 정상, 비정상 데이터 모두 사용할 경우,
      'similarity_4classes_DAGM2007.py' 실행


## Members

**Writer : 안정현(jh.ahn991@gmail.com)**




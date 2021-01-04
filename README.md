# README.md

## Project Name
Resnet기반 이미지 임베딩을 통한 DAGM2007 이미지 유사도 분석


#### -- Project Status: Continue

## Project Objective
* 이미지 임베딩
* 유사도 측정 및 분석

### Methods Used
* CNN (Resnet18)
* t-SNE
* similarity measures (Euclidean distance)
* Data Visualization (scatter plot, boxplot)
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
3. Embedding 
4. Measuring similarity
5. Visualization 

## Usage
1. 데이터 준비

    1.1 다운로드 (https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html)
  
    1.2 디렉토리 구조 설정 
      - 아래 경로에 있는 'DAGM2007.zip'의 디렉토리 구조와 동일하게 워킹 디렉토리 구조 설정
        (https://github.com/Ahn-Project/Anomaly_Detection/blob/dagm2007/data/DAGM2007.zip)

2. 특성 벡터 추출

    2.1 'train.py' 실행 (python3 train.py)
     
    2.2 'fvs_query.py' 실행 (python3 fvs_query.py)

3. 이미지 임베딩 및 유사도 측정 / 시각화
      
      'main.py' 실행    (python3 main.py)


## Members

**Writer : 안정현(jh.ahn991@gmail.com)**




# README.md

![image](https://user-images.githubusercontent.com/43360435/125564854-adbdf389-7d7a-4b3e-876e-4ebc733ae568.png)



## Project Name
Resnet기반 이미지 임베딩을 통한 DAGM2007 이미지 유사도 분석


#### -- Project Status: Completed

## Project Objective
* 이미지 임베딩
* 유사도 측정 및 분석

### Methods Used
* CNN (Resnet18)
* t-SNE
* similarity measures (Euclidean distance)
* Data Visualization (scatter plot, boxplot)
* etc. 

### Dependencies
* matplotlib==3.3.3
* numpy==1.19.4
* pandas==1.1.5
* scikit-learn==0.19.1
* scipy==1.5.4
* torch==1.7.1+cu101
* torchvision==0.8.2+cu101


## Process
1. Data Preparation
2. Feature Vector Extraction
3. Embedding 
4. Measuring similarity
5. Visualization 

## Usage
    
    
#### 1. 데이터 준비

   ##### 1.1 다운로드 
   (https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html)
  
   ##### 1.2 디렉토리 구조 설정 
   
   - 아래 경로에 있는 'DAGM2007.zip'의 디렉토리 구조와 동일하게 워킹 디렉토리 구조 설정
   
     (https://github.com/Ahn-Project/Anomaly_Detection/blob/dagm2007/data/DAGM2007.zip)


#### 2. 이미지 유사도 측정 / 시각화

   *** 전달 인자(argument) 2개 존재: '--data', '--dim'
        
   * '--data both': normal, abnormal 데이터 모두 사용 (default)   
   * '--data normal': normal 데이터만 사용  
   * '--dim nd': n-dim feature-vector 사용한 유사도 계산 (default)   
   * '--dim 2d': n-dim feature-vector에 tsne를 적용한 embedding vector 기반 유사도 계산
      
      
   ##### 'main.py' 실행  
      
    # main.py 내 'version'(L73), 'num_epochs'(L75), n_classes(L95,L97) 조정 필요
        
    python3 main.py
    python3 main.py --dim 2d
    python3 main.py --data normal
    python3 main.py --data normal --dim 2d




## Members

**Writer : 안정현(jh.ahn991@gmail.com)**




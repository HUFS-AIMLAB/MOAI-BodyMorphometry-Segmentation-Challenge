## MOAI 2020 Body Morphometry AI Segmentation Online Challenge 1st position solution 

Final Ranking : **1st** / 20 teams



## Abstract

MOAI 2020 Body Morphometry AI Segmentation Online Challenge는 원광대학교병원과 서울아산병원의 공동 연구 주제인 Body Morphometry 중 하나로, CT 영상에서 근육과 지방에 대하여 인공지능 기술을 기반으로 자동 Segmentation 기술 개발을 진행하는 Challenge



## Introduction

### Body Morphometry란?

 모든 의료 및 건강증진의 근본이 되는 데이터로 Body Morphometry 관련 질환은 모든 질환 중 가장 높은 유병률을 나타내며 성인 인구의 1/3 이상의 사회적 문제입니다. 특히, 근육, 지방, 뼈의 변화는 많은 질병과 건강악화의 원인이자 결과로 그 중요성은 나날이 증가하고 있습니다.



### Body Morphometry 중요성

세계 각국 질병구조가 만성병 구조로 바뀌고 고령화 사회로 진전됨에 따라 예방과 관리에 대한 중요성이 재인식 되고 있습니다. 이러한 만성질환은 발병원인이 복합적으로, 진단 및 치료의 어려움이 발생합니다. 이는 완치의 한계 발생과 오진율 증가의 원인이 되기도 합니다. 최근 이러한 문제점을 개선하기 위해 개인의료기록과 다양한 의료 데이터의 통합 분석을 통한 정밀 진단 및 치료, 예후예측을 통한 예방과 관리에 대한 연구가 급진전하고 있습니다. 특히, 비만, 근감소증, 골다공증과 같은 Body Morphometry 정보와 질병 간의 유의성에 관한 연구들이 보고되면서 다양한 의료 정보의 융합 연구에 대한 중요성과 필요성은 더욱 가중되고 있습니다.



## Method

### 데이터 특징

- 최적의 Windowing Value 선정의 어려움 
  - CT영상 특성상 Hounsfield unit이라는 물리적으로 정량적인 값을 가진다. 그에 따라 windowing value를 설정하여 주목하여야 하는 부위의 Contrast를 높일 수 있다. -> 주어진 Task는 피하지방, 골격근, 내장지방 3가지의 Class로 구성 돼 있으며 Class별 windowing value도 모두 달랐고 환자마다 보여지는 양상이 달랐기에 모든 Training Set에 동일한 Windowing Value를 사용하는 것은 적합하지 않아 보였다.



- 적은 Data의 갯수
  - 의료영상을 이용하는 Deep Learning의 경우 다른 분야보다 적은 Data를 이용해야하는 이슈가 있는데 이는 overfitting과 같은 일정 성능에서 정체되는 문제를 지니고 있다.
  - Challenge에서 제공받은 Data는 100개의 CT Data였으므로, 상당히 적었고 이를 보완하기 위한 과정을 생각했어야 했다.



- Data의 다양성
  - Data마다 획든된 장비의 Protocol, 대상 등이 모두 다르기에 noise가 있는 Data, Zoom-in, Zoom-out Data, Rotated Data 등을 관찰할 수 있었다.



- Label Image의 형태학적인 특징
  - Training Set에 대한 Label Image를 관찰했을 때, 모두 비어 있는 부분이나 끊긴 부분들이 없이 깔끔하게 Labeling 돼 있었다. 경험상 Test Data에 대한 결과에서는 작은 Hole이나 고립되어 떨어지는 Cluster들이 생길 것으로 예상돼 이를 Post-processing을 통해 보정하려 하였으며, 이는 Training set의 Label Image와 유사한 결과를 Test set에 도출하도록 고안해야 했다.



### Pre-processing

- 최적의 windowing value를 선정하는 것이 힘들기에 (-250, 250) range의 random level value와 (500, 1000) range의 random window width를 Discrete uniform distribution을 따르게 설정하여 Hounsfield Unit에 Generality를 가지도록 하였다.
- random한 windowing value를 가지는 Image를 한 환자당 50장 생성하여 주어진 100장의 Image를 5000장으로 늘려 2번 Data의 특징인 적은 Data의 개수 문제 완화

- Data Augmentation



### CNN Network

![](https://github.com/hwanseung2/Asan-segmentation/blob/main/img/img1.png)
![](https://github.com/hwanseung2/Asan-segmentation/blob/main/img/img2.png)

- model : ResNet-34 based U-Net structure CNN
- Input size : 512 X 512 X 3
- Loss Function : 0.5 * Generalized Dice Loss + 0.5 * Weighted Cross Entropy
- Batch size : 4
- Learning Rate : 3e-4
- The Number of traing epochs : 20
- Optimization method : Adam 



### Post-processing

- Hole Filling
- Connected component




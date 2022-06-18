# AST-Real-time-Emergency-Classification


## **Base on**
본 프로젝트는 YuanGongND님이 구현한 [ast](https://github.com/YuanGongND/ast) 소스 코드를 기반으로 구현했습니다.<br><br>

## **Introduction**
- 고령화 사회에 접어들며 노인 1인 가구 비중이 증가하며, 고독하게 생명과 직결된 위험에 놓여있는 노인 비중도 증가하고 있습니다. 정부와 지자체는 이 문제를 파악하여 SKT 등 기업과 협업하여 도움을 줄 수 있는 인공지능을 보급하고 있다. 이에 따라 AI 스피커에 탑재될 수 있는 실시간으로 위급 상황을 감지할 수 있는 기능을 개발하고자 합니다.<br><br>

## **To Do**
- Real-time으로 오디오 신호를 읽고, 응급 상황을 예측하고 즉각적인 대응<br>
- Transformer 기반 고성능 Audio Classification 모델 AST 채택<br><br>

## Model Architecture
AST(Audio Spectrogram Transformer)<br>
1) 오디오에 대한 STFT 결과물인 Spectrogram을 n개만큼 나누고 linear projection 수행합니다.<br>
2) Linear projection 결과물에 positional embedding을 거쳐서 포지션 값을 갖고 Transformer의 Encoder 통과합니다.<br>
3) Encoder의 결과물을 활성화 함수가 sigmoid(softmax)인 Dense layer를 지나 최종 결과물의 확률 예측합니다.<br><br>

## **AST 핵심 아이디어**
- Vision Transformer와 유사한 구조를 가져 Vision Transformer로 ImageNet을 학습한 Pre-trained 파라미터를 활용합니다. 이때, Transfer Learning을 수행했을 때 응급 상황 데이터 기준 40 epoch 99.1% 정확도를 달성하는 것을 확인할 수 있습니다. <br>
- Mixup 알고리즘을 사용하여 선택된 데이터와 랜덤한 데이터를 beta 분포로 섞어 학습에 활용합니다.<br><br>


## **How to Train**
학습을 수행하기 전, label indices.csv와 data.csv를 미리 준비합니다.<br>
그리고 egs/emergency 디렉토리에서 run_emergency.sh 스크립트를 실행합니다. (./run_emergency.sh)<br><br>

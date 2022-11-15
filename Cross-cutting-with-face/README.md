# SimilarImageFinder (유사이미지탐색기)


## Description

SimilarImageFinder 는 [yolofaceV5 repo](https://github.com/elyha7/yoloface), [yolov5-face repo](https://github.com/deepcam-cn/yolov5-face) 의 Face Detection을 이용합니다. 이 모델은 얼굴의 bounding box 와 5개의 face landmark 를 감지합니다. 그 후, 이미지 간의 landmark 거리(유클리드 거리)를 이용하여 유사 이미지를 판별합니다. 

## Installation

1. 우선, 버전 충돌을 피하기 위해 새로운 가상환경을 만드는 것을 추천합니다.

```python
codna activate create -n cross-cutting python=3.8
```

1. requirements.txt 다운

```python
pip install -r requirements.txt
```

## Usage example

### Youtube Video download

[Youtube to MP4 Converter](https://ssyoutube.com/en319/)

- 위 사이트를 이용해 유튜브에서 영상을 받는다.
- 영상을 받을 때 노래 시작점이 거의 비슷하게 받기를 추천한다.
- 혹은 시작점만 같게 편집하기를 추천한다.
- 받은 영상을 한 폴더에 넣어준다.

### SimilartImageFinder Example

```python
from SimilarImageFinder import SimilarImageFinder
# 영상의 절대경로 추가 example
video_path = '/Users/ideallim/Desktop/origin_video'
"""
video capture 생략하고 싶으면 False
Detector : yoloface
video_name : 영상 이름
period : 영상 30 fps 기준 period = 30 : 1초 간격으로 이미지 저장
"""
SimilarImageFinder('yoloface', 'video_name', video_path, period=30, video_capture=True)
```

- video_path : 영상을 받은 폴더의 경로
- Detector : yoloface (현재는 yoloface 만 지원, 추후 다른 Face Detection 도 추가 예정)
- video_name : 영상 이름
- period : image_capture 시간 주기 ex) 30 fps 영상 기준, period = 30 → 1초 간격으로 이미지 저장
- video_capture : 이미 이미지 저장을 했거나, 이미지 캡처 기능이 필요 없는 경우 False
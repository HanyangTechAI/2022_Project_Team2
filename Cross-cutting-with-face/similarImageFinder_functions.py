import glob
import numpy as np
import pandas as pd
from PIL import Image
import os


def get_all_images(dataset_path):
    images = glob.glob(os.path.join(dataset_path, '**/*.jpeg'))
    return images

def _zip_person(boxes, points, gpu=True):
    """
    box 와 landmark 의 리스트를 각 사람의 box 와 landmark 로 합치는 함수입니다.
    """
    if not gpu:
        person = zip(boxes[0], points[0])
        return list(person)
    else:
        person = zip(boxes, points)
        return list(person)

def _sort_by_x1(boxes):
    """
    감지한 boundary box list 를 x1이 작은 순서대로 (왼쪽부터) 정렬하는 함수입니다.
    """
    boxes = sorted(boxes, key=lambda x: x[0][0])
    return boxes

def get_max_area_fraction_and_crop_faces(boxes, imgs_path):
    """
    2개의 이미지에서 가장 영역이 큰 사람을 찾고, area fraction 을 반환하는 함수 입니다.
    """
    area_list = []  # 2개의 영상에서 각각 제일 큰 크기의 face area 를 담을 리스트
    crop_face_list = [] # 2개의 Crop Face Image 를 담을 리스트
    for faces in boxes: # people : Image 안에 있는 Face 좌표 리스트
        max_area = 0
        max_face = None
        for face in faces:
            x1, y1, x2, y2 = face[0], face[1], face[2], face[3]
            area = (x2-x1)*(y2-y1)
            if max_area < area:
                max_area = area
                max_face = face
        area_list.append(max_area)
        crop_face_list.append(max_face)
    # Face crop 이미지로 저장
    crop_face_list = [Image.open(imgs_path[i]).crop((face[0], face[1], face[2], face[3]))
                      for i, face in enumerate(crop_face_list)]

    # img1 = cv2.cvtColor(crop_face_list[0], cv2.COLOR_BGR2RGB)
    # cv2.imshow('img1', img1)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return area_list[0]/area_list[1], crop_face_list[0], crop_face_list[1]

def generate_batch(lst, batch_size):
    """  Yields batch of specified size """
    for i in range(0, len(lst), batch_size):
        yield lst[i: i + batch_size]

def detect_images_by_gpu(detector, dataset_path, batch_size=32):
    """
    image batch 와 gpu 를 이용해 이미지들을 분석하는 함수 입니다.
    :return df_detection(감지한
    """
    df_detection = pd.DataFrame(columns=['frame_num', 'video_id', 'detect_person_num', 'boxes', 'landmarks'])
    images = get_all_images(dataset_path)
    image_path_batches = [x for x in generate_batch(images, batch_size=batch_size)]

    for image_path_batch in image_path_batches:
        images_batches = []
        image_paths = []
        for image in image_path_batch:
            image_paths.append(image)
            image = np.array(Image.open(image))
            images_batches.append(image)
        boxes, points = detector.predict(images_batches) # batch images 분석

        for detected_image in list(zip(boxes,points, image_paths)): # image 별로
            boxes = detected_image[0]
            points = detected_image[1]
            image_path = detected_image[2].replace('\\','/') # corresponding window path
            # 영상 번호
            video_id = image_path.split('/')[-2]
            # frame 번호
            frame_num = image_path.split('/')[-1][:-5] # [:-5] -> .jpeg remove
            people = _sort_by_x1(_zip_person(boxes,points))

            if len(people) > 0:
                data = {
                    'frame_num': frame_num,
                    'video_id': video_id,
                    'detect_person_num': len(people),
                    'boxes': [person[0] for person in people],
                    'landmarks': [person[1] for person in people]
                }
                df_detection = df_detection.append(data, ignore_index=True)
                print(f'{frame_num}번 째 frame : {video_id} 영상 이미지 저장')
    return df_detection

def detect_images_by_cpu(detector, dataset_path):
    """
    cpu 를 이용한
    """
    df_detection = pd.DataFrame(columns=['frame_num', 'video_id', 'detect_person_num', 'boxes', 'landmarks'])
    images = get_all_images(dataset_path)  # 모든 이미지 경로를 가져옵니다.
    for image in images:
        image = image.replace('\\', '/')
        # 영상 번호
        video_id = image.split('/')[-2]
        # frame 번호
        frame_num = image.split('/')[-1][:-5]  # [:-5] -> .jpeg remove
        # 이미지 경로 -> ndarray
        image = np.array(Image.open(image))
        # get boundary box and landmarks
        boxes, points = detector.predict(image)
        # 감지된 사람 중 왼쪽에 있는 사람 순(x1이 작은 순)으로 정렬
        people = _sort_by_x1(_zip_person(boxes, points, gpu=False))
        # 1명 이상인 경우 / 1명만 검출하고 싶으면 == 1 로 변경
        if len(people) > 0:
            data = {
                'frame_num': frame_num,
                'video_id': video_id,
                'detect_person_num': len(people),
                'boxes': [person[0] for person in people],
                'landmarks': [person[1] for person in people]
            }
            df_detection = df_detection.append(data, ignore_index=True)
            print(f'{frame_num}번 째 frame : {video_id} 영상 이미지 저장')
    return df_detection

def filter_df(df):
    """
    얼굴을 감지한 데이터프레임에서 감지된 사람 수가 같은 데이터만 추출하여 데이터프레임을 만듭니다.
    """
    # 같은 프레임의 이미지들 중에서 적어도 2번 이상 사람이 감지된 프레임만 추출
    frame_index = df['frame_num'].value_counts()[df['frame_num'].value_counts() > 1].index
    # 'frame_num' index 설정
    df = df.set_index('frame_num')

    # 빈 df_filtered 생성
    df_filtered = pd.DataFrame(
        columns=['frame_num', 'video_id', 'detect_person_num', 'boxes', 'landmarks']
    ).set_index('frame_num')

    for frame_num in frame_index:
        df1 = df.loc[frame_num]
        mask = df1.loc[frame_num].duplicated(['detect_person_num'], keep=False)
        df_filtered = pd.concat([df_filtered, df1.loc[mask]])
    return df_filtered


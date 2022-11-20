from yoloface import face_detector
from util.distance import findEuclideanDistance
from itertools import combinations
from util.VideoCapture import VideoCapture
from util.plot import plot_by_cv2
import torch
from similarImageFinder_functions import *
from verifier import FaceVerifier, VGGFace

class SimilarImageFinder:
    def __init__(self, detector, video_name, video_path, period=30, video_capture=True):
        self.video_name = video_name
        if video_capture:
            # 영상 -> 이미지
            VideoCapture(period, video_name, video_path)
            # 같은 프레임의 이미지들 중에서 적어도 2번 이상 사람이 감지된 프레임만 추출해서 data frame 생성
            df = self._make_df_detection_by_detector(detector, video_name)
        else:
            df = pd.read_parquet(f'./dataset/{video_name}/detection_data/yoloface_data.parquet')
        # 유사한 이미지 탐색
        similarity_list = self.find_similarity_images(df)
        # 이미지 출력
        plot_by_cv2(similarity_list, video_name, 30)

    def _make_df_detection_by_detector(self, detector, video_name):
        """
        Face Detection 데이터 프레임을 만드는 함수입니다.
        :param: detector ['opencv', 'ssd', 'mtcnn', 'retinaface', 'yoloface]
        :return:
        """
        if detector == 'yoloface':
            try:
                torch.tensor(1).cuda()
                yolo_detector = face_detector.YoloDetector(target_size=720, gpu=1, min_face=90)
                print('gpu 사용')
                df_detection = detect_images_by_gpu(yolo_detector, video_name, batch_size=32)
            except:
                yolo_detector = face_detector.YoloDetector(target_size=720, gpu=-1, min_face=90)
                print('cpu 사용')
                df_detection = detect_images_by_cpu(yolo_detector, video_name)
        else:
            print('아직 미구현, Deepface Detector 는 얼굴을 추출하기 때문에 customizing 필요합니다')
        # frame_num 순으로 정렬
        df_detection = df_detection.sort_values(by='frame_num')
        # filter 된 dataframe
        df_filtered = filter_df(df_detection)
        # 데이터프레임 저장
        df_filtered.to_parquet(f'./dataset/{video_name}/detection_data/yoloface_data.parquet', engine='pyarrow')

        return df_filtered

    def find_similarity_images(self, df_filtered):
        similarity_list = []
        model = VGGFace.loadModel() # Face Verification 을 위한 VGGFace Model 불러오기
        for frame in df_filtered.index.unique():
            df_temp = df_filtered.loc[frame]
            # 감지된 사람 수
            detect_person_num = df_temp['detect_person_num'].iloc[0]
            # 영상 번호 리스트
            video_num_list = list(df_temp['video_num'])
            # 선택된 영상 번호
            selected_video = None
            # 거리
            dis_min = 100000000

            for selected_video_nums in list(combinations(video_num_list, 2)):
                videonum1, videonum2 = selected_video_nums[0], selected_video_nums[1]
                boxes = list(df_temp.loc[(df_temp['video_num'] == videonum1) | (df_temp['video_num'] == videonum2)]['boxes'])
                imgs_path = [f'./dataset/{self.video_name}/frame/{frame}/{videonum1}.jpg',
                             f'./dataset/{self.video_name}/frame/{frame}/{videonum2}.jpg']
                # 2개의 이미지에서 가장 큰 얼굴 간의 비율, face verification 을 위한 crop face image
                area_fraction, crop_face1, crop_face2 = get_max_area_fraction_and_crop_faces(boxes, imgs_path)
                verified = FaceVerifier.verify(crop_face1, crop_face2, model)['verified']
                print(verified)
                # compare area_fraction and face recognition
                if 0.8 < area_fraction < 1.2 and verified:
                    landmarks = list(
                        df_temp.loc[(df_temp['video_num'] == videonum1) | (df_temp['video_num'] == videonum2)]['landmarks'])
                    dis = findEuclideanDistance(landmarks[0], landmarks[1], detect_person_num)
                    print(dis)
                    if dis < dis_min:
                        dis_min = dis
                        selected_video = selected_video_nums
            if selected_video is not None:
                similarity_list.append((frame, selected_video, dis_min))
        similarity_list = sorted(similarity_list, key=lambda x: x[2])
        return similarity_list


if __name__ == '__main__':
    video_path = 'C:/Users/JungSupLim/Desktop/video'
    # video capture 생략하고 싶으면 False
    SimilarImageFinder('yoloface', 'idle_tomboy', video_path, period=30, video_capture=False)


from yoloface import face_detector
from util.distance import findEuclideanDistance
from itertools import combinations
from util.VideoCapture import VideoCapture
from util.plot import plot_by_cv2, plot_crop_face
import torch
from similarImageFinder_functions import *
from verifier import FaceVerifier, VGGFace
import os

class SimilarImageFinder:
    def __init__(self, detector, dataset_path: os.PathLike, detection=True):
        self.dataset_path = dataset_path
        self.df = None
        if detection:
            # 같은 프레임의 이미지들 중에서 적어도 2번 이상 사람이 감지된 프레임만 추출해서 data frame 생성
            df = self._make_df_detection_by_detector(detector, dataset_path)
        else:
            if os.path.isfile(self.dataset_path + '/filtered_detection_data.parquet'):
                df = pd.read_parquet(self.dataset_path + '/filtered_detection_data.parquet')
            else:
                print('make detection dataframe first') # dataframe 이 없음
        # 유사한 이미지 탐색
        similarity_list = self.find_similarity_images(df)
        # 이미지 출력
        plot_by_cv2(dataset_path, similarity_list, 30)

    def _make_df_detection_by_detector(self, detector, dataset_path):
        """
        Face Detection 데이터 프레임을 만드는 함수입니다.
        :param: detector ['opencv', 'ssd', 'mtcnn', 'retinaface', 'yoloface]
        :return:
        """
        if detector == 'yoloface':
            try:
                torch.tensor(1).cuda()
                yolo_detector = face_detector.YoloDetector(target_size=720, gpu=0, min_face=90)
                print('gpu 사용')
                df_detection = detect_images_by_gpu(yolo_detector, dataset_path, batch_size=16)
            except:
                yolo_detector = face_detector.YoloDetector(target_size=720, gpu=-1, min_face=90)
                print('cpu 사용, 시간이 오래걸리기 때문에 gpu를 이용하는 것을 권장합니다.')
                df_detection = detect_images_by_cpu(yolo_detector, dataset_path)
        else:
            print('아직 미구현, Deepface Detector 는 얼굴을 추출하기 때문에 customizing 필요합니다')
        # frame_num 순으로 정렬
        df_detection = df_detection.sort_values(by='frame_num')
        # 데이터프레임 저장
        df_detection.to_parquet(self.dataset_path + '/detection_data.parquet', engine='pyarrow')

        self.df = df_detection.copy()
        # filter 된 dataframe
        df_filtered = filter_df(df_detection)
        df_filtered.to_parquet(self.dataset_path + '/filtered_detection_data.parquet', engine='pyarrow')

        return df_filtered

    def find_similarity_images(self, df_filtered):
        similarity_list = []
        model = VGGFace.loadModel() # Face Verification 을 위한 VGGFace Model 불러오기
        for frame in df_filtered.index.unique():
            df_temp = df_filtered.loc[frame]
            # 감지된 사람 수
            detect_person_num_list = df_temp['detect_person_num'].unique()

            # 선택된 영상 번호
            selected_video = None
            # 거리
            dis_min = 100000000

            for n_person in detect_person_num_list:
                df_temp2 = df_temp[df_temp['detect_person_num'] == n_person]
                # 영상 번호 리스트
                video_id_list = list(df_temp2['video_id'])

                for selected_video_ids in list(combinations(video_id_list, 2)):
                    videoid1, videoid2 = selected_video_ids[0], selected_video_ids[1]
                    boxes = list(df_temp2.loc[(df_temp2['video_id'] == videoid1) | (df_temp2['video_id'] == videoid2)]['boxes'])
                    imgs_path = [self.dataset_path + f'/{videoid1}/{frame}.jpeg',
                                 self.dataset_path + f'/{videoid2}/{frame}.jpeg']

                    # 2개의 이미지에서 가장 큰 얼굴 간의 비율, face verification 을 위한 crop face image
                    area_fraction, crop_face1, crop_face2 = get_max_area_fraction_and_crop_faces(boxes, imgs_path)

                    # #Verified Test : 검증 확인
                    # verified = FaceVerifier.verify(crop_face1, crop_face2, model)['verified']
                    # #check not verified face
                    # if not verified: plot_crop_face(crop_face1, crop_face2)
                    # # check verified face
                    # if verified: plot_crop_face(crop_face1, crop_face2)

                    # compare area_fraction and face recognition
                    if 0.8 < area_fraction < 1.2 and FaceVerifier.verify(crop_face1, crop_face2, model)['verified']:
                        landmarks = list(
                            df_temp2.loc[(df_temp2['video_id'] == videoid1) | (df_temp2['video_id'] == videoid2)]['landmarks']
                        )

                        dis = findEuclideanDistance(landmarks[0], landmarks[1], n_person)

                        if dis < dis_min:
                            dis_min = dis
                            selected_video = selected_video_ids

            if selected_video is not None:
                similarity_list.append((frame, selected_video, dis_min))

        similarity_list = sorted(similarity_list, key=lambda x: x[2])
        print(similarity_list)

        return similarity_list


if __name__ == '__main__':
    video_path = './dataset/tomboy'
    # video capture 생략하고 싶으면 False
    # SimilarImageFinder('yoloface', 'idle_tomboy3', video_path, period=30, video_capture=True)
    SimilarImageFinder('yoloface', video_path, detection=False)


import os
import cv2
import glob


class VideoCapture:
    def __init__(self, period=30, video_name='temp', video_path='temp'):
        # 초기 dataset 디렉토리 생성
        if not os.path.isdir("./dataset"):
            os.mkdir('./dataset')
        if not os.path.isdir(f"./dataset/{video_name}"):
            os.mkdir(f"./dataset/{video_name}")
        # if not os.path.isdir(f"./dataset/{video_name}/origin_video"):
        #     os.mkdir(f"./dataset/{video_name}/origin_video")
        if not os.path.isdir(f"./dataset/{video_name}/frame"):
            os.mkdir(f"./dataset/{video_name}/frame")
        if not os.path.isdir(f"./dataset/{video_name}/detection_data"):
            os.mkdir(f"./dataset/{video_name}/detection_data")
        self.video_path = video_path
        self.frame_path = f"./dataset/{video_name}/frame"
        self.period = period
        self.video_name = video_name
        self.video_list = self._get_mp4_files(path=self.video_path)
        #
        self.video_to_image()

    # 여러개 영상 -> 이미지 생성
    def video_to_image(self):
        if len(self.video_list) == 0:
            print('영상 폴더 안에 영상을 넣어주세요')
        else:
            for i, video in enumerate(self.video_list):
                # 경로 -> VideoCapture 객체로 변환
                video = cv2.VideoCapture(video)
                self._video_to_frame(video, i, self.period, self.video_name)

    def _get_mp4_files(self, path, ext='.mp4'):
        """ Get all mp4 files """
        files = []
        files.extend(glob.glob(f'{path}/*{ext}'))
        files.sort(key=lambda p: (os.path.dirname(p), os.path.basename(p).split('.')[0]))
        return files

    def _video_to_frame(self, video, video_num, period, video_name):
        """
        영상의 Frame 별로 이미지를 생성하는 함수입니다.
        :param video: video_path
        :param video_num: 영상 순서
        :param period: 영상 30 fps 기준 period = 60 : 2초 간격으로 이미지 추출
        """
        cnt = 0
        while video.isOpened():
            ret, image = video.read()
            if int(video.get(1)) % period == 0:
                # 초기 폴더 생성
                if not os.path.isdir(os.path.join(self.frame_path, f'{int(video.get(1)/period)}')):
                    os.mkdir(os.path.join(self.frame_path, f'{int(video.get(1)/period)}'))
                # image 저장
                cv2.imwrite(os.path.join(f'./dataset/{video_name}/frame/{int(video.get(1)/period)}/{video_num}.jpg'), image)
            cnt += 1
            if cnt > int(video.get(1)):
                print("[Frame Captured] %d Image 생성 완료" % (video.get(1) / period))
                break
        video.release()


if __name__ == '__main__':
    VideoCapture(video_name='iu_lilac', period=90, video_path='/Users/jungsuplim/Desktop/origin_video')

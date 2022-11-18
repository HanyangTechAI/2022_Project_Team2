import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def get_all_images(video_name):
    """
    dataset 폴더 안에 있는 모든 이미지 경로를 반환하는 함수입니다.
    """
    images = []
    frames = glob.glob(f'./dataset/{video_name}/frame/*')
    for frame in frames:
        for image in glob.glob(frame + '/*.jpg'):
            images.append(image.replace('\\', '/'))
    return images

def get_max_area(boxes):
    area_list = []  # 2개의 영상에서 각각 제일 큰 크기의 face area 를 담을 리스트
    for people in boxes:
        max_area = 0
        for person in people:
            x1, y1, x2, y2 = person[0], person[1], person[2], person[3]
            area = (x2-x1)*(y2-y1)
            if max_area < area: max_area = area
        area_list.append(max_area)
    return area_list


def plot_by_matplotlib(similarity_list, video_name, plot_limit=20):
    n = 0
    for frame, selected_video, _ in similarity_list:
        fig = plt.figure()
        # plot_limit 까지 plot
        if n > plot_limit: break
        xlabels = ['selected_image1', 'selected_image2']
        i = 1
        selected_video_num1 = selected_video[0]
        selected_video_num2 = selected_video[1]

        ax = fig.add_subplot(1, 2, i)
        ax.imshow(Image.open(f'./dataset/{video_name}/frame/{frame}/{selected_video_num1}.jpg'))
        ax.set_xlabel(xlabels[0])
        ax.set_xticks([]), ax.set_yticks([])

        ax = fig.add_subplot(1, 2, i + 1)
        ax.imshow(Image.open(f'./dataset/{video_name}/frame/{frame}/{selected_video_num2}.jpg'))
        ax.set_xlabel(xlabels[1])
        ax.set_xticks([]), ax.set_yticks([])
        n += 1
        fig.show()


def plot_by_cv2(similarity_list, video_name, plot_limit=20):
    n = 0
    i = 1
    for frame, selected_video, _ in similarity_list:
        if n > plot_limit : break
        img1 = np.array(Image.open(f'./dataset/{video_name}/frame/{frame}/{selected_video[0]}.jpg').resize((480, 270)))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = np.array(Image.open(f'./dataset/{video_name}/frame/{frame}/{selected_video[1]}.jpg').resize((480, 270)))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        combine_imgs = np.concatenate((img1, img2), axis=0)
        cv2.imshow(f'{i}번째로 제일 유사한 Frame', combine_imgs)
        cv2.waitKey()
        i += 1
        n += 1
    cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_by_matplotlib(path, similarity_list, video_name, plot_limit=20):
    n = 0
    for frame, selected_video, _ in similarity_list:
        fig = plt.figure()
        # plot_limit 까지 plot
        if n > plot_limit: break
        xlabels = ['selected_image1', 'selected_image2']
        i = 1

        ax = fig.add_subplot(1, 2, i)
        ax.imshow(Image.open(path + f'/{selected_video[0]}/{frame}.jpeg'))
        ax.set_xlabel(xlabels[0])
        ax.set_xticks([]), ax.set_yticks([])

        ax = fig.add_subplot(1, 2, i + 1)
        ax.imshow(Image.open(path + f'/{selected_video[0]}/{frame}.jpeg'))
        ax.set_xlabel(xlabels[1])
        ax.set_xticks([]), ax.set_yticks([])
        n += 1
        fig.show()


def plot_by_cv2(path, similarity_list, plot_limit=20):
    n = 0
    i = 1
    for frame, selected_video, _ in similarity_list:
        if n > plot_limit: break
        img1 = np.array(Image.open(path + f'/{selected_video[0]}/{frame}.jpeg').resize((480, 270)))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = np.array(Image.open(path + f'/{selected_video[1]}/{frame}.jpeg').resize((480, 270)))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        combine_imgs = np.concatenate((img1, img2), axis=0)
        cv2.imshow(f'{i} Similarly Frame', combine_imgs)
        cv2.waitKey()
        i += 1
        n += 1
    cv2.destroyAllWindows()

def plot_crop_face(img1, img2):
    img1 = cv2.cvtColor(np.array(img1.resize((200,200))), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(np.array(img2.resize((200,200))), cv2.COLOR_BGR2RGB)

    combine_img = np.concatenate((img1, img2), axis=0)
    cv2.imshow(f'not same', combine_img)
    cv2.waitKey()

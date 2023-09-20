import cv2
import numpy as np
import os

from glob import glob


img_1_init_points = [(376, 140), (367, 179), (363, 229), (356, 257), 
                     (341, 287), (321, 326), (309, 368), (293, 417), 
                     (275, 492), (280, 573), (290, 616), (300, 652), 
                     (304, 678), (321, 732), (340, 776), (360, 812), 
                     (378, 831), (417, 846), (468, 855), (517, 857), 
                     (577, 855), (616, 850), (633, 834), (649, 809), 
                     (652, 787), (667, 756), (686, 723), (703, 688), 
                     (715, 660), (724, 627), (742, 556), (739, 504), 
                     (737, 440), (720, 390), (696, 355), (667, 325), 
                     (665, 300), (671, 264), (678, 219), (678, 193), 
                     (672, 175), (659, 163), (637, 154), (605, 143), 
                     (576, 137), (545, 134), (513, 131), (478, 129), 
                     (438, 129), (409, 130), (395, 132)]

img_2_init_points = [(274, 365), (264, 383), (242, 388), (213, 377), 
                     (167, 350), (130, 341), (106, 348), (84, 361), 
                     (50, 381), (10, 425), (21, 473), (38, 509), 
                     (58, 565), (91, 643), (127, 702), (162, 746), 
                     (194, 785), (248, 819), (256, 834), (313, 867), 
                     (346, 881), (433, 892), (520, 894), (574, 882), 
                     (648, 855), (700, 834), (735, 792), (757, 751), 
                     (794, 721), (840, 677), (878, 635), (893, 606), 
                     (905, 564), (909, 515), (902, 461), (888, 412), (868, 368), (823, 336), (769, 316), (709, 321), (659, 333), (622, 331), (601, 309), (572, 302), (525, 299), (491, 298), (455, 297), (418, 297), (384, 304), (355, 318), (335, 332), (309, 340), (307, 340)]

img_3_init_points = [(227, 77), (205, 144), (190, 197), (164, 271), 
                     (162, 313), (160, 354), (163, 406), (164, 441), 
                     (170, 491), (186, 537), (198, 590), (212, 631), 
                     (229, 693), (243, 737), (265, 783), (282, 809), 
                     (309, 835), (333, 870), (353, 898), (386, 917), 
                     (402, 920), (430, 933), (455, 940), (489, 947), 
                     (533, 954), (583, 955), (608, 949), (664, 918), 
                     (704, 872), (733, 821), (776, 744), (795, 699), 
                     (811, 656), (832, 577), (842, 519), (845, 482), 
                     (848, 431), (843, 389), (842, 351), (837, 311), 
                     (823, 243), (815, 204), (806, 168), (787, 120), 
                     (770, 97), (747, 73), (704, 46), (672, 36), 
                     (618, 30), (582, 27), (535, 24), (476, 24), 
                     (422, 25), (372, 32), (301, 47), (267, 57)]

def mouse_callback(event, x, y, flags, init_points) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        # print('mouse clicked')
        # print(x, y)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('img', img)

        init_points.append((x, y))


def set_init_point(image: np.ndarray) -> tuple:
    cv2.imshow('img', image)
    init_points = []
    cv2.setMouseCallback('img', mouse_callback, init_points)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

    return init_points
    

def activate_contour(grad_img: np.ndarray, points: list, alpha: float, beta: float, gamma: float) -> np.ndarray:

    for i in range(len(points)):
        energy_min = float('inf')

        prev_point = points[i-1]
        next_point = points[(i+1) % len(points)]

        # set search region
        search_param = 4

        for search_x in range(points[i][0] - search_param, points[i][0] + search_param + 1):
            for search_y in range(points[i][1] - search_param, points[i][1] + search_param + 1):

                energy_cont = pow(search_x - prev_point[0], 2) + pow(search_y - prev_point[1], 2)
                energy_curv = pow((prev_point[0]) - 2*(search_x) + (next_point[0]), 2) + pow((prev_point[1]) - 2*(search_y) + next_point[1], 2)
                energy_img = -1 * abs(grad_img[search_y][search_x])
                energy_total = alpha * energy_cont + beta * energy_curv + gamma * energy_img

                if energy_total < energy_min:
                    energy_min = energy_total
                    points[i] = (search_x, search_y)
                    
    return points

def gradient(img: np.ndarray, kernal_size: int) -> np.ndarray:

    grad_img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernal_size)
    grad_img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernal_size)
    grad_img = cv2.addWeighted(cv2.convertScaleAbs(grad_img_x), 0.5, cv2.convertScaleAbs(grad_img_y), 0.5, 0)

    # grad_img = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

    return grad_img


if __name__ == '__main__':
    Max_iter = 200
    param_img_1 = [1, 10, 10000, 3] # alpha, beta, gamma, kernal_size for img_1
    param_img_2 = [1, 10, 200, 3] # alpha, beta, gamma, kernal_size for img_2
    param_img_3 = [1, 10, 10000, 3] # alpha, beta, gamma, kernal_size for img_3

    img_list = glob('./test_img/*.jpg')
    img_list.sort()
    print(img_list)

    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if i == 0:
            init_points = img_1_init_points
            alpha, beta, gamma, kernal_size = param_img_1
        elif i == 1:
            init_points = img_2_init_points
            alpha, beta, gamma, kernal_size = param_img_2
        elif i == 2:
            init_points = img_3_init_points
            alpha, beta, gamma, kernal_size = param_img_3

        points_img = img.copy()
        init_points = set_init_point(img)
        print(init_points)
        img = points_img.copy()


        img = cv2.GaussianBlur(img, (5, 5), 0)
        grad_img = gradient(img, kernal_size)
        cv2.imshow('grad_img', grad_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img_copy = img.copy()
        width = img_copy.shape[1]
        height = img_copy.shape[0]
        video_writer = cv2.VideoWriter('result/{}.mp4'.format(os.path.basename(img_path).split('.')[0]), 
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       10,
                                       (width, height),
                                       False)
                                       
        for i in range(len(init_points)):
            cv2.circle(img_copy, init_points[i], 3, (0, 0, 255), -1)
            if i > 0:
                cv2.line(img_copy, init_points[i-1], init_points[i], (0, 0, 255), 1)
            if i == len(init_points)-1:
                cv2.line(img_copy, init_points[i], init_points[0], (0, 0, 255), 1)
        cv2.imshow('img', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for iter in range(Max_iter):
            print('iter: ', iter)

            if iter % 1 == 0:
                img_copy = img.copy()
                # cv2.imshow('img', img)
                for i in range(len(init_points)):
                    cv2.circle(img_copy, init_points[i], 3, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(img_copy, init_points[i-1], init_points[i], (0, 0, 255), 1)
                    if i == len(init_points) - 1:
                        cv2.line(img_copy, init_points[i], init_points[0], (0, 0, 255), 1)
            
            video_writer.write(img_copy)

            init_points = activate_contour(grad_img, init_points, alpha, beta, gamma)

        video_writer.release()
        video_writer = None
        
        cv2.imwrite('result/{}.jpg'.format(os.path.basename(img_path).split('.')[0]), img_copy)
        cv2.imshow('img', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
            




        
 



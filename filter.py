import cv2
import numpy as np
from model import FaceKeypointsCaptureModel


def apply_filter(frame, pts_dict):
    left_eye_center_x = pts_dict["left_eye_center_x"]
    left_eye_center_y = pts_dict["left_eye_center_y"]
    left_eye_inner_corner_x = pts_dict["left_eye_inner_corner_x"]
    left_eye_inner_corner_y = pts_dict["left_eye_inner_corner_y"]
    left_eye_outer_corner_x = pts_dict["left_eye_outer_corner_x"]
    left_eye_outer_corner_y = pts_dict["left_eye_outer_corner_y"]
    left_eyebrow_outer_end_x = pts_dict["left_eyebrow_outer_end_x"]
    left_eyebrow_outer_end_y = pts_dict["left_eyebrow_outer_end_y"]
    radius_left = distance((left_eye_center_x, left_eye_center_y),
                           (left_eye_inner_corner_x, left_eye_inner_corner_y))

    right_eye_center_x = pts_dict["right_eye_center_x"]
    right_eye_center_y = pts_dict["right_eye_center_y"]
    right_eye_outer_corner_x = pts_dict["right_eye_outer_corner_x"]
    right_eye_outer_corner_y = pts_dict["right_eye_outer_corner_y"]
    right_eye_inner_corner_x = pts_dict["right_eye_inner_corner_x"]
    right_eye_inner_corner_y = pts_dict["right_eye_inner_corner_y"]
    right_eyebrow_outer_end_x = pts_dict["right_eyebrow_outer_end_x"]
    right_eyebrow_outer_end_y = pts_dict["right_eyebrow_outer_end_y"]
    radius_right = distance((right_eye_center_x, right_eye_center_y),
                           (right_eye_inner_corner_x, right_eye_inner_corner_y))
    
    radius_eyes = distance((right_eye_center_x, right_eye_center_y),
                           (right_eyebrow_outer_end_x, right_eyebrow_outer_end_y))
    
    length_eyes = distance((left_eye_center_x, left_eye_center_y),
                           (right_eye_outer_corner_x, right_eye_outer_corner_y))
    
    mouth_center_top_lip_x = pts_dict["mouth_center_top_lip_x"]
    mouth_center_top_lip_y = pts_dict["mouth_center_top_lip_y"]
    mouth_center_bottom_lip_x = pts_dict["mouth_center_bottom_lip_x"]
    mouth_center_bottom_lip_y = pts_dict["mouth_center_bottom_lip_y"]
    height_mouth = distance((mouth_center_top_lip_x, mouth_center_top_lip_y),
                           (mouth_center_bottom_lip_x, mouth_center_bottom_lip_y))
    
    mouth_left_corner_x = pts_dict["mouth_left_corner_x"]
    mouth_left_corner_y = pts_dict["mouth_left_corner_y"]
    mouth_right_corner_x = pts_dict["mouth_right_corner_x"]
    mouth_right_corner_y = pts_dict["mouth_right_corner_y"]
    length_mouth = distance((mouth_left_corner_x, mouth_left_corner_y),
                           (mouth_right_corner_x, mouth_right_corner_y))
    
    nose_tip_x = pts_dict["nose_tip_x"]
    nose_tip_y = pts_dict["nose_tip_y"]
    face_length = int(1.7 * distance((mouth_left_corner_x, mouth_left_corner_y),
                           (mouth_right_corner_x, mouth_right_corner_y)))
    face_height = int(3 * distance((nose_tip_x, nose_tip_y),
                           (mouth_center_top_lip_x, mouth_center_top_lip_y)))
    
    filterpath = "GuyFawkes";
    frame = apply_filter_eye_helper(frame, int(left_eye_center_x), int(left_eye_center_y), int(radius_left))
    frame = apply_filter_eye_helper(frame, int(right_eye_center_x), int(right_eye_center_y), int(radius_right))
    #frame = apply_filter_mouth_helper(frame, int((left_eye_inner_corner_x+right_eye_inner_corner_x)/2),
                              #  int((left_eye_inner_corner_y+right_eye_inner_corner_y)/2), int(length_eyes), int(radius_eyes), filterpath)
    
    #frame = apply_filter_mouth_helper(frame, int(nose_tip_x),
     #                           int(nose_tip_y), int(face_length), int(face_height), filterpath)
    
    frame = apply_filter_mouth_helper(frame, int(mouth_center_top_lip_x),
                                int(mouth_center_top_lip_y), int(length_mouth), int(height_mouth), 'moustache')
    return frame


def apply_filter_eye_helper(frame, x, y, radius):
    adjust_rad = radius - 3
    filter_img = cv2.resize(cv2.imread("filters/sharingan.png"),
                            (2*adjust_rad, 2*adjust_rad))

    slice = frame[y-adjust_rad:y+adjust_rad, x-adjust_rad:x+adjust_rad, :]
    for i in range(slice.shape[2]):
        for j in range(slice.shape[1]):
            slice[filter_img[:, j, i] != 0, j, i] = filter_img[filter_img[:, j, i]!=0, j, i]
    frame[y-adjust_rad:y+adjust_rad, x-adjust_rad:x+adjust_rad, :] = slice
    return frame

def apply_filter_mouth_helper(frame, x, y, radius, height, filterpath):
    adjust_rad = radius - 3
    filter_img = cv2.resize(cv2.imread(f"filters/{filterpath}.png",cv2.IMREAD_UNCHANGED),
                            (2*radius,2*height))
    filter_img = cv2.cvtColor(filter_img, cv2.COLOR_RGB2RGBA).copy()
    slice = frame[y-height:y+height , x-radius:x+radius, :]
    channels = slice.shape[2]
    alpha_s = filter_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(channels):
        slice[: , :, c] = (alpha_s * filter_img[:, :, c] +
                                  alpha_l * slice[: , :, c])
    #for i in range(slice.shape[2]):
        #for j in range(slice.shape[1]):
            #slice[filter_img[:, j, i] != 0, j, i] = filter_img[filter_img[:, j, i]!=0, j, i]
    #frame[y-height:y+height , x-radius:x+radius, :] = slice
    return frame

def distance(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(sum((pt1-pt2)**2))

if __name__ == '__main__':
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
    
    import matplotlib.pyplot as plt
    import cv2
    img = cv2.cvtColor(cv2.imread('dataset/trial1.jpg'), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img, (96, 96))
    img1 = img1[np.newaxis, :, :, np.newaxis]
    
    print(img1.shape)

    pts, pts_dict = model.predict_points(img1)
    pts1, pred_dict1 = model.scale_prediction((0, 200), (0, 200))

    fr = apply_filter(img, pred_dict1)

    plt.figure(0)
    plt.imshow(fr, cmap='gray')
    plt.show()
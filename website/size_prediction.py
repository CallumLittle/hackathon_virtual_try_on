import tensorflow as tf
from PIL import Image
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
import numpy as np
from tf_bodypix.draw import draw_poses

bodypix_model = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16))
part_name_to_id = {
    'left_face': 0,
    'right_face': 1,
    'left_upper_arm_front': 2,
    'left_upper_arm_back': 3,
    'right_upper_arm_front': 4,
    'right_upper_arm_back': 5,
    'left_lower_arm_front': 6,
    'left_lower_arm_back': 7,
    'right_lower_arm_front': 8,
    'right_lower_arm_back': 9,
    'left_hand': 10,
    'right_hand': 11,
    'torso_front': 12,
    'torso_back': 13,
    'left_upper_leg_front': 14,
    'left_upper_leg_back': 15,
    'right_upper_leg_front': 16,
    'right_upper_leg_back': 17,
    'left_lower_leg_front': 18,
    'left_lower_leg_back': 19,
    'right_lower_leg_front': 20,
    'right_lower_leg_back': 21,
    'left_foot': 22,
    'right_foot': 23
}

keypoint_name_to_index = {
    'nose': 0,
    'leftEye': 1,
    'rightEye': 2,
    'leftEar': 3,
    'rightEar': 4,
    'leftShoulder': 5,
    'rightShoulder': 6,
    'leftElbow': 7,
    'rightElbow': 8,
    'leftWrist': 9,
    'rightWrist': 10,
    'leftHip': 11,
    'rightHip': 12,
    'leftKnee': 13,
    'rightKnee': 14,
    'leftAnkle': 15,
    'rightAnkle': 16
}


def get_body_part_and_keypoint_line_intersection(colored_mask, body_segmentation, keypoints, part_label, keypoints_labels):
    colored_mask_torso = colored_mask.copy()
    if part_label not in part_name_to_id:
        raise ValueError('wrong body part')
    if keypoints_labels[0] not in keypoint_name_to_index or keypoints_labels[1] not in keypoint_name_to_index:
        raise ValueError('wrong keypoints')
    colored_mask_segment = colored_mask.copy()
    
    colored_mask_segment[body_segmentation != part_name_to_id[part_label]] = [0, 0, 0]
    blank_image_array = np.ones(colored_mask.shape) * 0.
    image_with_keypoints_line = cv2.line(blank_image_array,
             (round(keypoints[keypoint_name_to_index[keypoints_labels[0]]].position.x), round(keypoints[keypoint_name_to_index[keypoints_labels[0]]].position.y)),
             (round(keypoints[keypoint_name_to_index[keypoints_labels[1]]].position.x), round(keypoints[keypoint_name_to_index[keypoints_labels[1]]].position.y)),
             (0, 255, 0),
             thickness=1)
    intersection = np.logical_and(colored_mask_segment.any(axis=-1), image_with_keypoints_line.any(axis=-1))
    
    intersection_points = []
    for x in range(intersection.shape[1]):
        for y in range(intersection.shape[0]):
            if intersection[y][x] != False:
                intersection_points.append((x, y))
    return intersection_points[0], intersection_points[-1]

def get_front_waist_coordinates(colored_mask, body_segmentation, keypoints):
    return get_body_part_and_keypoint_line_intersection(colored_mask,
                                                        body_segmentation,
                                                        keypoints,
                                                        part_label='torso_front',
                                                        keypoints_labels=('leftElbow', 'rightElbow'))

def get_side_waist_coordinates(colored_mask, body_segmentation, keypoints, part_labels=['torso_front', 'torso_back'], keypoints_label='leftElbow'):
    colored_mask_torso = colored_mask.copy()  
    colored_mask_torso[(body_segmentation != part_name_to_id[part_labels[0]]) & (body_segmentation != part_name_to_id[part_labels[1]]) ] = [0, 0, 0]
    
    coordinates = (round(keypoints[keypoint_name_to_index[keypoints_label]].position.x), round(keypoints[keypoint_name_to_index[keypoints_label]].position.y))
    
    horizontal_line = np.arange(colored_mask_torso.shape[1])[colored_mask_torso.any(axis=-1)[coordinates[1]]]
    y = coordinates[1]
    x1, x2 = horizontal_line[0], horizontal_line[-1]
    return ((x1, y), (x2, y))
    
def get_front_height_in_pixels(colored_mask, keypoints):
    non_empty_y_flag = colored_mask.any(axis=-1).any(axis=-1)
    top_y = min(np.arange(len(non_empty_y_flag))[non_empty_y_flag])
    bottom_y = round(keypoints[keypoint_name_to_index['rightAnkle']].position.y+keypoints[keypoint_name_to_index['leftAnkle']].position.y)*0.5
    return bottom_y - top_y
  
def get_front_height_coordinates(colored_mask, keypoints):
    non_empty_y_flag = colored_mask.any(axis=-1).any(axis=-1)
    top_y = round(min(np.arange(len(non_empty_y_flag))[non_empty_y_flag]))
    bottom_y = round((keypoints[keypoint_name_to_index['rightAnkle']].position.y+keypoints[keypoint_name_to_index['leftAnkle']].position.y)*0.5)
    x = round((keypoints[keypoint_name_to_index['rightAnkle']].position.x+keypoints[keypoint_name_to_index['leftAnkle']].position.x)*0.5)
    return ((x, top_y), (x, bottom_y))

def get_side_height_in_pixels(colored_mask, keypoints):
    non_empty_y_flag = colored_mask.any(axis=-1).any(axis=-1)
    top_y = min(np.arange(len(non_empty_y_flag))[non_empty_y_flag])
    bottom_y = keypoints[keypoint_name_to_index['leftAnkle']].position.y
    return bottom_y - top_y

def get_side_height_coordinates(colored_mask, keypoints):
    non_empty_y_flag = colored_mask.any(axis=-1).any(axis=-1)
    top_y = round(min(np.arange(len(non_empty_y_flag))[non_empty_y_flag]))
    bottom_y = round(keypoints[keypoint_name_to_index['leftAnkle']].position.y)
    x = round(keypoints[keypoint_name_to_index['leftAnkle']].position.x)
    return ((x, top_y), (x, bottom_y))

def get_euclidiean_distance_in_pixels(coordinates):
    return np.sqrt(np.abs(coordinates[0][0] - coordinates[1][0])**2 + np.sqrt(np.abs(coordinates[0][1] - coordinates[1][1])**2))

def get_ellipse_circumference(a, b):
    return 2*np.pi*np.sqrt(0.5*(a**2 + b**2))

def pixels_to_centimeters(height_in_cm, height_in_pixels, length_in_pixels):
    return height_in_cm * length_in_pixels / height_in_pixels


def get_waist_in_cm(front_waist_coordinates, side_waist_coordinates, height_in_cm, front_height_in_pixels, side_height_in_pixels):
    front_radius_in_pixels = get_euclidiean_distance_in_pixels(front_waist_coordinates)*0.5
    side_radius_in_pixels = get_euclidiean_distance_in_pixels(side_waist_coordinates)*0.5   
    front_radius_in_cm = pixels_to_centimeters(height_in_cm, front_height_in_pixels, front_radius_in_pixels)
    side_radius_in_cm = pixels_to_centimeters(height_in_cm, side_height_in_pixels, side_radius_in_pixels)
    return get_ellipse_circumference(front_radius_in_cm, side_radius_in_cm)


def get_bodypix_results(image):
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=0.75)
    body_segmentation = result.get_scaled_part_segmentation(mask)
    poses = result.get_poses()
    colored_mask = result.get_colored_part_mask(mask)
    return mask, body_segmentation, poses, colored_mask

# IN PROGRESS
def get_size_by_photos(front_image_path, side_image_path, height_in_cm):
    front_image = Image.open(front_image_path)
    front_image_array = tf.keras.preprocessing.image.img_to_array(front_image)
    front_mask, front_body_segmentation, front_poses, front_colored_mask = get_bodypix_results(front_image)
    
    side_image = Image.open(side_image_path)
    side_image_array = tf.keras.preprocessing.image.img_to_array(side_image)
    side_mask, side_body_segmentation, side_poses, side_colored_mask = get_bodypix_results(side_image)
    
    
    front_waist_coordinates = get_front_waist_coordinates(front_colored_mask, front_body_segmentation, front_poses[0].keypoints)
    front_height_coordinates = get_front_height_coordinates(front_colored_mask, front_poses[0].keypoints)
    
    front_height_in_pixels = get_front_height_in_pixels(front_colored_mask, front_poses[0].keypoints)
    front_image_array_with_mask_and_waist = cv2.line(
        (front_image_array + front_colored_mask).copy(),
        front_waist_coordinates[0],
        front_waist_coordinates[1],
        (255, 0, 0),
        thickness=3
    )
    front_image_array_with_mask_and_waist_height = cv2.line(front_image_array_with_mask_and_waist.copy(),
             front_height_coordinates[0],
             front_height_coordinates[1],
             (255, 0, 0),
             thickness=3)
    
    side_waist_coordinates = get_side_waist_coordinates(side_colored_mask, side_body_segmentation, side_poses[0].keypoints)
    side_height_coordinates = get_side_height_coordinates(side_colored_mask, side_poses[0].keypoints)
    side_height_in_pixels = get_side_height_in_pixels(side_colored_mask, side_poses[0].keypoints)
    side_image_array_with_mask_and_waist = cv2.line(
        (side_image_array + side_colored_mask).copy(),
             side_waist_coordinates[0],
             side_waist_coordinates[1],
             (255, 0, 0),
             thickness=10
    )
    
    
    side_image_array_with_mask_and_waist_height = cv2.line(
        side_image_array_with_mask_and_waist.copy(),
        side_height_coordinates[0],
        side_height_coordinates[1],
        (255, 0, 0),
        thickness=10
    )
    height = float(height_in_cm[0])
    waist_in_cm = get_waist_in_cm(front_waist_coordinates, side_waist_coordinates, height-5, front_height_in_pixels, side_height_in_pixels)
    
    
    return tf.keras.preprocessing.image.array_to_img(front_image_array_with_mask_and_waist_height), \
        tf.keras.preprocessing.image.array_to_img(side_image_array_with_mask_and_waist_height), \
        waist_in_cm
    #plt.imshow(tf.keras.preprocessing.image.array_to_img(image_array_with_mask_and_waist))


def save_segmented_images(front_image_segmented, side_image_segmented):
    tf.keras.preprocessing.image.save_img(
        'website/static/saved_segmented_images/front_segmented.JPG',
        front_image_segmented
    )
    tf.keras.preprocessing.image.save_img(
        'website/static/saved_segmented_images/side_segmented.JPG',
        side_image_segmented
    )

def product_size_recommender(your_size):
    size = ("XXS/XS", "S/M", "L/XL")
    waist = (104.0, 114.0, 114.0)
    if your_size <= waist[0]:
        return size[0]
    elif your_size <= waist[1]:
        return size[1]
    elif your_size <= waist[2]:
        return size[2]
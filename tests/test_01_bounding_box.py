import csv
import CS373_sign_detection as sign_detector

image_names = ['sign_1', 'sign_2', 'sign_3', 'sign_4', 'sign_5']
target_bounding_boxes_dict = {
    "sign_1": [[451, 251, 623, 503]],
    "sign_2": [[383, 298, 782, 509]],
    "sign_3": [[808, 469, 1144, 742], [305, 236, 654, 581]],
    "sign_4": [[525, 332, 824, 575]],
    "sign_5": [[538, 406, 768, 593], [235, 226, 421, 410]]
}


def read_output(image_name, step_num):
    with open(f'./tests/outputs/{image_name}_step_{step_num}_output.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = []
        for row in csv_reader:
            data.append(row)

    sample_output = []
    for row in data:
        sample_output.append(row['data'])
    
    return sample_output


def bbox_iou(boxA, boxB):
    xA_min, yA_min, xA_max, yA_max = boxA
    xB_min, yB_min, xB_max, yB_max = boxB

    # Compute intersection rectangle
    xI_min = max(xA_min, xB_min)
    yI_min = max(yA_min, yB_min)
    xI_max = min(xA_max, xB_max)
    yI_max = min(yA_max, yB_max)

    # Intersection width and height (if no overlap, clamp to 0)
    inter_width  = max(0, xI_max - xI_min)
    inter_height = max(0, yI_max - yI_min)
    inter_area   = inter_width * inter_height

    # Areas of each box
    areaA = (xA_max - xA_min) * (yA_max - yA_min)
    areaB = (xB_max - xB_min) * (yB_max - yB_min)

    # Union area = sum of areas minus intersection
    union_area = areaA + areaB - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def test_final_bounding_box():
    passed_test_cases = 0
    
    for image_name in image_names:
        input_filename = f'./Images/easy/{image_name}.png'
        
        # we read in the png fi le, and receive three pixel arrays for red, green and blue components, respectively
        # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
        (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
        image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
        image = sign_detector.scaleTo0And255And5_95Percentile(image, image_width, image_height)
        
        image_x = sign_detector.computeVerticalEdgesSobelAbsolute(image, image_width, image_height)
        image_y = sign_detector.computeHorizontalEdgesSobelAbsolute(image, image_width, image_height)
        image = sign_detector.computeEdgesSobelAbsolute(image_x, image_y, image_width, image_height)
        
        image = sign_detector.computeStandardDeviationImage7x7(image, image_width, image_height)
        image = sign_detector.computeThresholdGE(image, 31, image_width, image_height)
        
        image = sign_detector.dilation(image, image_width, image_height, kernel_radius=4, num_of_iterations=3)
        image = sign_detector.erosion(image, image_width, image_height, kernel_radius=4, num_of_iterations=3)
        
        image, component_labels = sign_detector.computeConnectedComponentLabeling(image, image_width, image_height)
        
        bounding_box_list = sign_detector.returnBBoxCoords(image, component_labels, image_width, image_height)
        
        try:
            bounding_box_len = len(bounding_box_list)
            target_bounding_box_len = len(target_bounding_boxes_dict[image_name])
            
            if bounding_box_len == target_bounding_box_len:
                for target_box, student_box in zip(target_bounding_boxes_dict[image_name], bounding_box_list):
                    IoU = bbox_iou(target_box, student_box)
                    if IoU >= 0.9:
                        passed_test_cases += 1
        except:
            continue
        
    
    assert passed_test_cases >= 4, f"You only passed {passed_test_cases}/5 test cases!"

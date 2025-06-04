import csv
import CS373_sign_detection as sign_detector

image_name = 'sign_5'
input_filename = f'./Images/easy/{image_name}.png'

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


def test_greyscale_conversion():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
    image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    
    sample_output = read_output(image_name, 1)
    index = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            pixel_value = image[y][x]
            assert (float(sample_output[index]) - 1.0) <= pixel_value <= (float(sample_output[index]) + 1.0), f"Incorrect pixel value at location y={y}, x={x}!"
            index += 1

def test_noralization():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
    image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    image = sign_detector.scaleTo0And255And5_95Percentile(image, image_width, image_height)
    
    sample_output = read_output(image_name, 2)
    index = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            pixel_value = image[y][x]
            assert (float(sample_output[index]) - 1.0) <= pixel_value <= (float(sample_output[index]) + 1.0), f"Incorrect pixel value at location y={y}, x={x}!"
            index += 1

def test_vertical_sobel_edge_detector():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
    image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    image = sign_detector.scaleTo0And255And5_95Percentile(image, image_width, image_height)
    image_x = sign_detector.computeVerticalEdgesSobelAbsolute(image, image_width, image_height)
    
    sample_output = read_output(image_name, 3)
    index = 0
    for y in range(len(image_x)):
        for x in range(len(image_x[y])):
            if x <= 6 or y <= 6 or y >= (len(image) - 6) or x >= (len(image[y]) - 6):
                index += 1
                continue
            
            pixel_value = image_x[y][x]
            assert (float(sample_output[index]) - 1.0) <= pixel_value <= (float(sample_output[index]) + 1.0), f"Incorrect pixel value at location y={y}, x={x}!"
            index += 1

def test_horizontal_sobel_edge_detector():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
    image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    image = sign_detector.scaleTo0And255And5_95Percentile(image, image_width, image_height)
    image_x = sign_detector.computeVerticalEdgesSobelAbsolute(image, image_width, image_height)
    image_y = sign_detector.computeHorizontalEdgesSobelAbsolute(image, image_width, image_height)
    
    sample_output = read_output(image_name, 4)
    index = 0
    for y in range(len(image_y)):
        for x in range(len(image_y[y])):
            if x <= 6 or y <= 6 or y >= (len(image) - 6) or x >= (len(image[y]) - 6):
                index += 1
                continue
            
            pixel_value = image_y[y][x]
            assert (float(sample_output[index]) - 1.0) <= pixel_value <= (float(sample_output[index]) + 1.0), f"Incorrect pixel value at location y={y}, x={x}!"
            index += 1

# x and y direction
def test_sobel_edge_detector():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
    image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    image = sign_detector.scaleTo0And255And5_95Percentile(image, image_width, image_height)
    image_x = sign_detector.computeVerticalEdgesSobelAbsolute(image, image_width, image_height)
    image_y = sign_detector.computeHorizontalEdgesSobelAbsolute(image, image_width, image_height)
    
    image = sign_detector.computeEdgesSobelAbsolute(image_x, image_y, image_width, image_height)
    
    sample_output = read_output(image_name, 5)
    index = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if x <= 6 or y <= 6 or y >= (len(image) - 6) or x >= (len(image[y]) - 6):
                index += 1
                continue
            
            pixel_value = image[y][x]
            assert (float(sample_output[index]) - 1.0) <= pixel_value <= (float(sample_output[index]) + 1.0), f"Incorrect pixel value at location y={y}, x={x}!"
            index += 1

def test_standard_deviation_filter():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
    image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    image = sign_detector.scaleTo0And255And5_95Percentile(image, image_width, image_height)
    image_x = sign_detector.computeVerticalEdgesSobelAbsolute(image, image_width, image_height)
    image_y = sign_detector.computeHorizontalEdgesSobelAbsolute(image, image_width, image_height)
    
    image = sign_detector.computeEdgesSobelAbsolute(image_x, image_y, image_width, image_height)
    
    image = sign_detector.computeStandardDeviationImage7x7(image, image_width, image_height)
    
    sample_output = read_output(image_name, 6)
    index = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if x <= 14 or y <= 14 or y >= (len(image) - 14) or x >= (len(image[y]) - 14):
                index += 1
                continue
            
            pixel_value = image[y][x]
            assert (float(sample_output[index]) - 1.0) <= pixel_value <= (float(sample_output[index]) + 1.0), f"Incorrect pixel value at location y={y}, x={x}!"
            index += 1

def test_thresholding():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = sign_detector.readRGBImageToSeparatePixelArrays(input_filename)
    image = sign_detector.computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    image = sign_detector.scaleTo0And255And5_95Percentile(image, image_width, image_height)
    image_x = sign_detector.computeVerticalEdgesSobelAbsolute(image, image_width, image_height)
    image_y = sign_detector.computeHorizontalEdgesSobelAbsolute(image, image_width, image_height)
    
    image = sign_detector.computeEdgesSobelAbsolute(image_x, image_y, image_width, image_height)
    
    image = sign_detector.computeStandardDeviationImage7x7(image, image_width, image_height)
    image = sign_detector.computeThresholdGE(image, 31, image_width, image_height)
    
    sample_output = read_output(image_name, 7)
    index = 0
    
    intersection = 0
    union = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if x <= 14 or y <= 14 or y >= (len(image) - 14) or x >= (len(image[y]) - 14):
                index += 1
                continue
            
            pixel_value = image[y][x]
            student_output = bool(pixel_value)
            target = bool(int(sample_output[index]))
            
            if student_output or target:
                union += 1
                if student_output and target:
                    intersection += 1
            
            index += 1
    
    if union == 0:
        IoU = 0.0
    else:
        IoU = intersection / union
    
    assert IoU >= 0.9, "Binary mask either too large or too small!"


def test_dilation():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
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
    
    sample_output = read_output(image_name, 8)
    index = 0
    
    intersection = 0
    union = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if x <= 14 or y <= 14 or y >= (len(image) - 14) or x >= (len(image[y]) - 14):
                index += 1
                continue
            
            pixel_value = image[y][x]
            student_output = bool(pixel_value)
            target = bool(int(sample_output[index]))
            
            if student_output or target:
                union += 1
                if student_output and target:
                    intersection += 1
            
            index += 1
    
    if union == 0:
        IoU = 0.0
    else:
        IoU = intersection / union
    
    assert IoU >= 0.9, "Binary mask either too large or too small!"


def test_erosion():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
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
    
    sample_output = read_output(image_name, 9)
    index = 0
    
    intersection = 0
    union = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if x <= 14 or y <= 14 or y >= (len(image) - 14) or x >= (len(image[y]) - 14):
                index += 1
                continue
            
            pixel_value = image[y][x]
            student_output = bool(pixel_value)
            target = bool(int(sample_output[index]))
            
            if student_output or target:
                union += 1
                if student_output and target:
                    intersection += 1
            
            index += 1
    
    if union == 0:
        IoU = 0.0
    else:
        IoU = intersection / union
    
    assert IoU >= 0.9, "Binary mask either too large or too small!"


def test_connected_component_labeling():
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
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
    
    sample_output = read_output(image_name, 10)
    index = 0
    
    intersection = 0
    union = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if x <= 14 or y <= 14 or y >= (len(image) - 14) or x >= (len(image[y]) - 14):
                index += 1
                continue
            
            pixel_value = image[y][x]
            student_output = bool(pixel_value)
            target = bool(int(sample_output[index]))
            
            if student_output or target:
                union += 1
                if student_output and target:
                    intersection += 1
            
            index += 1
    
    if union == 0:
        IoU = 0.0
    else:
        IoU = intersection / union
    
    assert IoU >= 0.9, "Binary mask either too large or too small!"


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
    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
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
    except:
        assert False, "No bounding box detected!"
    assert bounding_box_len == 2, "Too few or too many detected bounding boxes!"
    
    target_box_1 = [538, 406, 768, 593]
    student_box_1 = bounding_box_list[0]
    
    IoU_1 = bbox_iou(target_box_1, student_box_1)
    
    target_box_2 = [235, 226, 421, 410]
    student_box_2 = bounding_box_list[1]
    
    IoU_2 = bbox_iou(target_box_2, student_box_2)
    
    
    assert IoU_1 >= 0.9 and IoU_2 >= 0.9, f"Bounding box(es) too small or too large!"
    
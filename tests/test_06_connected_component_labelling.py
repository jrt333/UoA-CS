import csv
import CS373_sign_detection as sign_detector

image_names = ['sign_1', 'sign_2', 'sign_3', 'sign_4', 'sign_5']

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

def test_connected_component_labeling():
    for image_name in image_names:
        input_filename = f'./Images/easy/{image_name}.png'
        
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

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

def test_sobel_edge_detector():
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
        
        sample_output = read_output(image_name, 5)
        index = 0
        for y in range(len(image)):
            for x in range(len(image[y])):
                if x <= 6 or y <= 6 or y >= (len(image) - 6) or x >= (len(image[y]) - 6):
                    index += 1
                    continue
                
                pixel_value = image[y][x]
                assert (float(sample_output[index]) - 1.0) <= pixel_value <= (float(sample_output[index]) + 1.0), f"Image: {image_name} | Incorrect pixel value at location y={y}, x={x}!"
                index += 1

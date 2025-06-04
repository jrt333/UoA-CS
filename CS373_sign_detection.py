# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

TEST_MODE = False



class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array

# Step 1.1
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = None
    return greyscale_pixel_array

# Step 1.2
def scaleTo0And255And5_95Percentile(pixel_array, image_width, image_height):
    return pixel_array


# Step 2 - compute the vertical and horizontal sobel filter
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    return pixel_array

def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    return pixel_array

def computeEdgesSobelAbsolute(pixel_array_x, pixel_array_y, image_width, image_height):
    pixel_array = None
    return pixel_array


# Step 3- compute StandardDeviationImage7x7
def computeStandardDeviationImage7x7(pixel_array, image_width, image_height):
    return pixel_array


# Step 4 - compute the threshold values
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    return pixel_array


# Step 5.1 - compute the dilation
def dilation(pixel_array, image_width, image_height, kernel_radius, num_of_iterations=1):
    return pixel_array


# Step 5.2 - compute the erosion
def erosion(pixel_array, image_width, image_height, kernel_radius, num_of_iterations=1):
    return pixel_array


# Step 6 - compute connected component labelling
def computeConnectedComponentLabeling(binary_array, image_width, image_height):
    ccResult = createInitializedGreyscalePixelArray(image_width, image_height)
    component_labels = []
    return ccResult, component_labels


# Step 7 - get bounding boxes
def returnBBoxCoords(image, component_labels, image_width, image_height):
    if len(component_labels) <= 0:
        print("No region found!")
        return None

    bounding_box_list = []
    return bounding_box_list


def displayGreyscaleImage(image):
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(image, cmap='gray')
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.show()


# This is our code skeleton that performs the road sign detection.
def main(input_path, output_path):
    # This is the default input image, you may want to test more images by add the image file name to the 'image_names' list.
    image_names = ['sign_1']

    for image_name in image_names:
        input_filename = f'./Images/easy/{image_name}.png'
        if TEST_MODE:
            input_filename = input_path
        # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
        # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
        (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

        
        ##################################################################
        ##################################################################
        ##################################################################
        ## convert images to grayscale format
        image = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
        
        
        ## compute images 5_95 percent
        image = scaleTo0And255And5_95Percentile(image, image_width, image_height)


        ### Edge Detection (Sobel Filter)###
        image_x = computeVerticalEdgesSobelAbsolute(image, image_width, image_height)
        image_y = computeHorizontalEdgesSobelAbsolute(image, image_width, image_height)

        # compute the image from sobel vertical and horizontal results
        image = computeEdgesSobelAbsolute(image_x, image_y, image_width, image_height)

        
        # apply Standard deviation filter
        image = computeStandardDeviationImage7x7(image, image_width, image_height)


        # apply the threshold to images, you can use threshold value as 31.
        image = computeThresholdGE(image, 31, image_width, image_height)

        
        # apply dilation and erosion, number of dilation and erosion are 3 and the kernel_radius is 4
        image = dilation(image, image_width, image_height, kernel_radius=4, num_of_iterations=3)
        image = erosion(image, image_width, image_height, kernel_radius=4, num_of_iterations=3)
        
        
        # compute the component labels for images
        image, component_labels = computeConnectedComponentLabeling(image, image_width, image_height)


        # find the bounding boxs of the images
        ############################################
        ### Bounding box coordinates information ###
        ### bounding_box[0] = min x ###
        ### bounding_box[1] = min y ###
        ### bounding_box[2] = max x ###
        ### bounding_box[3] = max y ###
        ############################################
        bounding_box_list = returnBBoxCoords(image, component_labels, image_width, image_height)
        print('Bounding box location:', bounding_box_list)

        ##################################################################
        ##################################################################
        ##################################################################
        
        px_array = px_array_r  # reassign this variable if want to display a colourful (RGB) image
        
        # draw the bounding boxs on the images
        if not TEST_MODE:
            # Saving output image to the above directory
            # pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)

            # Show image with bounding box on the screen
            pyplot.imshow(px_array, cmap='gray', aspect='equal')
            pyplot.show()
        else:
            # Please, DO NOT change this code block!
            pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)
            # pyplot.imshow(px_array, cmap='gray', aspect='equal')
            # pyplot.show()


if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1

    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True

    main(input_path, output_path)

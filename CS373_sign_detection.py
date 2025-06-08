# Built in packages
import math
import sys
from collections import deque

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
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(image_height):
        for x in range(image_width):
            r = pixel_array_r[y][x]
            g = pixel_array_g[y][x]
            b = pixel_array_b[y][x]
            grey_val = 0.3 * r + 0.6 * g + 0.1 * b
            greyscale_pixel_array[y][x] = int(round(grey_val))
    return greyscale_pixel_array


# Step 1.2
def scaleTo0And255And5_95Percentile(pixel_array, image_width, image_height):
    flat = sorted(v for row in pixel_array for v in row)
    n = len(flat)
    low = flat[int(0.05 * n)]
    high = flat[int(0.95 * n)] - 1
    if high <= low:
        high = low + 1
    scale = 255.0 / (high - low)
    return [[int(round(max(0, min(255, (v - low) * scale)))) for v in row] for row in pixel_array]


# Step 2 - compute the vertical and horizontal sobel filter
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            g = sum(pixel_array[y + dy][x + dx] * kernel[dy + 1][dx + 1] for dy in range(-1, 2) for dx in range(-1, 2)) # dy, dx range from -1 to +1 to index neighbors relative to (y, x)
            result[y][x] = abs(g) / 8.0

    return result


def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            g = sum(pixel_array[y + dy][x + dx] * kernel[dy + 1][dx + 1] for dy in range(-1, 2) for dx in range(-1, 2))
            result[y][x] = abs(g) / 8.0

    return result


def computeEdgesSobelAbsolute(pixel_array_x, pixel_array_y, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            result[y][x] = pixel_array_x[y][x] + pixel_array_y[y][x]
            if result[y][x] > 255:
                result[y][x] = 255

    return result


# Step 3- compute StandardDeviationImage7x7
def computeStandardDeviationImage7x7(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    radius = 3
    area = (2 * radius + 1) ** 2

    for y in range(radius, image_height - radius):
        for x in range(radius, image_width - radius):
            window = [pixel_array[y + dy][x + dx] for dy in range(-radius, radius + 1) for dx in
                      range(-radius, radius + 1)]
            mean = sum(window) / area
            variance = sum(v * v for v in window) / area - mean * mean
            result[y][x] = math.sqrt(max(0.0, variance))

    return result


# Step 4 - compute the threshold values
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            result[y][x] = 255 if pixel_array[y][x] >= threshold_value else 0
    return result


# Step 5.1 - compute the dilation
def dilation(pixel_array, image_width, image_height, kernel_radius, num_of_iterations=1):
    offsets = [(dx, dy) for dy in range(-kernel_radius, kernel_radius + 1) for dx in
               range(-kernel_radius, kernel_radius + 1)]
    prev = [row[:] for row in pixel_array]
    result = [row[:] for row in pixel_array]
    for _ in range(num_of_iterations):
        for y in range(kernel_radius, image_height - kernel_radius):
            for x in range(kernel_radius, image_width - kernel_radius):
                result[y][x] = 255 if any(  #As soon as any neighbor is non-zero, the pixel should dilate to 255.
                    prev[y + dy][x + dx]
                    for dx, dy in offsets
                ) else 0
        prev, result = result, prev

    return prev


# Step 5.2 - compute the erosion
def erosion(pixel_array, image_width, image_height, kernel_radius, num_of_iterations=1):
    offsets = [
        (dx, dy)
        for dy in range(-kernel_radius, kernel_radius + 1)
        for dx in range(-kernel_radius, kernel_radius + 1)
    ]

    prev = [row[:] for row in pixel_array] #read from
    result = [row[:] for row in pixel_array] #write to

    for _ in range(num_of_iterations):
        for y in range(kernel_radius, image_height - kernel_radius):
            for x in range(kernel_radius, image_width - kernel_radius):
                result[y][x] = 255 if all( #when *all* neighbors are non-zero should the pixel remain 255
                    prev[y + dy][x + dx]
                    for dx, dy in offsets
                ) else 0
        prev, result = result, prev

    return prev


# Step 6 - compute connected component labelling
def computeConnectedComponentLabeling(binary_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    component_labels = []
    label = 1
    shifts = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy] # Precomputing avoids rebuilding this list for every pixel.

    for y in range(image_height):
        for x in range(image_width):
            if binary_array[y][x] and not result[y][x]:
                queue = deque([(x, y)])
                result[y][x] = label
                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in shifts:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < image_width and 0 <= ny < image_height and binary_array[ny][nx] and not
                        result[ny][nx]): # Check bounds and whether neighbor is foreground & unlabeled
                            result[ny][nx] = label
                            queue.append((nx, ny))
                component_labels.append(label)
                label += 1

    return result, component_labels


# Step 7 - get bounding boxes
def returnBBoxCoords(image, component_labels, image_width, image_height):
    boxes = []
    for lbl in component_labels:
        coords = [(x, y) for y in range(image_height) for x in range(image_width) if image[y][x] == lbl]
        if coords:
            xs, ys = zip(*coords)
            boxes.append([min(xs), min(ys), max(xs), max(ys)])

    return sorted(boxes, key=lambda b: b[0], reverse=True)


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
        (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(
            input_filename)

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

# # import cv2
# # from matplotlib import pyplot as plt
# # from cv2 import CHAIN_APPROX_NONE, RETR_TREE, ROTATE_180, ROTATE_90_CLOCKWISE, cvtColor, findContours, resize
# # import numpy as np
# # import cv2 
# # from PIL import Image
# # import math
# # from matplotlib import pyplot as plt
  
# # img_Path = "/Users/viglud/QWOP/test.png"
# # img = cv2.imread(img_Path)
  
# # # Opening image
# # img= cv2.bitwise_not(img)



# # #cropped = img[300:1610, 90:1080]



# # #cropped_Pendul_ID = img[1500:1550, 170:380]




# # plt.subplot(121),plt.imshow(img,cmap = 'gray')
# # plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# # # plt.subplot(122),plt.imshow(cropped,cmap = 'gray')
# # # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# # plt.show()


# # from PIL import Image
# # import numpy as np

# # def crop_image(image_path, output_path):
# #     # Load the image
# #     img = Image.open(image_path).convert('RGB')
# #     img_array = np.array(img, dtype=np.uint8)

# #     # Define a tolerance level for color comparison
# #     tol = 10

# #     # Find the non-white pixels
# #     non_white_pixels = np.any(np.abs(img_array - [255, 255, 255]) > tol, axis=-1)

# #     # Find the coordinates of the non-white pixels
# #     coords = np.argwhere(non_white_pixels)

# #     if coords.size == 0:
# #         print("No non-white pixels found, nothing to crop.")
# #         return

# #     # Get the bounding box
# #     x_min, y_min = coords.min(axis=0)[:2]
# #     x_max, y_max = coords.max(axis=0)[:2]

# #     # Crop the image using the bounding box
# #     cropped_img = img.crop((y_min, x_min, y_max, x_max))

# #     # Save the cropped image
# #     cropped_img.save(output_path)

# # # Example usage
# # crop_image('/Users/viglud/QWOP/test.png', '/Users/viglud/QWOP/result.png')


# import cv2
# import numpy as np

# def find_green_area(image):
#     # Convert image to RGB (OpenCV uses BGR by default)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Define the lower and upper bounds for the specific green color
#     lower_bound = np.array([39 - 10, 92 - 10, 44 - 10])
#     upper_bound = np.array([39 + 10, 92 + 10, 44 + 10])
    
#     # Find the green area
#     mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
    
#     # Find the coordinates of the green area
#     coords = np.argwhere(mask)
    
#     if coords.size == 0:
#         print("No matching green pixels found.")
#         return 0, 0

#     return coords.min(axis=0)[0], coords.max(axis=0)[0]

# def find_gray_area(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Threshold the image
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
#     # Find the coordinates of the non-white pixels
#     coords = np.argwhere(thresh)
    
#     return coords.min(axis=0)[0], coords.max(axis=0)[0]

# def main():
#     # Read the image
#     image = cv2.imread('/Users/viglud/QWOP/test.png')
    
#     # Find and crop the green search bar area
#     green_top, green_bottom = find_green_area(image)
#     cropped_green = image[green_bottom+1:, :]
    
#     # Find and crop the gray area
#     gray_top, gray_bottom = find_gray_area(cropped_green)
#     final_crop = cropped_green[gray_top+1:gray_bottom, :]
    
#     # Save the cropped image
#     cv2.imwrite('/Users/viglud/QWOP/result.png', final_crop)

# if __name__ == '__main__':
#     main()

import pygetwindow as gw
import pyautogui
import cv2
import numpy as np

# Find the window by title or other criteria
target_window = gw.getWindowsWithTitle("Google Chrome ")[0]

# Get the window's position and size
left, top, width, height = target_window.left, target_window.top, target_window.width, target_window.height

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('recorded_window.avi', fourcc, 20.0, (width, height))

while True:
    # Capture the screen region corresponding to the window
    screenshot = pyautogui.screenshot(region=(left, top, width, height))

    # Convert the screenshot to a NumPy array
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Write the frame to the video file
    out.write(frame)

    # Exit the loop when the user presses a key (e.g., 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter and close the OpenCV window
out.release()
cv2.destroyAllWindows()

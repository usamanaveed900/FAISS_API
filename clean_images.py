import os
import cv2
import glob

def clean_image_data(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = glob.glob(os.path.join(input_folder, '*.jpg')) 

    for image_path in image_paths:
        image = cv2.imread(image_path)
        
        # Resize image to the target size and convert to RGB mode
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save the resized and normalized image to the output folder
        image_filename = os.path.basename(image_path)
        print(image_filename+' '+str(image.shape)+' '+str(image.size)+'\n')
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    input_folder = 'Datasets/images_fb/images'
    output_folder = 'Datasets/images_fb/images_cleaned'
    target_size = (224, 224)

    clean_image_data(input_folder, output_folder, target_size)

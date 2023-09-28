import os
import cv2

# Function to convert PNG to JPG and preserve folder structure
def convert_png_to_jpg(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".png") or file.lower().endswith(".bmp"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
                output_path = os.path.splitext(output_path)[0] + ".jpg"
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Read and save PNG as JPG using OpenCV
                img = cv2.imread(input_path)
                cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # Adjust quality as needed

if __name__ == "__main__":
    # Replace with your input folder path
    # input_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\Mi3_Aligned"
    # input_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\S90_Aligned"
    # input_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\T3i_Aligned"
    input_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\SIDD_Medium_Srgb"
    # Output folder for converted JPGs
    # output_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\convertidas\\Mi3_Aligned"
    # output_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\convertidas\\S90_Aligned"
    # output_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\convertidas\\T3i_Aligned"
    output_folder = "D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\convertidas\\SIDD_Medium_Srgb"

    convert_png_to_jpg(input_folder, output_folder)
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import RRDBNet_arch as arch
import matplotlib.pyplot as plt

#alter device to CUDA or other GPU if you have such resources available.
def initialize_esrgan_model(model_path='models/RRDB_ESRGAN_x4.pth', device='cpu'):
    # Initialize the ESRGAN model architecture
    esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    # Load pre-trained weights
    esrgan_model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    # Set the model to evaluation mode
    esrgan_model.eval()
    # Move the model to the specified device (CPU/GPU)
    esrgan_model.to(device)
    return esrgan_model

def preprocess_image(img_path, device='cpu'):
    # Read LR image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    # Additional preprocessing (e.g., Gaussian blur for feature extraction)
    img_preprocessed = cv2.GaussianBlur(img, (3, 3), 0)
    # Convert image to PyTorch tensor and adjust dimensions
    img = torch.from_numpy(np.transpose(img_preprocessed[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    return img_LR, img_preprocessed, img

# Function to read LR image without preprocessing
def read_image(img_path, device='cpu'):
    # Read LR image without any preprocessing
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    # Convert image to PyTorch tensor and adjust dimensions
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    return img_LR, img

# Function to enhance LR image using ESRGAN model
def enhance_image(img_LR, esrgan_model):
    # Perform enhancement using the ESRGAN model
    with torch.no_grad():
        output_esrgan = esrgan_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    return output_esrgan

def resize_and_calculate_ssim(output_esrgan, img, win_size=3, data_range=1.0):
    # Resize output_esrgan to match the dimensions of img
    output_esrgan_resized = cv2.resize(output_esrgan.transpose(1, 2, 0), (img.shape[2], img.shape[1]))

    # Calculate SSIM after ESRGAN prediction
    ssim_value = ssim(output_esrgan_resized, img.squeeze().permute(1, 2, 0).cpu().numpy(),
                      win_size=win_size, data_range=data_range)
    return ssim_value

def main():
    esrgan_model = initialize_esrgan_model()
    test_img_folder = 'LR/*' #folder storing the test images

    print('Models initialized. \nTesting...')

    #arrays for storing the obtained SSIM values, to help in plotting them later
    ssim_values_with_preprocessing = []
    average_ssim_values_with_preprocessing = 0
    ssim_values_without_preprocessing = []
    average_ssim_values_without_preprocessing = 0

    for idx, path in enumerate(glob.glob(test_img_folder)):
        base = osp.splitext(osp.basename(path))[0]
        print(idx + 1, base)

        # With preprocessing
        img_LR, img_preprocessed, img = preprocess_image(path)
        output_esrgan_with_preprocessing = enhance_image(img_LR, esrgan_model)
        ssim_with_preprocessing = resize_and_calculate_ssim(output_esrgan_with_preprocessing, img)
        ssim_values_with_preprocessing.append(ssim_with_preprocessing)

        # Without preprocessing
        img_LR, img = read_image(path)  # Use the same LR image without preprocessing
        output_esrgan_without_preprocessing = enhance_image(img_LR, esrgan_model)
        ssim_without_preprocessing = resize_and_calculate_ssim(output_esrgan_without_preprocessing, img)
        ssim_values_without_preprocessing.append(ssim_without_preprocessing)

        cv2.imwrite('results/{:s}_rlt.png'.format(base), output_esrgan_with_preprocessing) #storing in the results folder

        print('SSIM for ESRGAN output (With Preprocessing):', round(ssim_with_preprocessing, 2))
        print('SSIM for ESRGAN output (Without Preprocessing):', round(ssim_without_preprocessing, 2))

        # Display images 
        plt.figure(figsize=(18, 6))
        
        #displaying the input image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')

        #displaying the output image without pre-processing
        plt.subplot(2, 3, 2)
        plt.imshow(output_esrgan_without_preprocessing.transpose(1, 2, 0))
        plt.title('ESRGAN Output (Without Preprocessing)')
        plt.axis('off')

        #displaying the output image after pre-processing
        plt.subplot(2, 3, 3)
        plt.imshow(output_esrgan_with_preprocessing.transpose(1, 2, 0))
        plt.title('ESRGAN Output (With Preprocessing)')
        plt.axis('off')

        plt.show()
    

    # Plotting
    x_values = list(range(1, 21))

    plt.figure(figsize=(10, 5))

    #graph showing the SSIMM vs image when passed without pre-processing
    plt.subplot(1, 2, 1)
    plt.plot(x_values, ssim_values_without_preprocessing, marker='o')
    plt.title('SSIM Without Preprocessing')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM Value')

    #graph showing the SSIMM vs image when passed after pre-processing
    plt.subplot(1, 2, 2)
    plt.plot(x_values, ssim_values_with_preprocessing, marker='o', color='orange')
    plt.title('SSIM With Preprocessing')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM Value')

    #graph showing the comparitive analysis of both of cases together
    plt.subplot(1, 3, 3)
    plt.plot(x_values, ssim_values_without_preprocessing, marker='o', label='Without Preprocessing')
    plt.plot(x_values, ssim_values_with_preprocessing, marker='o', color='orange', label='With Preprocessing')
    plt.title('SSIM Comparison')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()


    #calculating the average SISM value after running through the whole dataset
    average_ssim_values_with_preprocessing = sum(ssim_values_with_preprocessing)/len(ssim_values_with_preprocessing)
    average_ssim_values_without_preprocessing = sum(ssim_values_without_preprocessing)/len(ssim_values_without_preprocessing)

    print('Testing completed.')
    print('Average SSIM values (With Preprocessing):', average_ssim_values_with_preprocessing)
    print('Average SSIM values (Without Preprocessing):', average_ssim_values_without_preprocessing)


if __name__ == "__main__":
    main()

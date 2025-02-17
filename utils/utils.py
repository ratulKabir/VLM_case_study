import matplotlib.pyplot as plt
import cv2
import os

def save_image_qa_plot(image_path, question, answer, save_path='./results/output_plot.jpg'):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 2]})
    
    # Display image on the left
    axes[0].imshow(image)
    axes[0].axis("off")
    
    # Display question and answer on the right
    axes[1].text(0.5, 0.6, f"Q: {question}", fontsize=12, ha='center', wrap=True)
    axes[1].text(0.5, 0.4, f"A: {answer}", fontsize=12, ha='center', wrap=True)
    axes[1].axis("off")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# Example usage:
# save_image_qa_plot("/path/to/image.jpg", "What is in the image?", "An excavator.", "output_plot.jpg")
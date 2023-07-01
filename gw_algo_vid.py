import cv2
import numpy as np

def underwater_video_enhancement(video_path):
    """Enhances an underwater video to remove water and restore original colors.

    Args:
        video_path: The path to the input underwater video.

    Returns:
        None
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to save the enhanced video
    output_path = video_path[:-4] + '_enhanced.mp4'
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process each frame of the video
    for frame_index in range(total_frames):
        ret, frame = video.read()

        if not ret:
            break

        # Apply underwater image enhancement to the current frame
        enhanced_frame = underwater_image_enhancement(frame)

        # Write the enhanced frame to the output video
        output_video.write(enhanced_frame)

        # Display the progress
        print(f'Processed frame {frame_index + 1}/{total_frames}')

    # Release the video capture and writer objects
    video.release()
    output_video.release()

    print('Video processing completed.')

def underwater_image_enhancement(image, gamma=0.5):
    """Enhances an underwater image to remove water and restore original colors.

    Args:
        image: The input underwater image.

    Returns:
        The enhanced image with removed water and restored colors.
    """

    # Convert image to float32 format
    image = image.astype(np.float32) / 255.0

    # Compute the average values for each color channel
    avg_channel_values = np.mean(image, axis=(0, 1))

    # Compute the correction factors for each color channel
    correction_factors = np.divide(avg_channel_values, np.max(avg_channel_values))

    # Apply the correction factors to each pixel
    enhanced_image = np.divide(image, correction_factors)

    # Clip the pixel values to the valid range [0, 1] and convert back to uint8
    enhanced_image = np.clip(enhanced_image, 0, 1.0)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image

def main():
    # Path to the input underwater video
    video_path = r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\videos\new_vid.mp4'

    # Enhance the underwater video
    underwater_video_enhancement(video_path)

if __name__ == '__main__':
    main()

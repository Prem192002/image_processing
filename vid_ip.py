import cv2
import numpy as np

def dehaze_adaptive_histogram_equalization(image):
    """Dehazes an image using adaptive histogram equalization.

    Args:
        image: The input image.

    Returns:
        The dehazed image.
    """

    # Equalize the histogram of each channel of the image.
    for channel in range(3):
        image[:, :, channel] = cv2.equalizeHist(image[:, :, channel])

    return image

def main():
    # Open the video capture
    cap = cv2.VideoCapture(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\videos\new_vid.m4v.mp4')

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer to save the dehazed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\videos\output_video3.mp4', fourcc, fps, (width, height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Apply dehazing to the frame
        dehazed_frame = dehaze_adaptive_histogram_equalization(frame)

        # Save the dehazed frame
        out.write(dehazed_frame)

        # Display the dehazed frame
        cv2.imshow('Dehazed Frame', dehazed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

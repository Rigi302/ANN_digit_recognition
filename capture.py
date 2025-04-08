import cv2
import numpy as np
import neuralNetwork

# build and load
new_network = neuralNetwork.neuralnetwork(784, 100, 10, 0.2)
new_network.load_data()

# open the camera
cap = cv2.VideoCapture(0)

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break


        # change into b-g
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # resized the frame into 28*28
        resized_frame_28x28 = cv2.resize(gray_frame, (28, 28))


        pixel_data = resized_frame_28x28.flatten()

        # use the instance to check the output
        final_matrix = new_network.query((np.asarray(pixel_data, dtype=float) / 255.0 * 0.99) + 0.01)
        test_value = max(final_matrix)
        final_matrix = final_matrix.flatten()
        final_index = np.argwhere(final_matrix == test_value)
        index_final = int(final_index[0, 0])

        frame_with_recognition = new_network.show_output(index_final,frame)
        cv2.imshow('result frame',frame_with_recognition)

        # press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
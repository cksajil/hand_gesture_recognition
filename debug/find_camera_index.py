import cv2

def find_camera_indices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        else:
            cap.release()
            break
        index += 1
    return arr

if __name__ == "__main__":
    camera_indices = find_camera_indices()
    if camera_indices:
        print(f"Available camera indices: {camera_indices}")
    else:
        print("No cameras found.")

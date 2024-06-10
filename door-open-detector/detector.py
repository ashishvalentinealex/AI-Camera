import cv2
from ultralytics import YOLO

import threading

import requests
import base64

import uuid

def generate_unique_filename(prefix='', suffix=''):
    unique_id = str(uuid.uuid4())  # Generate a UUID
    unique_filename = f"{prefix}{unique_id}{suffix}"
    return unique_filename

def call_api(url, data):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Make a POST request with JSON data
        response = requests.post(url, json=data, headers=headers, verify=False)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse JSON response
            json_response = response.json()
            return json_response
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def file_to_base64(file_path):
    try:
        # Open the file in binary mode
        with open(file_path, "rb") as file:
            # Read the file contents
            file_contents = file.read()

            # Encode the file contents as base64
            base64_data = base64.b64encode(file_contents)
            return base64_data.decode('utf-8')  # Decode bytes to string
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Output folder where the image will be saved
output_folder = "output_images"

# File name for the saved image (without extension)
file_name = "frame_"
def save_frame_as_image(frame, output_folder, file_name, inOrOut, image_format='png'):
    # Construct the output file path
    unique_file_name = generate_unique_filename(prefix=file_name + inOrOut + '_', suffix='.' + image_format)
    output_path = f"{output_folder}/{unique_file_name}"

    # Write the frame as an image
    success = cv2.imwrite(output_path, frame)

    if success:
        print(f"Frame saved as {output_path}")
    else:
        print("Failed to save frame as image")

    file_path = output_path
    base64_data = file_to_base64(file_path)
    if base64_data:
        print("Base64 encoded data:")
        # print(base64_data)
    else:
        print("Failed to read the file or convert to base64.")


    api_url = "https://3.7.97.129:9444/ADInterface/services/rest/pipra_customservice/createNotice"
    request_data = {
        "CreateNoticeRequest": {
            "serviceType": "createNotice",
            "message": "Door Opened" if inOrOut == 'Open' else "Door Closed",
            "userId": 1000019,
            "reference": None,
            "textMessage":  "Door Opened" if inOrOut == 'Open' else "Door Closed",
            "description": "Door Opened" if inOrOut == 'Open' else "Door Closed",
            "ADLoginRequest": {
                "user": "santosh",
                "pass": "santosh",
                "lang": "112",
                "ClientID": 1000002,
                "RoleID": 1000011,
                "OrgID": 1000004,
                "WarehouseID": 1000009,
                "stage": "0"
            },
            "fileName": unique_file_name,
            "fileData": base64_data
        }
    }

    # Call the API
    response_data = call_api(api_url, request_data)

    # Process the response
    if response_data:
        print("Response received:")
        print(response_data)
    else:
        print("Failed to receive response from the API.")

def annotate(video_path, output_video_path, model_path):
    # Initialize YOLO model
    model = YOLO(model_path)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (frame_width, frame_height))
    
    # Run prediction on the entire video
    conf = 0.5  # Confidence threshold
    results = model.predict(video_path, conf=conf)

    door_status = "Unknown"
    
    # Process each frame based on the detection results
    for idx in range(len(results)):
        ret, frame = cap.read()
        if not ret:
            break
    
        # Check if the 'Strip' class (class ID 0) is detected in the current frame
        detections = results[idx].boxes
        strip_detected = any(detection.cls == 0 for detection in detections)
    
        # Determine message based on the presence of the 'Strip' class
        message = "Open" if strip_detected else "Closed"
    
        # Annotate the frame
        color = (0, 255, 0) if message == "Open" else (0, 0, 255)
        cv2.putText(frame, message, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

        if (door_status != message):
            print('Door status changed to: ' + message)
            t1 = threading.Thread(target=save_frame_as_image, args=(frame, output_folder, file_name, message, 'png',))
            t1.start()
            t1.join()

            door_status = message

    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    video_path = "./shutter-open-close.mp4"
    output_video_path = "./annotated-output.mp4"
    model_path = "./CustomDoorDetection.pt"
    annotate(video_path, output_video_path, model_path)

if __name__ == "__main__":
    main()

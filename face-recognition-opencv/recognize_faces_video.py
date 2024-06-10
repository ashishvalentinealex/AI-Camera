# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

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
            "message": "Unauthorised person",
            "userId": 1000019,
            "reference": None,
            "textMessage":  "Unauthorised person",
            "description": "Unauthorised person",
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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
		#if Unknown is found in the frame, push the notification
		if (name == 'Unknown'):
			t1 = threading.Thread(target=save_frame_as_image, args=(frame, output_folder, file_name, 'out', 'png',))
			t1.start()
			t1.join()

	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces t odisk
	if writer is not None:
		writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()
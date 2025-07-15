# Fire and Smoke Detection Using AI

This project focuses on building a simple and effective system for detecting fire and smoke
The goal is to enable automatic detection that can help in taking fast action and minimizing damage.

## Project Details
- I experimented with multiple model versions to evaluate and compare their performance.
- Several techniques were tested to improve accuracy and reliability.
- All the results, observations, and comparisons are documented in the attached PDF file.
- ## Notebook here [ Google Colab notebook-fire and smoke Detection](https://colab.research.google.com/drive/1orHAleKdz2HGamDQeyl6gKzknB3tADHn?usp=sharing)

## AI-Docs

The PDF includes:
- A summary of all the versions I tested  
- Visual examples and output comparisons  
 [Download the full report](Docs/AI-part.pdf)


# AI-Based Fire & Smoke Detection API

This part contains a FastAPI application integrated with a YOLO model for fire and smoke detection,
connected to Firebase for sensor data.

Below are instructions on how to test the model, run the API, and open the Swagger UI
---
### 1. Install Dependencies
install required packages using:

bash
pip install -r requirements.txt


### 3. Run the Server
Start the FastAPI server using:

bash
uvicorn main:app --reload

The API will be available at:
http://127.0.0.1:8000

Interactive API docs (Swagger UI):
http://127.0.0.1:8000/docs

### How to Test the Model
Use the /predict/ POST endpoint on Swagger UI or via curl/postman to send an image file  or video.
The response will contain detected objects such as fire or smoke with their confidence scores.

### Camera & Firebase Integration
Once you access the endpoint (/set-camera-url/)and provide the IP address of the camera, 
the system will Start listening to Firebase for real-time sensor data.

If the received values exceed a defined threshold, the system will Automatically activate the camera stream using the provided IP.
The camera will continue running
&
if you want to stop it call:
(/stop-stream/) endpoint to terminate the stream.


### Notes
All detections are logged into fire_detections.json

The last detection frame is saved as last_detected_frame.jpg


### Video Demo
[*Fire & Smoke Detection Demo Video*](https://drive.google.com/file/d/1zXrvtEXBHXu9cwNwGDzt7nQhF6tF2L5q/view?usp=sharing)

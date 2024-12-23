import cv2
import torch
from PIL import Image
import heapq
import boto3
import json
import threading
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Object Detection Module (Perception) ---
class ObjectDetection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def detect_objects(self, frame):
        img = Image.fromarray(frame)
        results = self.model(img)
        return results.render()[0], results.pandas().xyxy[0]

# --- Path Planning Module ---
class PathPlanning:
    def __init__(self, grid):
        self.grid = grid

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g_score = g_score[current] + 1

                if 0 <= neighbor[0] < len(self.grid) and 0 <= neighbor[1] < len(self.grid[0]) and self.grid[neighbor[0]][neighbor[1]] != 1:
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

# --- Cloud Synchronization Module ---
class CloudSync:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_data(self, data, file_name):
        try:
            with open(file_name, "w") as f:
                json.dump(data, f)
            self.s3.upload_file(file_name, self.bucket_name, file_name)
            logging.info(f"Data uploaded to {self.bucket_name}/{file_name}")
        except Exception as e:
            logging.error(f"Error uploading data: {e}")

# --- Autonomous Driving Integration ---
class DriverXAI:
    def __init__(self, grid, bucket_name):
        self.object_detection = ObjectDetection()
        self.path_planning = PathPlanning(grid)
        self.cloud_sync = CloudSync(bucket_name)

    def process_frame(self, frame):
        processed_frame, detected_objects = self.object_detection.detect_objects(frame)
        return processed_frame, detected_objects

    def find_route(self, start, goal):
        return self.path_planning.astar(start, goal)

    def upload_route_to_cloud(self, route):
        self.cloud_sync.upload_data({"route": route}, "route_data.json")

    def upload_detected_objects(self, detected_objects):
        self.cloud_sync.upload_data(detected_objects.to_dict(), "detected_objects.json")

    def overlay_path(self, frame, path):
        for point in path:
            cv2.circle(frame, (point[1] * 50 + 25, point[0] * 50 + 25), 10, (0, 255, 0), -1)
        return frame

# --- Main Program ---
if __name__ == "__main__":
    # Grid for Path Planning (1 = obstacle, 0 = drivable)
    grid = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    ]
    
    # Initialize DriverX AI
    driverx = DriverXAI(grid, "driverx-routes")

    # Start Video Capture
    cap = cv2.VideoCapture(0)  # Replace with video file path for testing
    start = (0, 0)
    goal = (3, 3)

    # Calculate route in a separate thread
    threading.Thread(target=lambda: driverx.upload_route_to_cloud(driverx.find_route(start, goal))).start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for object detection
        processed_frame, detected_objects = driverx.process_frame(frame)

        # Overlay planned route on frame
        route = driverx.find_route(start, goal)
        if route:
            processed_frame = driverx.overlay_path(processed_frame, route)

        # Display detected objects and confidence scores
        for _, row in detected_objects.iterrows():
            label = f"{row['name']} ({row['confidence']:.2f})"
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Upload detected objects to cloud
        driverx.upload_detected_objects(detected_objects)

        # Display frame
        cv2.imshow("DriverX AI - Perception", processed_frame)
        logging.info("Detected Objects: %s", detected_objects)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
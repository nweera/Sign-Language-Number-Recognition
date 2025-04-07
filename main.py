import os
import cv2
import numpy as np
import mediapipe as mp
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
from collections import Counter

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)
        self.trackingCon = float(trackingCon)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 0:  # Highlight the wrist joint
                            cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        LmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myhand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myhand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    LmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return LmList

class NumberSignRecognizer:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.model = None
        # Increase detection confidence to reduce false positives
        self.detector = handDetector(detectionCon=0.8, maxHands=1)
        # For data augmentation
        self.augmentation_enabled = True
    
    def extract_hand_features(self, img):
        """Extract hand landmarks as features from an image with improved normalization"""
        # Find hands
        img = self.detector.findHands(img, draw=False)
        landmarks = self.detector.findPosition(img, draw=False)
        
        features = []
        if len(landmarks) >= 21:  # Ensure we have all landmarks for a complete hand
            # Get hand bounding box for better normalization
            min_x = min(lm[1] for lm in landmarks)
            min_y = min(lm[2] for lm in landmarks)
            max_x = max(lm[1] for lm in landmarks)
            max_y = max(lm[2] for lm in landmarks)
            
            # Calculate hand size for normalization
            hand_width = max(1, max_x - min_x)  # Avoid division by zero
            hand_height = max(1, max_y - min_y)
            
            # Compute angles between landmarks to make rotation-invariant features
            wrist = landmarks[0]  # Wrist landmark
            middle_mcp = landmarks[9]  # Middle finger MCP joint
            
            # Calculate the angle between wrist and middle finger base
            angle_rad = np.arctan2(middle_mcp[2] - wrist[2], middle_mcp[1] - wrist[1])
            angle_deg = np.degrees(angle_rad)
            
            # Add angle as a feature
            features.append(angle_deg / 360.0)  # Normalize angle
            
            # Add normalized distances between fingers
            for i in range(4, 21, 4):  # Tips of fingers (thumb, index, middle, ring, pinky)
                for j in range(4, 21, 4):
                    if i != j:
                        # Calculate normalized distance between fingertips
                        dist = np.sqrt((landmarks[i][1] - landmarks[j][1])**2 + 
                                     (landmarks[i][2] - landmarks[j][2])**2)
                        features.append(dist / max(hand_width, hand_height))
            
            # Add normalized positions relative to wrist
            for lm in landmarks:
                # Normalize by hand bounding box size
                features.append((lm[1] - wrist[1]) / hand_width)
                features.append((lm[2] - wrist[2]) / hand_height)
            
            # Calculate relative distances and angles between key landmarks
            # These features are more robust to variations in hand orientation
            for i in range(0, 21, 4):  # Key landmarks (wrist and fingertips)
                for j in range(i+4, 21, 4):
                    if j < 21:
                        # Relative distance
                        dist = np.sqrt((landmarks[i][1] - landmarks[j][1])**2 + 
                                     (landmarks[i][2] - landmarks[j][2])**2)
                        features.append(dist / max(hand_width, hand_height))
                        
                        # Relative angle
                        angle = np.arctan2(landmarks[j][2] - landmarks[i][2], 
                                         landmarks[j][1] - landmarks[i][1])
                        features.append(angle / (2 * np.pi))
        else:
            # If no complete hand is detected, return a zero vector
            # Using a larger feature vector size now
            features = [0] * 100
            
        # Ensure consistent feature vector length
        if len(features) < 100:
            features.extend([0] * (100 - len(features)))
        
        return np.array(features[:100])  # Ensure exact feature count
    
    def augment_image(self, img):
        """Apply augmentation to create variations of the input image"""
        augmented_images = [img]  # Start with the original image
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Rotation augmentation (slight rotations)
        for angle in [-15, -10, -5, 5, 10, 15]:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented_images.append(rotated)
        
        # Scale augmentation (slight zoom in/out)
        for scale in [0.9, 0.95, 1.05, 1.1]:
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            scaled = cv2.warpAffine(img, M, (w, h))
            augmented_images.append(scaled)
        
        # Brightness augmentation
        for alpha in [0.8, 0.9, 1.1, 1.2]:
            bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            augmented_images.append(bright)
        
        return augmented_images
    
    def load_dataset(self):
        """Load images from the dataset and extract features with augmentation"""
        X = []  # Features
        y = []  # Labels
        
        # Process each folder (0-9)
        for number in range(10):
            folder_path = os.path.join(self.dataset_path, str(number))
            if not os.path.exists(folder_path):
                print(f"Warning: Folder for number {number} not found at {folder_path}")
                continue
                
            print(f"Processing images for number {number}...")
            
            # Process each image in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    
                    # Apply data augmentation if enabled
                    if self.augmentation_enabled:
                        augmented_images = self.augment_image(img)
                    else:
                        augmented_images = [img]  # Just use the original image
                    
                    # Process each augmented image
                    for aug_img in augmented_images:
                        # Extract features
                        features = self.extract_hand_features(aug_img)
                        
                        # Only add if valid features were extracted
                        if not np.all(features == 0):
                            X.append(features)
                            y.append(number)
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train a machine learning model on the dataset"""
        print("Loading dataset...")
        X, y = self.load_dataset()
        
        if len(X) == 0:
            print("Error: No valid data found. Check your dataset path.")
            return False
        
        print(f"Dataset loaded: {len(X)} samples with {len(X[0])} features each")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Random Forest classifier with more estimators for better accuracy
        print("Training model...")
        self.model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        
        # Check for overfitting
        if train_accuracy - test_accuracy > 0.2:
            print("Warning: Model may be overfitting. Consider adding more diverse training data.")
        
        # Save the trained model
        with open('number_sign_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved to number_sign_model.pkl")
        
        return True
    
    def load_model(self, model_path='number_sign_model.pkl'):
        """Load a previously trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Train a model first.")
            return False
    
    def predict_from_image(self, img):
        """Predict the number from an image"""
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return None
        
        features = self.extract_hand_features(img)
        
        # If no valid hand features were detected
        if np.all(features == 0):
            return None
            
        # Make prediction
        prediction = self.model.predict([features])[0]
        confidence = np.max(self.model.predict_proba([features]))
        
        return prediction, confidence
    
    
    def run_webcam_detection(self):
        """Run real-time detection using webcam with improved stability"""
        if self.model is None:
            if not self.load_model():
                print("Training new model...")
                if not self.train_model():
                    print("Failed to train model. Check your dataset.")
                    return

        cap = cv2.VideoCapture(0)

        # Adjust camera settings for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus

        pTime = 0

        # For tracking stability of predictions - increased history length
        prediction_history = []
        max_history = 10

        # Minimum confidence threshold to accept a prediction
        min_confidence = 0.3

        print("\nWebcam detection started.")
        print("Controls:")
        print("- S: Take a snapshot and save")
        print("- Q: Quit")

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to get frame from webcam")
                break

            # Mirror the image horizontally to make it more intuitive
            img = cv2.flip(img, 1)

            # Process the whole image
            img = self.detector.findHands(img)
            landmarks = self.detector.findPosition(img, draw=True)

            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # Show FPS
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Make prediction if hand is detected
            if landmarks and len(landmarks) >= 15:  # Ensure enough landmarks are detected
                result = self.predict_from_image(img)
                if result is not None:
                    number, confidence = result

                    # Only consider predictions with reasonable confidence
                    if confidence >= min_confidence:
                        # Add to history for stability
                        prediction_history.append(number)
                        if len(prediction_history) > max_history:
                            prediction_history.pop(0)

                    # Get most common prediction from history
                    if prediction_history:
                        counter = Counter(prediction_history)
                        stable_prediction = counter.most_common(1)[0][0]
                        stability = counter[stable_prediction] / len(prediction_history)

                        # Display prediction and confidence
                        cv2.putText(img, f"Number: {stable_prediction}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.putText(img, f"Conf: {confidence:.2f}", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, f"Stability: {stability:.2f}", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Clear history when no hand is detected for a while
                if len(prediction_history) > 0:
                    # Keep the history for a bit to prevent flickering
                    if len(prediction_history) > max_history // 2:
                        prediction_history.pop(0)

            # Display instructions
            cv2.putText(img, "S: Snapshot | Q: Quit",
                        (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Number Sign Detection", img)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Take a snapshot and save
                timestamp = int(time.time() * 1000)
                snapshot_filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(snapshot_filename, img)
                print(f"Snapshot saved to {snapshot_filename}")

        cap.release()
        cv2.destroyAllWindows()



    def evaluate_model_with_visualization(self):
        """Evaluate model performance with confusion matrix visualization"""
        if self.model is None:
            if not self.load_model():
                print("No model available. Train a model first.")
                return
                
        # Load the dataset
        X, y = self.load_dataset()
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y, y_pred)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = range(10)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Number')
        plt.xlabel('Predicted Number')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        

def main():
    print("Number Sign Language Recognition System")
    print("=====================================")
    
    # Set path to your dataset - update this to your path
    dataset_path = r"C:\Users\user\Desktop\sign numbers dataset"
    
    recognizer = NumberSignRecognizer(dataset_path)
    
    while True:
        print("\nMenu:")
        print("1. Train new model")
        print("2. Run webcam detection")
        print("3. Evaluate model")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            recognizer.train_model()
        elif choice == '2':
            recognizer.run_webcam_detection()
        elif choice == '3':
            recognizer.evaluate_model_with_visualization()
        elif choice == '4':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
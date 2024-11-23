## **Calculating the Average Linear Speed of the International Space Station (ISS)**

### **Overview**
This script calculates the average linear speed of the International Space Station (ISS) using two independent approaches:
1. **Image-Based Displacement Calculation**
2. **Geographical Coordinates-Based Calculation**

By combining these methods, the script produces a robust and accurate estimate of the ISS's velocity, saved to a file named `result.txt`.

---

### **Requirements**
#### **Hardware:**
- Raspberry Pi with a High-Quality Camera Module.
- Sense HAT (optional, but compliant with Astro Pi competition requirements).

#### **Software:**
- Python 3.8+
- Required Python libraries: 
  - `picamera`
  - `orbit`
  - `time`
  - `exif`
  - `datetime`
  - `pathlib`
  - `decimal`
  - `cv2` (OpenCV)
  - `math`
  - `statistics`
  - `numpy`

---

### **How It Works**
#### **Method 1: Image-Based Calculation**
1. **Capture Images**: 
   - The script uses the PiCamera to take sequential images at fixed intervals.
2. **Feature Matching**: 
   - Using the SIFT algorithm, the script identifies and matches keypoints between pairs of consecutive images.
3. **Pixel Displacement to Kilometers**: 
   - The script calculates the mean pixel displacement and converts it to kilometers using the camera's Ground Sample Distance (GSD) derived from the ISS's elevation.
4. **Velocity Calculation**: 
   - The velocity is computed based on the displacement in kilometers and the time interval between images.

#### **Method 2: Coordinates-Based Calculation**
1. **Geographical Coordinates**:
   - The script uses the Skyfield library to retrieve the ISS's geographical coordinates (longitude and latitude) at the moment each photo is taken.
2. **Haversine Formula**:
   - This formula computes the great-circle distance between two points on Earth.
3. **Velocity Calculation**:
   - The script calculates the velocity by dividing the great-circle distance by the time interval.

#### **Final Velocity**
- The script takes the median of velocities calculated by both methods and writes the final value to `result.txt` with up to 5 significant digits.

---

### **Usage Instructions**
1. **Setup**:
   - Install the required Python libraries using `pip`.
   - Connect the Raspberry Pi HQ Camera to the Raspberry Pi.
2. **Run the Script**:
   - Execute the script on the Raspberry Pi. The script runs for approximately 9 minutes, capturing data and performing calculations.
3. **Output**:
   - **Result**: The average ISS velocity is saved in `result.txt`.
   - **Logs**: Intermediate data (timestamps, elevations, distances, velocities, etc.) are saved in `data.txt` for additional analysis.

---

### **Important Notes**
- **Astro Pi Compliance**:
  - This script adheres to the Mission Space Lab Rulebook requirements by using the camera as a primary data source and incorporating additional geographical data for increased accuracy.
- **Acknowledgment**:
  - Permission for this dual-method approach was granted by Astro Pi Mission Control (see email correspondence in the script comments).

---

### **File Descriptions**
- **`main.py`**:
  - The main script for capturing photos, processing data, and calculating the ISS velocity.
- **`result.txt`**:
  - Contains the final computed ISS velocity.
- **`data.txt`**:
  - Stores intermediate data for debugging and verification purposes.
- **Captured Images**:
  - Photos taken by the PiCamera are saved in the same directory as the script.

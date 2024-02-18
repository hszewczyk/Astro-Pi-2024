"""
The approach to solving the problem is by using two methods of calculating the average linear speed of the International Space Station. 

First way:
Firstly, taking photos is needed to grab key points and descriptors of each one. Iterating through each pair of photos makes it
possible to calculate mean displacement - displacement in pixels between two photos. Having information about elevation above the earth 
and Raspberry Pi High Quality Camera specifications we were able to calculate ground sample distance of each photo and transform
displacement from pixels to kilometres. After grabbing metadata from each photo we can store the time difference between the two photos and
calculate the average linear speed.

Second way:
At the moment of taking the photo, we are acquiring the longitude and latitude of the ISS using the skyfield library. After conversion to 
radians we are able to use the haversine formula for spherical geometry and calculate the great-circle distance between two points on Earth. 
For the Earth haversine formula cannot be guaranteed correct to better than 0.5% but in our calculations, the error is negligibly small.

Lastly, we calculate the median of the velocities we received and save it to the file named 'result.txt' without exceeding the limit of 
5 significant digits.

One of the rules in the Mission Space Lab Rulebook is 'Your program uses at least one Sense HAT sensor or the camera.'. Our approach is to
not only use the camera but also calculations based on geographical coordinates. Since the camera will not be fully responsible for the
result, we asked Astro-Pi and that was the response:

henryk.szewczyk09@gmail.com to enquiries@astro-pi.org 4 February 2024:
Good morning,
My team and I have just completed writing a script to perform calculations that in the end save the average linear speed of the ISS to the 
result.txt file. We used a camera and after taking photos for every X seconds we matched features and calculated speed. Since it is not a 
very precise method (because of the possibility of having a cloudy environment and the fact that clouds are moving with the wind we can't 
consider it not knowing the speed and direction of the wind at specific locations and times) and we still have 2 weeks since the end of 
the competition we thought about seconds approach and calculate the mean of the two approaches. One of the allowed libraries is Skyfield 
which can return the actual position of the ISS (longitude and latitude). Are we allowed to use geographical coordinates and calculate the 
great-circle distance using the 'Haversine formula' or 'Thaddeus Vincenty formula'? After that, we would calculate the mean of the speed 
based on photos and coordinates. One of the points in the rulebook is 'Your program uses at least one Sense HAT sensor or the camera.'. 
We will be using the camera but it won't be fully responsible for the final result.

Thank you very much,
Henryk Szewczyk

enquiries@astro-pi.org to henryk.szewczyk09@gmail.com 7 February 2024:
Hi Henryk,

That would be allowed, absolutely. Also, as you have the opportunity to run code on the ISS, you could also gather data from other sensors, 
too. Even if you don't use the data to calculate the speed, you can download it and use it later. You may be able to run other experiments 
on the data you gather, even if you haven't had the idea yet. Make the most of the ten minutes you have!

Kind regards,

Astro Pi Mission Control


Based on that e-mails we acknowledged that our approach is fully legitimate and will not raise concerns when testing.
"""

from picamera import PiCamera
from orbit import ISS
from time import sleep
from exif import Image
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal, getcontext
import cv2
import math
import statistics
import numpy as np

# Get time of start of the program
start_time = datetime.now() 
now_time = datetime.now()

number_photos = 25 # number of photos to be taken
time_interval = 10 # time interval between taking photos

# PiCamera resolution
res_width = 4056 
res_height = 3040

base_folder = Path(__file__).parent.resolve()

# Define necessary variables
elevations = []
velocities1 = []
velocities2 = []
longs = []
lats = []

# To express earth radius at any point we use authalic radius (in metres) which is the radius of a hypothetical perfect sphere that 
# has the same surface area as the reference ellipsoid.
mean_radius = 6371007.18

cam = PiCamera()
cam.resolution = (res_width, res_height)

def measure_time():
    """
    Measure the elapsed time since the script started
    Time limit for the experiment is 10 minutes (600 seconds) but 60 seconds is needed to properly 
    save data to the result file and close the script. 

    Returns:
        bool: True if less than 9 minutes have passed since the script started.
    """
    elapsed_time = datetime.now() - start_time
    return elapsed_time < timedelta(seconds = 540)

def take_photo(number):
    """
    Capture a photo using the PiCamera and gather ISS coordinates.

    Args:
        number (int): The photo number used in the image filename.
    """
    # Capture an image using the PiCamera and save it with the specified filename
    cam.capture(f"{base_folder}/img{number}.jpg")

    # Obtain International Space Station (ISS) coordinates
    elevation, longitude, latitude = get_iss_coordinates()

    # Append obtained coordinates to global lists
    elevations.append(elevation)
    longs.append(longitude)
    lats.append(latitude)

    # Pause script execution for the specified time interval
    sleep(time_interval)

def get_iss_coordinates():
    """
    Retrieve International Space Station (ISS) coordinates.

    Returns:
        Tuple[float, float, float]: Tuple containing ISS elevation, longitude, and latitude.
    """
    iss = ISS()
    point = iss.coordinates()
    return point.elevation.m, point.longitude.degrees, point.latitude.degrees

def get_time(image):
    """
    Get the datetime from the metadata of a photo.

    Args:
        image (str): File path of the photo.

    Returns:
        datetime: Datetime when the photo was taken.
    """
    # Open the photo and extract metadata
    with open(image, 'rb') as img_file:
        img = Image(img_file)
        
        # Get the original datetime from the photo metadata
        time_str = img.get('datetime_original')
        
        # Parse the datetime string to match a specific format
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')

    return time


def get_time_diff(img1, img2):
    """
    Calculate the time difference between the timestamps of two photos.

    Args:
        img1 (str): File path for the first photo.
        img2 (str): File path for the second photo.

    Returns:
        float: Time difference in seconds.
    """
    # Get timestamps for the two photos
    time1 = get_time(img1)
    time2 = get_time(img2)

    # Calculate time difference in seconds
    time_diff = time2 - time1

    seconds = (
                time_diff.days * 24 * 60 * 60 * 1000 +
                time_diff.seconds * 1000 +
                time_diff.microseconds / 1000
                ) / 1000

    return seconds

def conv_to_cv(img1, img2):
    """
    Convert photos to OpenCV image objects.

    Args:
        img1 (str): File path of the first photo.
        img2 (str): File path of the second photo.

    Returns:
        tuple: Tuple containing OpenCV image objects for the two photos.
    """
    # Read the photos as color images using OpenCV
    img1_cv = cv2.imread(img1, 1)
    img2_cv = cv2.imread(img2, 1)

    return img1_cv, img2_cv

def calc_features(img1_cv, img2_cv):
    """
    Calculate features using the SIFT algorithm on two given photos.

    Args:
        img1_cv (numpy.ndarray): OpenCV grayscale image object for the first photo.
        img2_cv (numpy.ndarray): OpenCV grayscale image object for the second photo.

    Returns:
        tuple: Tuple containing keypoints and descriptors for the two photos.
    """
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Calculate keypoints and descriptors for the first photo
    keypts1, descr1 = sift.detectAndCompute(img1_cv, None)

    # Calculate keypoints and descriptors for the second photo
    keypts2, descr2 = sift.detectAndCompute(img2_cv, None)

    return keypts1, keypts2, descr1, descr2

def calc_matches(descr1, descr2):
    """
    Calculate good matches between two sets of descriptors using the brute-force method.

    Args:
        descr1 (numpy.ndarray): Descriptors for the first set of keypoints.
        descr2 (numpy.ndarray): Descriptors for the second set of keypoints.

    Returns:
        list: List of good matches.
    """
    # Create a brute-force matcher object
    brute_force = cv2.BFMatcher()

    # Match descriptors using the brute-force method
    matches = brute_force.knnMatch(descr1, descr2, k=2)

    # Filter good matches based on Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    return good_matches

def find_matching_coords(keypts1, keypts2, matches):
    """
    Find coordinates of displacement for matching keypoints.

    Args:
        keypts1 (list): List of keypoints for the first photo.
        keypts2 (list): List of keypoints for the second photo.
        matches (list): List of good matches.

    Returns:
        tuple: Two lists containing coordinates of displacement for the matched keypoints.
    """
    # Initialize lists to store coordinates
    coords1 = []
    coords2 = []

    # Iterate through every match
    for match in matches:
        # Get indices of keypoints in the two images
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Get coordinates of the matching keypoints
        (x1, y1) = keypts1[img1_idx].pt
        (x2, y2) = keypts2[img2_idx].pt

        # Append coordinates to the lists
        coords1.append((x1, y1))
        coords2.append((x2, y2))

    return coords1, coords2

def calc_mean_distance(coords1, coords2):
    """
    Calculate the mean distance of matches between two sets of coordinates.

    Args:
        coords1 (list): List of coordinates for the first set.
        coords2 (list): List of coordinates for the second set.

    Returns:
        float: Mean distance of matches.
    """
    # Merge two lists of coordinates together
    merged_coords = list(zip(coords1, coords2))

    # Initialize lists to store coordinate differences
    x_list = []
    y_list = []

    # Iterate through every element of the merged coordinates list
    for coord in merged_coords:
        # Calculate X and Y coordinate differences
        x_diff = coord[0][0] - coord[1][0]
        y_diff = coord[0][1] - coord[1][1]

        # Reject false matches where both X and Y differences are zero
        if x_diff == 0 and y_diff == 0:
            continue

        # Append differences to the lists
        x_list.append(x_diff)
        y_list.append(y_diff)

    # Calculate median X and Y coordinate differences only if list is not empty
    if not x_list:
        median_x = 0
    else:
        median_x = statistics.median(x_list)
    
    if not y_list:
        median_y = 0
    else:
        median_y = statistics.median(y_list)

    # Calculate multidimensional Euclidean distance
    distance = math.hypot(median_x, median_y)

    return distance

def calc_GSD(elevation):
    """
    Calculate the Ground Sample Distance (GSD) based on elevation.

    Args:
        elevation (float): Elevation above the ground level (in meters).

    Returns:
        float: Ground Sample Distance (GSD) in meters per pixel.
    """
    # Focal length of the 16mm C-mount Lens mounted to Raspberry Pi HQ Camera (in meters)
    focal_length = 0.004712
    
    # Sensor width of the Raspberry Pi HQ Camera (in meters)
    sensor_width = 0.006287
    
    # Calculate Ground Sample Distance (GSD) in meters per pixel
    GSD = sensor_width * elevation / res_width / focal_length

    return GSD

def calc_velocity(displacement, elevation, time_diff):
    """
    Calculate velocity based on displacement, elevation, and time difference.

    Args:
        displacement (float): Displacement between two points on the Earth's surface (in meters).
        elevation (float): Elevation above the ground level (in meters).
        time_diff (int): Time difference between two measurements (in seconds).

    Returns:
        float: Velocity in kilometers per second.
    """
    # Calculate the sine function of the triangle
    sina = 0.5 * displacement / mean_radius

    if -1 <= sina <= 1:  # Check if calculated sine is within valid range
        # Calculate the angle of the triangle
        angle = math.degrees(math.asin(sina))
        
        # The calculated angle is 2 times smaller than the final angle because only half of the triangle is used
        final_angle = 2 * angle

        # Calculate the length of the arc in the circle (Great Circle Distance)
        great_circle_distance = 2 * math.pi * (elevation + mean_radius) * final_angle / 360

        # Calculate velocity using the length of the arc and time difference
        velocity = great_circle_distance / time_diff / 1000
    else:
        # If sine value is invalid, calculate velocity using chord in a circle as displacement
        velocity = displacement / time_diff / 1000

    return velocity

def calc_speed(time_diff, dist, elevation, GSD):
    """
    Calculate the speed of the International Space Station (ISS).

    Args:
        time_diff (int): Time difference between two measurements (in seconds).
        dist (float): Distance traveled by the ISS in pixels.
        elevation (float): Elevation above the ground level (in meters).
        GSD (float): Ground Sample Distance (in meters per pixel).

    Returns:
        None
    """
    # Calculate displacement of the space station in meters
    displacement = dist * GSD

    # Calculate velocity based on displacement, elevation, and time difference
    velocity = calc_velocity(displacement, elevation, time_diff)

    # Append the calculated velocity to the list of velocities
    velocities1.append(velocity)

def calc_speed_geographical(elevation, time_diff, lat1, lat2, long1, long2):
    """
    Calculate the speed of the International Space Station (ISS) using geographical coordinates.

    Args:
        elevation (float): Elevation above the ground level (in meters).
        time_diff (int): Time difference between two measurements (in seconds).
        lat1 (float): Latitude of the first location (in degrees).
        lat2 (float): Latitude of the second location (in degrees).
        long1 (float): Longitude of the first location (in degrees).
        long2 (float): Longitude of the second location (in degrees).

    Returns:
        None
    """
    # Set precision for Decimal calculations
    getcontext().prec = 50

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    long1 = math.radians(long1)
    long2 = math.radians(long2)

    # Calculate the haversine formula value
    p = Decimal(math.pow(math.sin(0.5 * (lat2 - lat1)), 2)) + Decimal(Decimal(math.cos(lat1))
        * Decimal(math.cos(lat2)) * Decimal(math.pow(math.sin(0.5 * (long2 - long1)), 2)))

    if -math.pi / 2 < Decimal(math.asin(p)) < math.pi / 2:
        # Calculate the distance using the haversine formula
        d = 2 * (Decimal(mean_radius) + Decimal(elevation)) * Decimal(math.atan2(math.sqrt(p), math.sqrt(1 - p)))

        # Calculate velocity using the distance and time difference
        velocity2 = d / Decimal(time_diff) / 1000
    else:
        velocity2 = 0

    # Append the calculated velocity to the list of velocities
    velocities2.append(float(velocity2))

take_photo(0) # Take first photo

# Iterate through each pair of photos: 1 and 2; 2 and 3; 3 and 4; ... ; n-1 and n
for i in range(0, number_photos - 1):
    m_time = measure_time()

    if m_time: # Check if there is still availible time
        take_photo(i+1)

        # Define directories for the photos
        img1 = f"img{i}.jpg"
        img2 = f"img{i+1}.jpg"

        time_diff = get_time_diff(img1, img2)
        img1_cv, img2_cv = conv_to_cv(img1, img2)
        keypts1, keypts2, descr1, descr2 = calc_features(img1_cv, img2_cv)
        matches = calc_matches(descr1, descr2)
        coords1, coords2 = find_matching_coords(keypts1, keypts2, matches)
        dist = calc_mean_distance(coords1, coords2)
        GSD = calc_GSD(elevations[i])
        calc_speed(time_diff, dist, elevations[i], GSD)
        calc_speed_geographical(elevations[i], time_diff, lats[i], lats[i+1], longs[i], longs[i+1])

        # Save all gathered data to data.txt file for personal purpose
        data_file_path = base_folder / "data.txt"
        with open(data_file_path, "a", buffering = 1) as file:
            text = [f"time{i}: {time_diff}s \n", f"lat: {lats[i]} \n", f"long: {longs[i]} \n", f"elev: {elevations[i]} \n", f"dist: {dist} \n",
                    f"vel1: {velocities1[i]} \n", f"vel2: {velocities2[i]} \n"]
            file.writelines(text)
    # If there is no time left - exit the loop and perform needed calculations
    else:
        break

# Close the camera
cam.close()

# Calculate median velocities for the two approaches
iss_velocity1 = statistics.median(velocities1)
iss_velocity2 = statistics.median(velocities2)

# Determine the overall ISS velocity considering both approaches
if iss_velocity1 == 0 and iss_velocity2 == 0:
    iss_velocity = 0
elif iss_velocity1 == 0 or iss_velocity2 == 0:
    iss_velocity = iss_velocity1 + iss_velocity2
else:
    iss_velocity = 0.5 * (iss_velocity1 + iss_velocity2)

# Format the result with 4 decimal places
formatted_result = "{:.4f}".format(iss_velocity)

# Save the result to a text file
result_file_path = base_folder / "result.txt"
data_file_path = base_folder / "data.txt"
with open(result_file_path, "w", buffering = 1) as file:
    file.write(str(formatted_result))

# Save all gathered data to data.txt file for personal purpose
with open(data_file_path, "a", buffering = 1) as file:
    text = [f"vel1: {velocities1} \n", f"vel2: {velocities2} \n", f"iss_vel1: {iss_velocity1} \n", f"iss_vel2: {iss_velocity2} \n", 
            str(datetime.now()-start_time)]
    file.writelines(text)

# Exit the script
exit()
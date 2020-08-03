import math
import uuid
import cv2
import matplotlib as mpl
import random
import folium
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np
from selenium import webdriver


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        return cols, rows
    else:
        return None, None


def image_change(image, old_image):
    im1 = Image.fromarray(image)
    im2 = Image.fromarray(old_image)
    diff = ImageChops.difference(im2, im1)

    cols, rows = autocrop(np.array(diff))

    return cv2.rectangle(image.copy(), (rows[0], cols[0]), (rows[-1] + 1, cols[-1] + 1), (0, 255, 0), 2)


pi = math.pi
MOD = 0.0015


def location_pic(location, driver=None, zoom=20, tiles=None):
    filename = str(uuid.uuid4())
    if driver is None:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(
            'chromedriver', chrome_options=chrome_options)

    if tiles is None:
        world_map = folium.Map(
            location=location, zoom_start=zoom
        )
    else:
        world_map = folium.Map(
            location=location,  zoom_start=zoom, tiles=tiles)
    world_map.save("temp.html")
    driver .get("file:///content/temp.html")
    time.sleep(3)
    driver.save_screenshot(f'images/{filename}.png')
    return f'images/{filename}.png'


def get_geojson_grid(upper_right, lower_left, n=6):
    """Returns a grid of geojson rectangles, and computes the exposure in each section of the grid based on the vessel data.

    Parameters
    ----------
    upper_right: array_like
        The upper right hand corner of "grid of grids"s.

    lower_left: array_like
        The lower left hand corner of "grid of grids" s.

    n: integer
        The number of rows/columns in the (n,n) grid.

    Returns
    -------

    list
        List of "geojson style" dictionary objects   
    """

    all_boxes = []

    lat_steps = np.linspace(lower_left[0], upper_right[0], n+1)
    lon_steps = np.linspace(lower_left[1], upper_right[1], n+1)

    lat_stride = lat_steps[1] - lat_steps[0]
    lon_stride = lon_steps[1] - lon_steps[0]

    for lat in lat_steps[:-1]:
        for lon in lon_steps[:-1]:
            # Define dimensions of box in grid
            upper_left = [lon, lat + lat_stride]
            upper_right = [lon + lon_stride, lat + lat_stride]
            lower_right = [lon + lon_stride, lat]
            lower_left = [lon, lat]

            # Define json coordinates for polygon
            coordinates = [
                upper_left,
                upper_right,
                lower_right,
                lower_left,
                upper_left
            ]

            geo_json = {"type": "FeatureCollection",
                        "properties": {
                            "lower_left": lower_left[::-1],
                            "upper_right": upper_right[::-1]
                        },
                        "features": []}

            grid_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                }
            }

            geo_json["features"].append(grid_feature)

            all_boxes.append(geo_json)

    return all_boxes


def PointsInCircum(_x, y, r, n=100):
    ret = []
    for x in range(0, n+1):
        ret.append((
            _x+float(math.cos(2*pi/n*x)*r),
            y+float(math.sin(2*pi/n*x)*r)
        ))
    return ret


def all_grid(grid, centers):
    """
    Assign prob to all the grid
    """
    for _grid in grid:
        _grid["prob"] = assign_prob(_grid, centers)
        color = plt.cm.Greens(_grid["prob"])
        color = mpl.colors.to_hex(color)
        _grid["color"] = color
    return grid


def assign_prob(geo_json, centers):
    _points = geo_json["features"][0]["geometry"]["coordinates"][0][:-1]
    xx, yy = zip(*_points)
    centroid = (sum(xx) / len(_points), sum(yy) / len(_points))[::-1]
    prob = 0
    for center in centers:
        distance = distance_(centroid, center["center"])
        radius = center["rad_strips"]
        if distance <= radius[0]:
            prob += 0.5*center["trust"]
        elif distance >= radius[0] and distance <= radius[1]:
            prob += 0.3*center["trust"]
        elif distance >= radius[1] and distance <= radius[2]:
            prob += 0.17*center["trust"]
        else:
            prob += 0*center["trust"]

    return prob


def check_range_circle(distance, radius):
    """
    Check in which range of the radius does the distance lie in
    """
    if distance < radius:
        prob = 0
    elif distance > radius and distance < 2*radius:
        prob = 1
    elif distance > 2 * radius and distance < 3*radius:
        prob = 2
    else:
        prob = -1
    return prob


def distance_(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def displace(lat, lng, theta, distance):
    """
    Displace a LatLng theta degrees counterclockwise and some
    meters in that direction.
    Notes:
        http://www.movable-type.co.uk/scripts/latlong.html
        0 DEGREES IS THE VERTICAL Y AXIS! IMPORTANT!
    Args:
        theta:    A number in degrees.
        distance: A number in meters.
    Returns:
        A new LatLng.
    """
    theta = np.float32(theta)

    delta = np.divide(np.float32(distance), np.float32(6471))

    def to_radians(theta):
        return np.divide(np.dot(theta, np.pi), np.float32(180.0))

    def to_degrees(theta):
        return np.divide(np.dot(theta, np.float32(180.0)), np.pi)

    theta = to_radians(theta)
    lat1 = to_radians(lat)
    lng1 = to_radians(lng)

    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) +
                     np.cos(lat1) * np.sin(delta) * np.cos(theta))

    lng2 = lng1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(lat1),
                             np.cos(delta) - np.sin(lat1) * np.sin(lat2))

    lng2 = (lng2 + 3 * np.pi) % (2 * np.pi) - np.pi

    return to_degrees(lat2), to_degrees(lng2)


def angle_between_vectors_degrees(u, v):
    """Return the angle between two vectors in any dimension space,
    in degrees."""
    return np.degrees(
        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

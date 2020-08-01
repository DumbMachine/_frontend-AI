import math
import numpy as np

pi = math.pi

def get_geojson_grid(upper_right, lower_left, n=6):
    """Returns a grid of geojson rectangles, and computes the exposure in each section of the grid based on the vessel data.

    Parameters
    ----------
    upper_right: array_like
        The upper right hand corner of "grid of grids" (the default is the upper right hand [lat, lon] of the USA).

    lower_left: array_like
        The lower left hand corner of "grid of grids"  (the default is the lower left hand [lat, lon] of the USA).

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
                            "lower_left": lower_left,
                            "upper_right": upper_right
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



def PointsInCircum(_x,y, r,n=100):
    ret = []
    r = 3*r
    for x in range(0,n+1):
        ret.append(
            (
                _x+float(math.cos(2*pi/n*x)*r),
                y+float(math.sin(2*pi/n*x)*r)
            )
        )
    return ret 

def all_grid(grids, centers):
    """
    Assign prob to all the grids
    """
    for grid in grids:
        _points = grid["features"][0]["geometry"]["coordinates"][0][:-1]
        xx, yy = zip(*_points)
        centroid = (sum(xx) / len(_points), sum(yy) / len(_points))
        temp =  assign_prob(centroid, centers)
        grid["prob_dist"] = sum(temp)
    return grids

def assign_prob(centroid, centers):
    """
    Assign probability of the `point` by comparing the distance with the centers
    """
    prob = []
    for center in centers:
        distance = cal_dist(*centroid, *center["center"])
        strip_no = check_range(distance, center["radius"])
        if strip_no is not None:
            prob.append(center["trust"] * center["strip"][strip_no])
    return prob


def check_range(distance, radius):
    """
    Check in which range of the radius does the distance lie in
    """
    if distance < radius:
        prob = 0
    elif distance > radius and distance < 2*radius:
        prob = 1
    elif distance > 2* radius and distance < 3*radius:
        prob = 2
    else:
        prob = None
    return prob

def cal_dist(x1, y1, x2, y2):
    """
    Calculate the distance between the two points
    """
    return math.hypot(x2-x1, y2-y1)
import cv2
import numpy as np


# ================================
#      Perspective Transform
# ================================


def perspective_transform(img, src, dst):
    """Performs a perspective transform.
    Used to convert the Rover camera's POV to a "top-down" world view."""

    # Get transform matrix using cv2.getPerspectiveTransform()
    transform_matrix = cv2.getPerspectiveTransform(src, dst)

    # Warp image using cv2.warpPerspective()
    # Note: warped image has the same size as input image
    warped = cv2.warpPerspective(
        img, transform_matrix, (img.shape[1], img.shape[0]))

    # Added Mask to only process data from the rover's POV
    # [Removes data NOT in Rover's POV]
    mask = cv2.warpPerspective(src=np.ones_like(img[:, :, 0]), M=transform_matrix,
                               dsize=(img.shape[1], img.shape[0]),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Cropped the mask to narrow the Rover's POV. Improves Rover's navigation.
    mask = mask[35:160, 80:240]
    # Adding a black border to cropped mask to regain original shape of (160, 320)
    mask = cv2.copyMakeBorder(
        mask, 35, 0, 80, 80, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return warped, mask


# =============================
#      Color Thresholding
# =============================


def color_thresh(img, rgb_thresh=(160, 160, 160)):
    """Identifies all pixels above the input RGB threshold.
    A Threshold of RGB > 150 is used to identify the ground pixels.
    A Max Threshold is used to prevent sunny areas being favored."""

    # Select all values from the RGB Red channel from the input img
    # Create a zero array with the same dimensions as the Red channel
    color_select = np.zeros_like(img[:, :, 0])

    # Assign True to each pixel above input threshold values in RGB
    # Assign False to all pixels below the threshold
    within_thresh = (img[:, :, 0] > rgb_thresh[0]) \
                    & (img[:, :, 1] > rgb_thresh[1]) \
                    & (img[:, :, 2] > rgb_thresh[2])

    # Assign 1 to all True values in above_thresh
    color_select[within_thresh] = 1

    # Return the binary image
    return color_select


# ====================================
#      Coordinate Transformations
# ====================================


def rover_coords(binary_img):
    """Converts from image coordinates to rover coordinates.
    Accepts the binary image returned from color_thresh().
    Returns [x, y] coordinates with an origin at the Rover's camera."""

    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()

    # Calculate pixel positions with reference to the rover position
    # being at the center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(float)

    return x_pixel, y_pixel


def to_polar_coords(x_pixel, y_pixel):
    """Convert the [x_pixel, y_pixel] from rover_coords() to [distance, angle].
    Where each pixel position is represented by its distance from the origin
    and counterclockwise angle from the positive x-direction."""

    # Calculate distance from Rover to each pixel
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)

    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)

    return dist, angles


def rotate_pix(xpix, ypix, yaw):
    """Maps rover space pixels to world space.
    Used for turning the Rover."""

    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180

    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))

    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    """Translate rover-centric coordinates back into world coordinates."""

    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos

    return xpix_translated, ypix_translated


def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    """Performs the rotation and translation actions required
    for Rover movement & mapping the environment."""

    # Apply rotation using the rotate_pix() function
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)

    # Apply translation using the translate_pix() function
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)

    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)

    return x_pix_world, y_pix_world


# =========================
#      Detecting Rock
# =========================


def find_rocks(img, levels=(110, 110, 50)):
    """Find rocks by applying a RGB threshold to detect rock samples within
    the input threshold. Very similar to the color_thresh() function."""

    # The rocks have high Red and Green levels & low Blue levels
    rockpix = ((img[:, :, 0] > levels[0])
               & (img[:, :, 1] > levels[1])
               & (img[:, :, 2] < levels[2]))

    color_select = np.zeros_like(img[:, :, 0])
    color_select[rockpix] = 1

    return color_select


# ===========================
#      Rover Perception
# ===========================


def perception_step(Rover):
    """Update the Rover's state by using the functions defined above to
    perform all required perception steps to update the Rover."""

    # ==================================================================
    # 1) Define source and destination points for perspective transform
    # ==================================================================

    # The bottom offset is required because the rover's POV angle
    # Reaches the ground in front of the Rover.
    bottom_offset = 6

    # The destination box will be 2*dst_size on each side
    dst_size = 5

    # Camera image is received by Rover.img
    image = Rover.img

    # The source (actual) and destination (desired) points are defined to warp
    # the input image to a grid where each 10x10 pixel square represents 1 square meter
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size,
                               image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size,
                               image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size,
                               image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size,
                               image.shape[0] - 2 * dst_size - bottom_offset]])

    # =================================
    # 2) Apply perspective transform
    # =================================

    warped, mask = perspective_transform(Rover.img, source, destination)

    # =============================================================================
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # =============================================================================

    # Map of navigable pixels
    threshed = color_thresh(warped) * mask
    obs_map = np.absolute(np.float32(threshed) - 1) * mask
    rock_map = find_rocks(warped, levels=(110, 110, 50))

    # ============================================================================
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # ============================================================================

    # Multiplying by 255 because threshed & obs_map are only 1's & 0's
    # World Map's Red Channel
    Rover.vision_image[:, :, 0] = obs_map * 255
    # World Map's Blue Channel
    Rover.vision_image[:, :, 2] = threshed * 255

    # ==========================================================
    # 5) Convert map image pixel values to rover-centric coords
    # ==========================================================

    xpix, ypix = rover_coords(threshed)
    xpix_obs, ypix_obs = rover_coords(obs_map)

    # ===========================================================
    # 6) Convert rover-centric pixel values to world coordinates
    # ===========================================================

    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size

    # Rover ---> World Pixels
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1],
                                    Rover.yaw, world_size, scale)
    # Obstacles ---> World Pixels
    obs_x_world, obs_y_world = pix_to_world(xpix_obs, ypix_obs, Rover.pos[0],
                                            Rover.pos[1], Rover.yaw,
                                            world_size, scale)

    # ==================================================================
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # ==================================================================

    # World Map's Navigable Pixels
    Rover.worldmap[y_world, x_world, 2] += 255
    # Clear opposing data to improve Fidelity
    Rover.worldmap[obs_y_world, obs_x_world, 2] -= 40

    # World Map's Obstacles
    Rover.worldmap[obs_y_world, obs_x_world, 0] += 255
    # Clear opposing data to improve Fidelity
    Rover.worldmap[y_world, x_world, 0] -= 90

    # ==============================================================
    # 8) Convert rover-centric pixel positions to polar coordinates
    # ==============================================================

    # Update Rover pixel distances and angles
    dist, angles = to_polar_coords(xpix, ypix)

    # Find Rover Coordinates
    rock_x, rock_y = rover_coords(rock_map)

    # Find the closest Rock to Rover
    rock_dist, rock_angles = to_polar_coords(rock_x, rock_y)

    if rock_map.any():
        # At Rock to World Map
        rock_x_world, rock_y_world = pix_to_world(
            rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 255
        Rover.vision_image[:, :, 1] = rock_map * 225

    else:
        Rover.vision_image[:, :, 1] = 0

    if len(rock_dist) > 0:
        if Rover.mode == 'reverse':
            Rover.mode = 'reverse'
        else:
            Rover.mode = 'going_to_rock'
            Rover.rock_dists = rock_dist
            Rover.rock_angle = rock_angles
    else:
        Rover.nav_dists = dist
        Rover.nav_angles = angles

    return Rover

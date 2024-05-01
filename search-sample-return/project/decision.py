import numpy as np


# =====================================================
# ---> Set Forward
# =====================================================


def set_forward(Rover):
    """Handles all forward navigation if no rock samples
    are visible. If a rock is visible, the "going_to_rock"
    function is called."""
    Rover.mode = 'forward'

    # Check the extent of navigable terrain
    if len(Rover.nav_angles) >= Rover.stop_forward:
        Rover.brake = 0
        # If mode is forward, navigable terrain looks good
        # and velocity is below max, then throttle
        if Rover.vel < Rover.max_vel:
            # Set throttle value to throttle setting
            Rover.throttle = Rover.throttle_set
        else:  # Else coast
            Rover.throttle = 0

        # Set steering to average angle clipped to the range +/- 15
        Rover.steer = np.clip(
            np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)

    # If there's a lack of navigable terrain pixels then go to 'stop' mode
    elif len(Rover.nav_angles) < Rover.stop_forward:
        # Set mode to "stop" and hit the brakes!
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.steer = 0
        Rover.mode = 'stop'
        return Rover


# =====================================================
# ---> Set Reverse
# =====================================================


def set_reverse(Rover):
    """Called when the Rover.stuck_count reaches values
    set within the Forward & going_to_rock mode.
    Additionally used to backup the Rover after picking
    up a rock."""
    Rover.mode = 'reverse'

    Rover.brake = 0
    Rover.throttle = -0.6

    # Prevents the Rover from getting stuck on top of
    # or within rocks.
    if Rover.vel > -0.02 and Rover.vel <= 0.02:
        Rover.stuck_in_stuck_counter += 1
    else:
        if Rover.stuck_in_stuck_counter >= 0.5:
            Rover.stuck_in_stuck_counter -= 0.5

    # Required to prevent a Numpy RuntimeWarning
    # If the rover is in front of an obstacle
    # np.mean() cannot comput len(Rover.nav_angles)
    if len(Rover.nav_angles) < Rover.go_forward:
        Rover.steer = 0
    elif Rover.stuck_in_stuck_counter >= 25:
        Rover.steer = 15
        Rover.throttle = 0
    else:
        # Steer in the opposite direction as direction
        # that lead to getting stuck
        Rover.steer = -(np.clip(
            np.mean(Rover.nav_angles * 180 / np.pi), -15, 15))

    if Rover.stuck_count >= 0.5:
        Rover.stuck_count -= 0.5
    else:
        Rover.throttle = Rover.throttle_set
        Rover.stuck_count = 0
        Rover.stuck_in_stuck_counter = 0
        Rover.mode = 'forward'
        return Rover


# =====================================================
# ---> Stopping
# =====================================================


def set_stop(Rover):
    """Called when the len(Rover.nav_angles) < Rover.go_forward
    Transitions to Forward mode when untrue."""
    Rover.mode = 'stop'

    # If we're in stop mode but still moving keep braking
    if Rover.vel > 0.2:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0

    # If we're not moving (vel < 0.2) then do something else
    elif Rover.vel <= 0.2:
        # Now we're stopped and we have vision data to see if there's a path forward
        if len(Rover.nav_angles) < Rover.go_forward:
            Rover.throttle = 0
            # Release the brake to allow turning
            Rover.brake = 0
            Rover.steer = -15

        # If we're stopped but see sufficient navigable terrain in front then go!
        if len(Rover.nav_angles) >= Rover.go_forward:
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            # Set steer to mean angle
            Rover.steer = np.clip(
                np.mean(Rover.nav_angles * 180 / np.pi), -17, 17)
            Rover.mode = 'forward'
            return Rover


# =====================================================
# ---> Cut Out
# =====================================================


def cut_out(Rover):
    """Improves Rover's navigation by cutting out of long turns.
    The list: Rover.steer_cuts[] is unevenly distributed to give
    a randomized approach. These random cuts allow the Rover to
    eventual navigate the entire map."""
    Rover.mode = 'cut_out'

    # Prevent Rover from turning into an obstacle
    # when turning out of a circle
    if len(Rover.nav_angles) < Rover.stop_forward:
        Rover.mode = 'stop'
        return Rover

    # Rover.steer_cuts is a list of negative & positive
    # values. The list is used to give the Rover a
    # randomized directional choice. Prevents the Rover
    # from missing areas of the map & infinite circles.
    if Rover.steer_cut_index >= len(Rover.steer_cuts):
        Rover.steer_cut_index = 0
    Rover.steer = Rover.steer_cuts[Rover.steer_cut_index]

    if Rover.cut_out_count >= 1.0:
        Rover.cut_out_count -= 1.0
    else:
        Rover.cut_out_count = 0
        Rover.steer_cut_index += 1
        Rover.mode = 'forward'
        return Rover


# =====================================================
# ---> Going to a Rock
# =====================================================


def going_to_rock(Rover):
    """Called in perception.py if a rock is visible.
    Sets the Rover's steering toward the closest rock."""
    Rover.mode = 'going_to_rock'

    # Pointing steer angles to the closest Rock
    Rover.steer = np.clip(
        np.mean(Rover.rock_angle * 180 / np.pi), -15, 15)

    # Slow Down & Prevent backwards movement
    if Rover.vel > 1 or Rover.vel < -0.03:
        Rover.brake = 1
    else:
        Rover.brake = 0

    # Setting a low max velocity to prevent
    # hard stops when near a sample
    if Rover.vel < 0.8:
        Rover.throttle = 0.2
    else:
        Rover.throttle = 0

    # If the Rover is close enough to pick-up
    if Rover.near_sample == 1:
        Rover.brake = 10
        Rover.mode = 'picking_rock'
        return Rover


# =====================================================
# ---> Picking up a Rock
# =====================================================


def picking_rock(Rover):
    """Called when Rover.near_sample == 1"""
    Rover.mode = 'picking_rock'

    Rover.steer = 0

    if not Rover.picking_up:
        Rover.send_pickup = True
    else:
        Rover.send_pickup = False

    if Rover.near_sample == 0:
        # Do a short backup after picking rock
        # Prevents Rover from turning around
        # after picking a rock near a wall
        Rover.stuck_count = 30
        Rover.mode = 'reverse'
        return Rover


# =====================================================
# --->  THE MAIN() FUNCTION
# =====================================================


def decision_step(Rover):
    """Decision tree for determining throttle, brake and steer commands.
    NOTE: Based on the output of the perception_step() function.
    """

    # Verify if Rover has vision data
    if Rover.nav_angles is not None:
        # Check for Rover.mode status

        # ---> Reverse
        # ====================
        if Rover.mode == 'reverse':
            set_reverse(Rover)

        # ---> Stop
        # ====================
        elif Rover.mode == 'stop':
            if Rover.stuck_count >= 50:
                Rover.mode = 'reverse'
                return Rover
            set_stop(Rover)

        # ---> Forward
        # ====================
        elif Rover.mode == 'forward':
            # Setting brake to 0 in case pervious
            # mode applied a brake
            Rover.brake = 0

            # Conditions for Cutting Out:
            if Rover.vel >= 1.3:
                # Call the cut_out() function
                if Rover.cut_out_count >= 50.0:
                    Rover.mode = 'cut_out'
                    return Rover
                # If going Forward, Vel > 1.8 AND Steering
                # at -15 or 15: add to the counter
                elif Rover.steer == 15.0 or Rover.steer == -15.0:
                    Rover.cut_out_count += 1
                # Else: subtract from counter
                else:
                    if Rover.cut_out_count >= 1:
                        Rover.cut_out_count -= 1

            # Checking if Rover is Stuck:
            if Rover.throttle == Rover.throttle_set:
                # If Rover is stuck
                if Rover.stuck_count >= 55.0:
                    Rover.mode = 'reverse'
                    return Rover
                # If going Forward, with Vel in range(-0.2, 0.06)
                # AND in full throttle: add to the counter
                elif Rover.vel < 0.06 and Rover.vel > -0.2:
                    Rover.stuck_count += 1
                # Else: subtract from counter
                else:
                    if Rover.stuck_count >= 0.5:
                        Rover.stuck_count -= 0.5

            # If not stuck OR about to cut out, go forward
            set_forward(Rover)

        # ---> Going to Rock
        # ====================
        elif Rover.mode == 'going_to_rock':
            # Checking if Rover is Stuck:
            if Rover.throttle == 0.2 and Rover.near_sample == 0:
                if Rover.stuck_count >= 60.0:
                    Rover.mode = 'reverse'
                    return Rover
                elif Rover.vel < 0.05 and Rover.vel > -0.2:
                    Rover.stuck_count += 1
                else:
                    if Rover.stuck_count >= 0.5:
                        Rover.stuck_count -= 0.5
            going_to_rock(Rover)

        # ---> Picking Rock
        # ====================
        elif Rover.mode == 'picking_rock':
            picking_rock(Rover)

        # ---> Cut Out
        # ====================
        elif Rover.mode == 'cut_out':
            cut_out(Rover)

        else:
            print("ERROR: Unknown state")

    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    if Rover.near_sample == 1 and Rover.vel == 0 and not Rover.picking_up:
        Rover.stuck_count = 0
        Rover.send_pickup = True
    else:
        Rover.send_pickup = False

    return Rover

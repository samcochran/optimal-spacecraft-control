# Animations

[Download](https://github.com/samcochran/Gravitational-Slingshot/tree/master/Animations) the animations from here.

Check out our [other animations](nbody_slingshot.md) for modelling the nbody problem and finding slingshot trajectories.

## Introduction

Here we have animations of the optimal paths found by our optimal control solutions. The thrust line in the animation is directed opposite the control, representing the direction fuel would be burned in order to generate the thrust required by the spacecraft for the optimal solution. All of these solutions are for a spacecraft navigating the solar system, which the Sun and Jupiter as primary bodies influencing the gravity.

## First control attempt

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/initial_control_attempt.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

Interestingly, the control's direction rotates 180 degrees as it passes by Jupiter to avoid getting caught by Jupiter's gravitational field.  Note the continued and varied fuel use throughout the flight.

## Slingshot off the sun

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/second_control_attempt.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

Here, we note that since the target destination is far below where the no-thrust trajectory would carry the spacecraft, the control solution has the craft thrust strongly downward after passing by the sun, cutting off the thrust for the remainder of the flight. This is interesting, and it fits with our intuition that it is best to spend the beginning of the flight using fuel to maneuver the craft into the desired trajectory, and then to stop thrusting to conserve fuel until arriving at the destination. The thrust in the beginning to the left accelerates the craft towards the sun and the target destination on the other side. This also demonstrates the spacecraft using the sun's gravity as a slingshot to get to the target past.

## Third control attempt

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/third_control_attempt.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

In this scenario, our destination is again far below where the no-thrust trajectory would take the spacecraft, so we have to thrust downward.  But in this case, the dynamics of the system are such that a more continuous burn is needed through most of the flight.  Notice how much thrust it has to use and how it slowly tapers off as time passes. It is interesting to compare this to the slingshot maneuvers above and below where after the slingshot is complete the spacecraft requires very little thrust and adjustment to its trajectory.

## A successful slingshot maneuver with control

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/control_attempt_4.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

Interestingly, we see that the majority of the thrust for this flight takes place at the very beginning, and cuts off as soon as the slingshot maneuver is complete. This matches well with our intuition that it is best to thrust hard through the slingshot maneuver while close to the planet, and then to allow the craft to continue without thrust towards the destination for the remainder of the flight.  Similarly to the slingshot depicted in the second animation, all of the maneuvering to put the craft onto a trajectory leading to the destination takes place at the very beginning of the flight. We also emphasize how fast the thrust cuts off after the slingshot is completed in this example.  It is pretty amazing that our model achieved this using only soft constraints on the control.

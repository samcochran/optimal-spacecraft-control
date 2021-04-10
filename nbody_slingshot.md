# Animations - N-body problem and Slingshots

[Download](https://github.com/samcochran/Gravitational-Slingshot/tree/master/Animations) the animations from here.

Checkout our newer [optimal control animations](index.md).

## Automating the initial condition selection process

### Using optimization tools to find initial velocity conditions for the third body that result in a trajectory passing through a desired point in space

I created a function that accepts a goal point and a guess for a set of initial conditions, and finds the initial velocity condidions for the third body that allow it to pass through the goal point.  Though imperfect, this does a good job of allowing us to just pass in a point we want the third body to pass through and then generating ICs that give us the desired trajectory passing through that point.

Amazingly, we are able to execute a wide array of very specific trajectories by changing only the initial velocity conditions for the third body .  Using optimization tools, we are able to very accurately get the third body to pass through the point we want it to by starting it with the right velocity.  Thus, as long as we can place the third body in the indicated position in space with the proper velocity, it is able to move through the maneuvers needed to arrive at the desired destination without having to use any power or thrust.  If we can place it right, it simply moves with the gravitational currents through the slingshot maneuver and to the target point.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/guided.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/guided2.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/guided3.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

## Finding viable initial conditions to execute the slingshot maneuver

### Clean slingshot gridsearch

Here we overlay the animations of several viable slingshot maneuvers (note that we are using a toy problem with equal mass primaries here).  As we can see, slight changes in the initial velocities result in significant changes in the end trajectory, but with the scale of velocities we are using, we at least get consistent, predictable trajectories in the plane.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/clean_slingshot.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Chaotic perturbations

We again overlay the animations of several slingshot maneuvers with slightly different initial velocities, but in this case, the velocities are too small.  As a result, the trajectories we get can be chaotic, changing drastically even with only slightly different initial conditions.  We note that, as seen in the animation above, this chaotic behavior is tamed by using greater starting velocities for the third body, making it much less subject to chaotic oscillations caused by the primaries since it escapes sooner and more reliably.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/chaotic.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Solar System Basic

This model, like the previous ones, assumes that the third body is massless, but now we are using with the proper mass ratio to model the interaction between the sun and Jupiter.  As we can see, we are able to pull off a successful slingshot maneuver.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/solar_system_basic.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Solar System General

This model also uses the proper mass ratio for the sun and Jupiter, rather than the toy problems in other animations. Now, however we give the third body a small positive mass (rather than modeling it as being massless).  This is slightly more realistic, but encouragingly, the results are basically identical to the previous model that used a massless third body, indicating that our model is structurally stable.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/solar_system_general.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

## Experimenting with 3D dynamics and rotational coordinates

### Escape

This was our first time finding initial conditions that allow the third body to slingshot off one of the primaries and then escape the system.  The light gray arrows show the acceleration field of the system, and indicate where a massless body would accelerate at each point in space due to the gravitational force of the two primaries.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/escape.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Escape (in rotating coordinates)

Here we have plotted our successful, escaped slingshot, now in rotating coordinates.  That is, we shift our frame of reference with the primaries so that they appear to remain stationary while the third body moves around them.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/escape_rotated.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Escape 3D

Here, we plot the same escape trajectory as before, but with one significant change.  Now, we have started the 3rd body's postion along the -z axis and gave it a positive z velocity.  The model seems to be robust to this change, and the results are comfortingly similar to those from before.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/escape_3body_3d.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Escape 3D (in rotating coordinates)

Here, we plot the same thing as above, but again in rotating coordinates, so our frame of reference rotates along with the two primary bodies.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/escape_rotated_3d.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

## Animating our initial efforts to model the dynamics of the system using a toy problem

### Messy

This model shows a messy trajectory achieved by choosing specific initial conditions for the toy problem with equal masses.  It is interesting to watch how the third body moves in tandem with one of the larger primaries, oscillating aroudn it as both orbit the barycenter.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/messy_3d.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Tron

This set of initial conditions produce an animation that is reminiscient of a tron lightbike match, where purple cuts off green. 

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/tron.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Jagged

Interestingly, these initial conditions cause the third body to begin to move in one direction, then come to a complete stop before moving back to follow one of the primaries.  The result is a jagged trajectory.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/jagged.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Close

Here, we finally found a set of initial conditions that came close to allowing for a successful slingshot, though the third body curves back in.  Still, it does gain a lot of velocity through the maneuver, which is what we are trying to capture in our model.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/close.mp4" type="video/mp4">
Your browser doesn't support this video
</video>

### Better Attempt

Here, we get closer to a successful slingshot.

<video width="640" height="640" controls>
<source src="https://github.com/samcochran/Gravitational-Slingshot/raw/master/Animations/best_attempt.mp4" type="video/mp4">
Your browser doesn't support this video
</video>



This readme describes what the different fields of droneConfig represent, and the required/expected inputs.


- "rotor config" - dictionary of which links the rotor numbers (as described in the flight controller) to their position on the quadrotor when facing forward. 
- "rotor1 rotation direction" - a string which indicates which way rotor 1 rotates. Possible options are "CCW" (counterclockwise) or "CW" (clockwise)
- "rotor radius" - radius of the rotor
- "b" - characteristic length of the quadrotor arm (i.e. the diagonal distance from the rotor centers to the center of gravity)
- "optitrack marker cg offset" - (only relevant if using the OptiTrack external motion capturing system) dictionary transforms the center of the optitrack markers to the quadrotor center of gravity
- "axis direction correction" - (only relevant if using the OptiTrack external motion capturing system) dictionary which describes the transformation from the OptiTrack ground coordinate system to the quadrotor's body axis system. The order of rotations is specified under 'order' (e.g. ZYX means rotate about Z, then Y, then X)
- "optitrack ground axes correction" - (only relevant if using the OptiTrack external motion capturing system) like axis direction correction, this dictionary describes any necessary transformations to align the optitrack coordinate system with the conventional optitrack coordinate system (x-forward, z-right, y-up)
- "flip attitude sign" - Booleans which will flip the direction of the on-board attitude angles (to convert on-board attitude to other coordinate systems)
- "flip accelerometer sign" - Booleans which will flip the direction of the on-board accelerations (e.g. to match flight controller axis system to conventional aerospace system)
- "idle RPM" - The idle eRPM of the quadrotor
- "max RPM" - The maximum eRPM of the quadrotor
- "moment of inertia" - matrix of the quadrotor's moment of inertia
- "betaflight raw acceleration correction factor" - (Only relevant if betaflight is used as the flight controller) conversion from logged acceleration values to acceleration in g
- "flight controller" - name of the flight controller protocol used
- "mass" - (Typically unused) average mass of the quadrotor. 
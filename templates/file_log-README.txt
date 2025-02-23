This is a readme for the file_log.csv explaining what how it is used, and how information should be inputted here. 

GENERAL:
- The file_log.csv is used by the processing scripts (see droneIdentification) to fetch relevant flight data.

PURPOSE:
- Keep track of directories of raw flight data and associated file names
- Keep track of directories and filenames of corresponding flight videos (if available)
- Keep track of directory and filename of the (static) drone configuration file (which stores additional metadata such as the rotor numbering and configuration)
- Keep track of additional metadata:
	- Number of batteries used in flight
	- What quadrotor was flown
	- Mass of the quadrotor
	- Type of flight (line-of-sight, LOS, or first-person-view, FPV)
	- Any additional comments (e.g. wind conditions, types of manoeuvres conducted)


COLUMN NAMES: (format: - <Column name> - <Type of variable interpretted by processing scripts> - <description>) 
- Flight ID - string - Unique ID corresponding to the flight; can be used to link to additional metadata. 
- Rigid Body Name - string - (Only relevant for OptiTrack* flights) Name assigned to the rigid body in the OpitTrack system. 
- Quadrotor - string - name of the quadrotor. 
- Initial Yaw Offset - float (degrees) - (Only relevant for OptiTrack* flights) Yaw offset between the optitrack axis system (when facing computer screen: x-forward, y-up, z-right) and drone axis system (x-forward, y-right, z-down). It is standard practice in the Cyberzoo to initialize the rigid body with the drone (positive) x-axis aligned with the (positive) OptiTrack z-axis (i.e. offset = 90 degrees). 
- Raw OptiTrack Name - string - (Only relevant for OptiTrack* flights) filename (excluding extension) of the OptiTrack tracking data corresponding to the flight
- Raw OT Path - string - (Only relevant for OptiTrack* flights) directory of the file specified in "Raw OptiTrack Name"
- Onboard Name - string - Filename of the quadrotor logged flight data (from BetaFlight). 
- OB Path - string - directory of the file specified in "Onboard Name"
- OB row skip - int - The number of rows from the start of the "Onboard Name".csv file to the start of the column headers. Typically filled with MetaData
- OB num columns - int - The number of columns associated with the logged flight data in the "Onboard Name".csv file.
- Indoor or Outdoor - stirng - Describes if flight was conducted indoors or outdoors, specify "Indoor" or "Outdoor". 
- Flight controller - string - Flight controller (e.g. BetaFlight, iNav, Px4) used by quadrotor.
- Batteries - string - The number of batteries used for flight, in conventional FPV format (1S, 2S, 4S, 6S, etc.)
- Mass - float (grams) - Mass of the quadrotor
- Drone Config File - string - Filename of the .json file which stores the static metadata associated with the quadrotor (e.g. moment of inertia)
- Drone Config Path - string - directory corresponding to "Drone Config File"
- Video - string - Filename of the corresponding (on-board) video of the flight, if applicable
- Video Path - Directory corresponding to "Video"
- Comments - string - Any additional comments (NOTE: Do not use the ',' in these comments as it is the delimiter for the .csv file)



*OptiTrack is an external motion capturing system which aids in state estimatation (in particular, velocity derivation) for indoor flights in the CyberZoo of TU Delft. 
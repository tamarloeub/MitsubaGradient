# MitsubaGradient
Creating a new plugin for mistuba :	gradient integrator

## Repositories Overview
master repository					- 	according to ECCV 2020 paper "Monotonicity Prior for Cloud Tomography"
adding_air_to_mitsuba repository 	- 	adding air particles to the scene, for the forward and inverse solution. 
										The beta (optical density) of air is significantly smaller than the beta of clouds droplets.
										Therefore, we implemented a constant value of beta air for a simpler solution, with accurate results.

## Installation
This code uses Mitsuba 0.6. For installation and documentation, see https://www.mitsuba-renderer.org/index_old.html
MitsubaGradient installaton:
	1. 	Download the code.
	2. 	Follow Mitsuba installation steps, including running 'scons'.
	3. 	In mtswrapper directory, run : 'pip install .' and 'pip2 install .'
	4. 	In mitsuba directory, run : 'source setpath.sh'
	5. 	For master repo solution: run script GradeintWrapper_crop_cloud_multiscale_with_smooth_duplicated_cloud_monoz_prior.py
		For adding_air_to_mitsuba repo solution: run script GradeintWrapper_multiscale_with_prior_and_air.py

## Usage and Contact
The wrapper (python and C++ changes in mitsuba renderer) was created by Tamar Loeub (Technion - Israel Institute of Technology).
If you find this package useful or have any questions, please let me know at tamar.loeub@gmail.com. 
If you use this package in an academic publication please acknowledge the publication (see LICENSE file).
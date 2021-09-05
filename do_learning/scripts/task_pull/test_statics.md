# Test
Three PPO policies were trained for comparison.
- single-camera input (looking up)
- multi-camera fusion (looking up, looking forward, and looking backward)
- force-vision fusion (one camera looking up and a force sensor at the joint of the sidebar)
- force-vision fusion lighter network (lighter DNN)

**Camera perspectives**

![m-camera](images/multi-camera.png)

**Reward function** (same for all three policies)
- if success, reward = 100
- else if fail ,reward = -10
- else reward = 10\*delta_door_angle - step_penalty - force_penalty, where step_penalty is 0.1, force_penalty is 1 when detected force exceeds 70 N, otherwise 0.

## Summary  
### training performance
The training performance (episodic total reward)

![total-reward](images/episode-total-reward.png)

Conclusion: vision-force sensor fusion > multiple cameras > single camera

### policy evaluation (in the environment for training)
| \ | success rate | average steps | average max-force (N) | least step case trajectory cost (robot+sidebar) (m) |
| :----: | :----: | :----: | :----: | :----: |
| single-camera input | 100% | 16 (13-20) | 148 (30 - 463) | 0.835 + 1.247 |
| multi-camera fusion | 98%  | 13 (10-47) | 191 (19 - 756) | 0.893 + 1.108 |
| force-vision fusion | 100% | 11 (10-13) | 182 (26 - 826) | 0.571 + 1.015 |
| force-vision light  | 100% | 21 (17-28) | 148 (43 - 668) | 1.674 + 1.637 |

### policy generalization
Success rates of different policies applied in different environments
| \ | env-0 | env-1 | env-2 | env-3 | env-4 | env-5 | env-6 | env-7 | env-8 | env-9| env-10|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| single camera input | 100% | 55% | 20%  | 7%   | 5%  | 0%  | 100% | 100% | 100% | 100% | 87% |
| multi-camera fusion | 98%  | 93% | 11%  | 73%  | 42% | 0%  | 100% | 87%  | 98%  | 76%  | 55% |
| force-vision fusion | 100% | 98% | 100% | 100% | 83% | 53% | 100% | 72%  | 98%  | 72%  | 92% |
| force-vision light  | 100% | 99% | 61%  | 98%  | 98% | 0%  | 99%  | 96%  | 99%  | 94%  | 92% |

Environments
- env-0: same settings as the environment for training
- env-1: different settings in door color, door frame color, wall color, lighting condition
- env-2: different settings in door color, door frame color, wall color, lighting condition, wheel-ground friction coefficients, door hinge spring force
- env-3: different settings in door width, door color, door frame color, wall color, lighting condition, wheel-ground friction coefficients
- env-4: different settings in door width, door color, door frame color, door handle color, wall color, lighting condition, wheel-ground friction coefficients, door hinge spring force
- env-5: env-2 with changing wheel-ground **friction coefficients** and adding **camera noise** (Gaussian noise with variance of 0.02)
- env-6: env-0 with camera position change (the hook is 5cm lower, camera is 2cm back in x direction)
- env-7: env-0 with camera rotation (up camera rotation angle is 10 degree)
- env-8: env-0 with camera rotation (up camera rotation angle is 5 degree)
- env-9: env-0 with camera rotation (up camera rotation angle is 15 degree)
- env10: env-0 with left-handle swing door and left side bar

![generalize](images/generalization.png)   

## Environments and Statistics
### env 0 - training environment
![env-0](images/env_0.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: 10kg
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: yellow
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: gray
- door handle:
  - color: white
- wall:
  - color: white
- lighting:
  - one in room: constant = 1
  - one out room: constant = 0.5
- wheel-ground friction: mu1=0.98, mu2=0.98
- camera noise: 0.00

**Test Statistics**
- *single camera*
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **16**, minimum 13 [85], maximum 20 [55]
  - average value: average **97.213**, lowest 84.022 [85], highest 100.905 [57]
  - max force: average 148.215, smallest 30.342 [86] largest 463.159 [78]
  - trajectory of the case with least step  

![least step case](images/env0-single-camera-85.png)
- *multiple cameras*
  - **success rate: 98 / 100**
  - failure case [35,60]
  - steps: average **13**, minimum 10 [25], maximum 47 [96]
  - average value: average **97.332**, lowest 86.985 [63], highest 100.153 [95]
  - max force: average 191.523, smallest 19.581 [60] largest 756.482 [57]
  - trajectory of the case with least step  

![least step case](images/env0-multiple-cameras-25.png)
- *force-vision sensor fusion*   
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **11**, minimum 10 [6], maximum 13 [8]
  - average value: average **94.978**, lowest 85.984 [97], highest 96.888 [59]
  - max force: average 182.153, smallest 25.967 [91] largest 826.050 [48]
  - trajectory of the case with least step  

![least step case](images/env0-force-vision-6.png)

### env 1
![env-1](images/env_1.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: 10kg
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: **wood**
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: wood
- door handle:
  - color: white
- wall:
  - color: **painted wall**
- **lighting**:
  - one in room: constant = 0.2
- wheel-ground friction: mu1=0.98, mu2=0.98
- camera noise: 0.00

**Test Statistics**
- *single camera*
  - **success rate: 55 / 100**
  - failure case [1, 2, 4, 5, 6, 7, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 27, 29, 30, 32, 33, 34, 37, 47, 50, 51, 54, 58, 59, 62, 63, 64, 65, 67, 68, 73, 78, 81, 82, 88, 92, 94, 95, 97]
  - steps: average **17**, minimum 14 [16], maximum 28 [19]
  - average value: average **63.274**, lowest 47.699 [18], highest 76.013 [16]
  - max force: average 282.924, smallest 195.541 [50] largest 427.336 [5]
  - trajectory of the case with least step  

![least step case](images/env1-single-camera-16.png)
- *multiple cameras*
  - **success rate: 93 / 100**
  - failure case [6,17,32,42,45,60,66]
  - steps: average **14**, minimum 11 [5], maximum 35 [11]
  - average value: average **121.757**, lowest 105.400 [11], highest 131.029 [23]
  - max force: average 289.416, smallest 24.736 [56] largest 835.247 [83]
  - trajectory of the case with least step  

![least step case](images/env1-multiple-cameras-5.png)
- *force-vision sensor fusion*   
  - **success rate: 98 / 100**
  - failure case [86,97]
  - steps: average **12**, minimum 11 [6], maximum 14 [37]
  - average value: average **96.093**, lowest 88.525 [8], highest 105.453 [94]
  - max force: average 213.772, smallest 83.557 [94] largest 705.967 [18]
  - trajectory of the case with least step  

![least step case](images/env0-force-vision-6.png)

### env 2
![env-2](images/env_2.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: **20kg**
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: **red**
- **door hinge**:
  - spring reference: 3 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: **black**
- door handle:
  - color: white
- wall:
  - color: **bricks**
- **lighting**:
  - one in room: constant = 1
  - one out room: constant = 1
- **wheel-ground friction**: mu1=0.5, mu2=0.5
- camera noise: 0.00

**Test Statistics**
- *single camera*
  - **success rate: 20 / 100**
  - failure case [0, 1, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 32, 33, 34, 35, 37, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 62, 64, 65, 66, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
  - steps: average **17**, minimum 16 [0], maximum 28 [3]
  - average value: average **87.226**, lowest 77.046 [12], highest 97.874 [19]
  - max force: average 221.412, smallest 118.263 [17] largest 439.820 [11]
  - trajectory of the case with least step  

![least step case](images/env2-single-camera-0.png)
- *multiple cameras*
  - **success rate: 11 / 100**
  - failure case [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 22, 23, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
  - steps: average **13**, minimum 12 [2], maximum 14 [0]
  - average value: average **263.409**, lowest 260.037 [10], highest 267.516 5
  - max force: average 247.133, smallest 86.688 [4] largest 354.034 [5]
  - trajectory of the case with least step  

![least step case](images/env2-multiple-cameras-2.png)
- *force-vision sensor fusion*   
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **12**, minimum 11 [71], maximum 16 [47]
  - average value: average **90.290**, lowest 86.193 [33], highest 96.886 [29]
  - max force: average 164.474, smallest 61.902 [0] largest 672.001 [53]
  - trajectory of the case with least step  

![least step case](images/env2-force-vision-71.png)

### env 3
![env-3](images/env_3.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: **15kg**
  - width: **0.75m**
  - thickness: 4.5cm
  - height: 2.1m
  - color: **wood pallet**
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: **flat black**
- door handle:
  - color: white
- wall:
  - color: **yellow**
- **lighting**:
  - one in room: constant = 0.5
  - one out room: constant = 0.5
- **wheel-ground friction**: mu1=0.7, mu2=0.98
- camera noise: 0.00

**Test Statistics**
- *single camera*
  - **success rate: 7 / 100**
  - failure case [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99]
  - steps: average **14**, minimum 13 [1], maximum 18 [6]
  - average value: average **55.746**, lowest 48.061 [4], highest 65.856 [6]
  - max force: average 163.814, smallest 114.304 [5] largest 234.275 [6]
  - trajectory of the case with least step  

![least step case](images/env3-single-camera-1.png)
- *multiple cameras*
  - **success rate: 73 / 100**
  - failure case [1, 3, 7, 10, 12, 18, 27, 34, 35, 38, 40, 44, 47, 58, 63, 66, 67, 68, 77, 79, 82, 83, 84, 87, 90, 92, 93]
  - steps: average **12**, minimum 11 [4], maximum 24 [18]
  - average value: average **170.775**, lowest 145.139 [63], highest 195.609 [46]
  - max force: average 243.057, smallest 75.241 [13] largest 791.945 [71]
  - trajectory of the case with least step

![least step case](images/env3-multiple-cameras-4.png)
- *force-vision sensor fusion*   
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **11**, minimum 10 [0], maximum 19 [73]
  - average value: average **81.165**, lowest 75.127 [71], highest 86.228 [73]
  - max force: average 205.455, smallest 42.100 [85] largest 699.727 [38]
  - trajectory of the case with least step  

![least step case](images/env3-force-vision-0.png)

### env 4
![env-4](images/env_4.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: **30kg**
  - width: **1.05m**
  - thickness: 4.5cm
  - height: 2.1m
  - color: **dark gray**
- **door hinge**:
  - spring reference: 3 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: **black**
- door handle:
  - color: **gold**
- wall:
  - color: **green**
- **lighting**:
  - one in room: constant = 0.2
- **wheel-ground friction**: mu1=0.7, mu2=0.98
- camera noise: 0.00

**Test Statistics**
- *single camera*
  - **success rate: 5 / 100**
  - failure case [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99]
  - steps: average **19**, minimum 17 [3], maximum 24 [4]
  - average value: average **78.247**, lowest 65.137 [1], highest 89.616 [0]
  - max force: average 305.611, smallest 196.409 [4] largest 420.931 [0]
  - trajectory of the case with least step

![least step case](images/env4-single-camera-3.png)
- *multiple cameras*
  - **success rate: 42 / 100**
  - failure case [2, 4, 5, 6, 8, 10, 11, 13, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29, 31, 34, 35, 36, 39, 40, 41, 43, 44, 46, 47, 48, 49, 53, 54, 55, 57, 59, 60, 63, 64, 65, 68, 70, 76, 78, 79, 80, 82, 85, 86, 88, 92, 94, 95, 96, 97, 99]
  - steps: average **19**, minimum 15 [0], maximum 37 [25]
  - average value: average **152.508**, lowest 92.918 [11], highest 189.009 [16]
  - max force: average 186.881, smallest 77.614 [26] largest 558.366 [13]
  - trajectory of the case with least step  

![least step case](images/env4-multiple-cameras-0.png)
- *force-vision sensor fusion*   
  - **success rate: 83 / 100**
  - failure case []
  - steps: average **15**, minimum 14 [10], maximum 27 [80]
  - average value: average **89.734**, lowest 80.222 [33], highest 90.109 [8]
  - max force: average 156.385, smallest 69.380 [46] largest 347.058 [77]
  - trajectory of the case with least step

![least step case](images/env4-force-vision-10.png)

### env 5
![env-5](images/env_2.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: **20kg**
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: **red**
- **door hinge**:
  - spring reference: 3 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: **black**
- door handle:
  - color: white
- wall:
  - color: **bricks**
- **lighting**:
  - one in room: constant = 1
  - one out room: constant = 1
- **wheel-ground friction**: mu1=0.7, mu2=0.98
- camera noise: Gaussian noise with variance of 0.02

**Test Statistics**
- *single camera*
  - **success rate: 0 / 100**
- *multiple cameras*
  - **success rate: 0 / 100**
- *force-vision sensor fusion*   
  - **success rate: 53 / 100**
  - failure case [6, 8, 9, 12, 17, 18, 22, 24, 25, 27, 31, 32, 34, 35, 36, 38, 39, 41, 42, 44, 45, 46, 48, 49, 53, 55, 56, 57, 59, 60, 62, 63, 64, 65, 67, 73, 74, 75, 76, 84, 87, 88, 91, 94, 95, 96, 98]
  - steps: average **14**, minimum 12 [0], maximum 36 [23]
  - average value: average **90.206**, lowest 85.573 [46], highest 95.281 [50]
  - max force: average 240.688, smallest 50.582 [35] largest 700.568 [23]
  - trajectory of the case with least step

![least step case](images/env5-force-vision-0.png)

### env 6
![env-6](images/env_0.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: 10kg
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: yellow
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: gray
- door handle:
  - color: white
- wall:
  - color: white
- lighting:
  - one in room: constant = 1
  - one out room: constant = 0.5
- wheel-ground friction: mu1=0.98, mu2=0.98
- camera noise: 0.00
- **camera position**: 5 cm lower in z direction (joint_fbumper_hook z - 0.05), 2 cm back in x direction (cam_up, x = 0)

**Test Statistics**
- *single camera*
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **17**, minimum 16 [8], maximum 21 [73]
  - average value: average **95.689**, lowest 91.199 [95], highest 98.436 [93]
  - max force: average 147.575, smallest 38.728 [2] largest 265.578 [8]
  - trajectory of the case with least step

![least step case](images/env6-single-camera-8.png)
- *multiple cameras*
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **13**, minimum 11 [29], maximum 35 [58]
  - average value: average **100.656**, lowest 88.369 [58], highest 104.538 [15]
  - max force: average 174.547, smallest 22.938 [12] largest 685.727 [96]
  - trajectory of the case with least step

![least step case](images/env6-multiple-cameras-29.png)
- *force-vision sensor fusion*   
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **12**, minimum 10 [1], maximum 14 [48]
  - average value: average **91.558**, lowest 80.651 [17], highest 95.068 [8]
  - max force: average 162.625, smallest 36.708 [16] largest 536.654 [42]
  - trajectory of the case with least step

![least step case](images/env6-force-vision-1.png)


### env 7
![env-7](images/env_0.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: 10kg
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: yellow
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: gray
- door handle:
  - color: white
- wall:
  - color: white
- lighting:
  - one in room: constant = 1
  - one out room: constant = 0.5
- wheel-ground friction: mu1=0.98, mu2=0.98
- camera noise: 0.00
- **camera position**: rotation (yaw) angle 10 degree (camp_up roll = 0.174)

**Test Statistics**
- *single camera*
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **15**, minimum 15 [2], maximum 24 [73]
  - average value: average **96.329**, lowest 89.91 [65], highest 101.624 [8]
  - max force: average 113.69, smallest 37.74 [85] largest 344.972 [73]
  - trajectory of the case with least step

![least step case](images/env7-single-camera-2.png)
- *multiple cameras*
  - **success rate: 87 / 100**
  - failure case [6, 12, 17, 22, 39, 65, 69, 73, 74, 78, 92, 95, 98]
  - steps: average **13**, minimum 10 [0], maximum 35 [53]
  - average value: average **95.334**, lowest 88.29 [53], highest 98.42 [75]
  - max force: average 194.68, smallest 20.02 [57] largest 783.35 [71]
  - trajectory of the case with least step

![least step case](images/env6-multiple-cameras-29.png)
- *force-vision sensor fusion*   
  - **success rate: 72 / 100**
  - failure case [3, 5, 15, 21, 23, 26, 27, 28, 31, 33, 35, 38, 41, 44, 51, 54, 56, 61, 67, 71, 76, 80, 81, 83, 84, 94, 96, 99]
  - steps: average **18**, minimum 11 [21], maximum 55 [52]
  - average value: average **92.788**, lowest 77.85 [50], highest 96.81 [52]
  - max force: average 293.977, smallest 36.476 [14] largest 3581 [21]
  - trajectory of the case with least step

![least step case](images/env7-force-vision-21.png)

### env 8
![env-8](images/env_0.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: 10kg
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: yellow
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: gray
- door handle:
  - color: white
- wall:
  - color: white
- lighting:
  - one in room: constant = 1
  - one out room: constant = 0.5
- wheel-ground friction: mu1=0.98, mu2=0.98
- camera noise: 0.00
- **camera position**: rotation (yaw) angle 5 degree (up roll 0.087)

**Test Statistics**
- *single camera*
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **16**, minimum 15 [8], maximum 24 [6]
  - average value: average **97.086**, lowest 91.77 [47], highest 101.164 [66]
  - max force: average 117.655, smallest 28.95 [40] largest 405.96 [6]
  - trajectory of the case with least step

![least step case](images/env8-single-camera-8.png)
- *multiple cameras*
  - **success rate: 98 / 100**
  - failure case [32,87]
  - steps: average **13**, minimum 9 [73], maximum 35 [6]
  - average value: average **95.89**, lowest 88.88 [64], highest 98.93 [73]
  - max force: average 192.82, smallest 21.28 [77] largest 748.90 [53]
  - trajectory of the case with least step

![least step case](images/env8-multiple-cameras-73.png)
- *force-vision sensor fusion*   
  - **success rate: 98 / 100**
  - failure case [81,95]
  - steps: average **11**, minimum 9 [95], maximum 34 [20]
  - average value: average **94.37**, lowest 82.47 [73], highest 96.87 [54]
  - max force: average 190.279, smallest 22.181 [58] largest 664.92 [76]
  - trajectory of the case with least step

![least step case](images/env8-force-vision-21.png)

### env 9
![env-9](images/env_0.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: 10kg
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: yellow
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: gray
- door handle:
  - color: white
- wall:
  - color: white
- lighting:
  - one in room: constant = 1
  - one out room: constant = 0.5
- wheel-ground friction: mu1=0.98, mu2=0.98
- camera noise: 0.00
- **camera position**: rotation (yaw) angle 15 degree (up roll 0.2617)

**Test Statistics**
- *single camera*
  - **success rate: 100 / 100**
  - failure case []
  - steps: average **16**, minimum 14 [43], maximum 49 [98]
  - average value: average **95.026**, lowest 78.94 [73], highest 107.19 [98]
  - max force: average 134.98, smallest 42.529 [11] largest 655.27 [98]
  - trajectory of the case with least step

![least step case](images/env9-single-camera-43.png)
- *multiple cameras*
  - **success rate: 76 / 100**
  - failure case [2, 6, 11, 12, 14, 17, 25, 32, 33, 39, 40, 47, 48, 58, 62, 65, 74, 78, 85, 91, 92, 94, 95, 97]
  - steps: average **14**, minimum 10 [20], maximum 23 [16]
  - average value: average **94.15**, lowest 89.32 [53], highest 97.42 [20]
  - max force: average 187.15, smallest 44.63 [54] largest 686.69 [57]
  - trajectory of the case with least step

![least step case](images/env9-multiple-cameras-20.png)
- *force-vision sensor fusion*   
  - **success rate: 72 / 100**
  - failure case [4, 8, 9, 13, 18, 23, 34, 35, 37, 38, 42, 43, 45, 52, 53, 57, 58, 60, 66, 72, 75, 79, 80, 82, 87, 88, 89, 93]
  - steps: average **34**, minimum 16 [9], maximum 53 [30]
  - average value: average **83.51**, lowest 71.43 [13], highest 93.47 [42]
  - max force: average 277.457, smallest 52.23 [9] largest 725.41 [13]
  - trajectory of the case with least step

![least step case](images/env9-force-vision-9.png)

### env 10 (fourwheeler_base_l)
![env-10](images/env_5.png)
100 test cases with random initial pose of the mobile robot

**Environment Settings**
- door:
  - mass: 10kg
  - width: 0.9m
  - thickness: 4.5cm
  - height: 2.1m
  - color: yellow
- door hinge:
  - spring reference: 2 (number of spring)
  - spring stiffness: 1
- door frame:
  - color: gray
- door handle:
  - color: white
- wall:
  - color: white
- lighting:
  - one in room: constant = 1
  - one out room: constant = 0.5
- wheel-ground friction: mu1=0.98, mu2=0.98
- camera noise: 0.00
- **camera position**: rotation (yaw) angle 10 degree (camp_up roll = 0.174)

**Test Statistics**
- *single camera*
  - **success rate: 87 / 100**
  - failure case [6, 14, 25, 37, 48, 53, 56, 59, 82, 84, 86, 92, 95]
  - steps: average **19**, minimum 14 [22], maximum 44 [11]
  - average value: average **82.99**, lowest 72.66 [52], highest 90.73 [57]
  - max force: average 210.21, smallest 50.208 [31] largest 463.67 [77]

- *multiple cameras*
  - **success rate: 55 / 100**
  - failure case [0, 1, 6, 8, 10, 12, 14, 17, 20, 21, 24, 26, 27, 30, 32, 34, 36, 38, 41, 42, 43, 44, 46, 47, 48, 50, 54, 55, 58, 62, 64, 65, 70, 73, 77, 81, 84, 89, 92, 94, 95, 96, 97, 98, 99]
  - steps: average **15**, minimum 10 [1], maximum 24 [28]
  - average value: average **81.88**, lowest 68.78 [28], highest 89.1 [18]
  - max force: average 457.29, smallest 86.33 [1] largest 955.35 [49]

- *force-vision sensor fusion*   
  - **success rate: 92 / 100**
  - failure case [9, 13, 18, 43, 49, 75, 79, 93]
  - steps: average **11**, minimum 9 [46], maximum 14 [11]
  - average value: average **58.88**, lowest 47.18 [41], highest 72.26 [68]
  - max force: average 258.839, smallest 48.83 [46] largest 889.96 [15]

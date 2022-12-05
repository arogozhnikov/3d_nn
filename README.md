# Marching neural networks

Visualizing level surfaces of a neural network using webGL + raymarching technique.

Correction: technique is close raymarching, but step sizes are computed differently.

All the NN-related computing uses shaders, thus having GPU is recommended.

Demo: [https://arogozhnikov.github.io/3d_nn/](https://arogozhnikov.github.io/3d_nn/)


<a href='https://arogozhnikov.github.io/3d_nn/' />
    <img src='https://raw.githubusercontent.com/arogozhnikov/3d_nn/master/images/screen_capture.png' height='300' />
    <img src='https://raw.githubusercontent.com/arogozhnikov/3d_nn/master/images/neural_network_3d.gif' width='300' height='300' />
</a>


## Used libraries

Visualization relies on [THREE.js](https://github.com/mrdoob/three.js) for rendering and [CCapture.js](https://github.com/spite/ccapture.js) for capturing animations.

The code is minimalistic and a bit messy, but no other dependencies / frameworks.

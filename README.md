# 3d mnist

Use this project to generate visually appealing renders of mnist
dataset which can be used in various computer vision experiments.

The script **works in blender 2.82**, but in 2.83 axis of rotations seems to be flipped so digits will have unexpected orientation.
## installation & usage

place mnist_util.py somewhere in blender python sys.path directory.

mnist.py accepts following arguments:

**--source**  directory with mnist files in blender-readable format such as png
**--render**  output directory for rendered images  
**--meta** directory for metadata such as camera position and orientation  
**--blend** blend file to use

example
```
blender --background --python ~/.config/blender/2.82/scripts/startup/mnist.py -- --meta=~/projects/mnist_saver/render_desc --render=~/projects/mnist_saver/render --source=~/projects/mnist_saver/train --blend=~/Documents/untitled.blend
```
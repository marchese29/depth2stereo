# depth2stereo
A tool for converting black-and-white depth maps to cross-eyed stereograms.

## Installation Instructions
A Note: I will eventually be simplifying this, for now you'll need to install all of the modules on
your own.
### Dependencies
You will first need to install all of the required python dependencies.  I will be using `pip` to install my dependencies, but you are more than welcome to do it a different way if you prefer.  To install the dependencies, simply run the command:
```
pip install numpy matplotlib
```
You will additionally need to install opencv along with its python bindings.

Once you have installed the opencv python bindings, you should be good to go, you may need to copy a couple files or create symlinks if you're using a virtualenv.
### Getting the Code
It's git, the standard operating procedure: `git clone git@github.com:marchese29/depth2stereo.git`.

## Usage
The usage text is pretty self-explanatory:
```
usage: driver.py [-h] [--reverse] [--stretch] [--num-layers 5-25]
                 [--center-plane 0-25]
                 depthmap

Depth-map to stereogram converter.

positional arguments:
  depthmap             File location of the depth map in a readable image
                       format.

optional arguments:
  -h, --help           show this help message and exit
  --reverse            Indicates that the depth map is inverted (darker ==
                       closer)
  --stretch            Scale values to 0-255
  --num-layers 5-25    The number of layers of depth.
  --center-plane 0-25  The number of depth layers to appear behind the center
                       plane.

```

## Results
### Input:
![cyclops depth map](https://github.com/marchese29/depth2stereo/blob/master/examples/cyclops.jpg)
### Output:
![cyclops depth map](https://github.com/marchese29/depth2stereo/blob/master/examples/cyclops_result.png)

## Future Plans
* Use a depth-map to turn a standard image into a stereogram.
* Provide more potential points of user customization.

## Looking at the Results
Take a look at [this infographic](http://www.neilcreek.com/2008/02/28/how-to-see-3d-photos/) that explains how to get the full 3D effect of the final product.

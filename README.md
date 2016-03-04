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
                 [--center-plane 0-25] [--jpeg]
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
  --jpeg               Indicates that we should attempt to squash
                       interpolation.

```

## Algorithm
The algorithm implemented in this file is a row-by-row operation that is not order dependent.  It is based on the premise that objects popping out of the page are farther apart from each other, and objects that are closer together appear behind the page.
### The Naive Approach
A naive approach would be to have a simple shift map that correlates pixel values to shift lengths. This would look something like the following:
```
┌----------┬----------┬----------┐    ┌----------┬----------┬----------┬----------┬----------┐
|          |          |          |    |          |          |          |          |          |
|    100   |    ...   |    250   | => |    ...   |    100   |    ...   |    ...   |    250   |
|          |          |          |    |          |          |          |          |          |
└----------┴----------┴----------┘    └----------┴----------┴----------┴----------┴----------┘
```
Note that higher intensity values are shifted further to the right on the far right image, which matches the way a depth-map conveys its information (brighter == closer). However, you may notice a problem with this:
```
┌----------┬----------┐    ┌----------┬----------┬----------┬----------┐
|          |          |    |          |          |          |          |
|    10    |    250   | => |    10    |    ???   |    ???   |    250   |
|          |          |    |          |          |          |          |
└----------┴----------┘    └----------┴----------┴----------┴----------┘
```
Using the naive approach can result in tearing when you see a smaller shift immediately to the left of a larger shift.  Above, an intensity of 10 does not get shifted, while a 250 gets shifted by 2 pixels.  This leaves two blank pixels in the shifted results that one would reasonably expect to have meaningful depth data.

### Reverse Scanning
The algorithm implemented in this program is meant to address the lack of 3d data that causes the tearing between pixels of different intensities.  The algorithm occurs in two parts; first it searches the current scanline for instances of pixel ranges in which tearing may occur, and then it performs the shift and corrects for areas where a tear has left an unintended blank pixel.
#### Identifying Likely Tears:
In general, tears will occur along regions where depth map intensities increase over the entire pixel range.  So the first step of the algorithm is to identify a range of pixels that is strictly decreasing from right to left.
```
┌----------┬----------┬----------┬----------┐
|          |          |          |          |
|    100   |    10    |    120   |    250   |
|          |   start  |          |    end   |
└----------┴----------┴----------┴----------┘
```
We then generate the shifted result using the standard shifts.  Any torn pixels are assigned an intensity of 0.
```
┌----------┬----------┬----------┐    ┌----------┬----------┬----------┬----------┬----------┐
|          |          |          |    |          |          |          |          |          |
|    10    |    120   |    250   | => |    10    |     0    |    120   |     0    |    250   |
|          |          |          |    |          |          |          |          |          |
└----------┴----------┴----------┘    └----------┴----------┴----------┴----------┴----------┘
```
#### Interpolating Tears:
We now travel across both the original region, as well as the shifted region with pointers, and we know we encounter a tear any time the ptrs don't point to the same value.  To fix this, the data is extrapolated from the left.  Whenever both pointers are equal, both get incremented.  If the two pointers are not equal, a tear is identified, filled in from the original pointer, and only the shift-range pointer is incremented.  The process is shown below for the example shown above.
```
               ORIGINAL                                          RESULT
  ┌----------┬----------┬----------┐    ┌----------┬----------┬----------┬----------┬----------┐
  |          |          |          |    |          |          |          |          |          |
  |    10    |    120   |    250   | == |    10    |     0    |    120   |     0    |    250   |
  |   lptr   |          |          |    |   rptr   |          |          |          |          |
  └----------┴----------┴----------┘    └----------┴----------┴----------┴----------┴----------┘
  ┌----------┬----------┬----------┐    ┌----------┬----------┬----------┬----------┬----------┐
  |          |          |          |    |          |  change  |          |          |          |
=>|    10    |    120   |    250   | != |    10    |     0    |    120   |     0    |    250   |
  |          |   lptr   |          |    |          |   rptr   |          |          |          |
  └----------┴----------┴----------┘    └----------┴----------┴----------┴----------┴----------┘
  ┌----------┬----------┬----------┐    ┌----------┬----------┬----------┬----------┬----------┐
  |          |          |          |    |          |          |          |          |          |
=>|    10    |    120   |    250   | == |    10    |    10    |    120   |     0    |    250   |
  |          |   lptr   |          |    |          |          |   rptr   |          |          |
  └----------┴----------┴----------┘    └----------┴----------┴----------┴----------┴----------┘
  ┌----------┬----------┬----------┐    ┌----------┬----------┬----------┬----------┬----------┐
  |          |          |          |    |          |          |          |  change  |          |
=>|    10    |    120   |    250   | != |    10    |    10    |    120   |     0    |    250   |
  |          |          |   lptr   |    |          |          |          |   rptr   |          |
  └----------┴----------┴----------┘    └----------┴----------┴----------┴----------┴----------┘
  ┌----------┬----------┬----------┐    ┌----------┬----------┬----------┬----------┬----------┐
  |          |          |          |    |          |          |          |          |          |
=>|    10    |    120   |    250   | == |    10    |    10    |    120   |    120   |    250   |
  |          |          |   lptr   |    |          |          |          |          |   rptr   |
  └----------┴----------┴----------┘    └----------┴----------┴----------┴----------┴----------┘
=> DONE
```
Now that we have obtained the final shifted range, we insert it back into the final result.

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

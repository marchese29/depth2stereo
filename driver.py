'''
depth2stereo
By: Daniel Marchese <marchese.daniel@gmail.com>

This software will take a depth-map image and convert it to an equivalent stereogram.  This is all
done using an algorithm akin to the methodology described in the graphic here:
http://i.imgur.com/966ieVz.png

This is a work in progress, future plans include:
- Use depth map to convert standard image to a stereogram.
- Much more capability for user intervention.
'''
import argparse
import os
import sys
import traceback

import cv2
from cv2 import cv
import matplotlib.pyplot as plt
import numpy as np


def validate_args(args):
	'''Validates the contents of the given command line arguments.'''
	args.depthmap = os.path.abspath(args.depthmap)
	if not os.path.isfile(args.depthmap):
		raise IOError('%s is not a valid path.' % args.depthmap)

	# TODO: Validate that the images are readable.

	return args


def setup_args():
	'''Configures and validates the command line arguments.'''
	parser = argparse.ArgumentParser(description='Depth-map to stereogram converter.')

	# In place arguments.
	parser.add_argument('depthmap',
		help='File location of the depth map in a readable image format.')

	# Optional arguments
	parser.add_argument('--reverse', action='store_true',
		help='Indicates that the depth map is inverted (darker == closer)')
	parser.add_argument('--stretch', action='store_true',
		help='If a 255 pixel is not present, stretch the brightest pixel to 255.')
	parser.add_argument('--save_image',
		help='Save the image at the end of program execution to the given location.')

	return validate_args(parser.parse_args())


def main():
	try:
		args = setup_args()
	except IOError as err:
		return 'Error: %s' % str(err)

	try:
		img = cv2.imread(args.depthmap, cv.IMREAD_GRAYSCALE)
	except Exception, e:
		raise e
	
	return 0


if __name__ == '__main__':
	try:
		sys.exit(main())
	except Exception, e:
		sys.exit(traceback.format_exc())

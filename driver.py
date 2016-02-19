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
import math
import os
import sys
import traceback

import cv2
import matplotlib.cm as cm
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
    parser.add_argument('--stretch', action='store_true', help='Scale values to 0-255')
    parser.add_argument('--num-layers', type=int, default=10, choices=range(5,26), metavar='5-25',
        help='The number of layers of depth.')
    parser.add_argument('--center-plane', type=int, default=0, choices=range(0,26), metavar='0-25',
        help='The number of depth layers to appear behind the center plane.')

    return validate_args(parser.parse_args())


def calculate_stereogram(depth_map, cmd_args):
    '''This is where all of the number crunching originates.  This function will calculate and
    return the stereogram from the given depth map.

    Algorithm Details to follow...
    '''
    # This keeps track of all attempts to write to a specific pixel in the resulting stereogram.
    cumulative_stereogram = np.zeros((depth_map.shape[0], depth_map.shape[1], cmd_args.num_layers),
        dtype=np.uint8)

    # This keeps track of the number of attempts to write to a specific pixel in the image.
    manip_counts = np.zeros((depth_map.shape[0], depth_map.shape[1]), dtype=np.uint8)

    # Copy over the original image to the left.
    np.copyto(cumulative_stereogram[:,:depth_map.shape[1],0], depth_map)

    # The intensity depth of a layer (range of depths mapping to the same layer).
    layer_depth = 255.0 / float(cmd_args.num_layers)

    for j in xrange(depth_map.shape[1]):
        for i in xrange(depth_map.shape[0]):
            layer_number = int(math.floor(depth_map[i,j] / layer_depth))
            new_j = j + layer_number - cmd_args.center_plane

            # Only write in the stereogram if the points are still in bounds.
            if new_j >= 0 and new_j < depth_map.shape[1]:
                # Don't overwrite old values quite yet.
                cumulative_stereogram[i,new_j,manip_counts[i,new_j]] = depth_map[i,j]
                manip_counts[i,new_j] += 1

    return cumulative_stereogram.max(axis=2)


def main():
    '''The main entry point for the program.'''
    try:
        args = setup_args()
    except IOError as err:
        return 'Error: %s' % str(err)

    try:
        img = cv2.imread(args.depthmap, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError('There was an error reading the image at %s' % args.depthmap)
    except IOError as err:
        return str(err)

    depth_map = np.copy(img)
    if args.stretch:
        depth_map = ((depth_map.astype(float) / float(depth_map.max())) * 255.0).astype(np.uint8)
    if args.reverse:
        depth_map = 255 - depth_map

    # Run the calculation.
    stereogram = calculate_stereogram(depth_map, args)

    # Un-reverse if necessary
    if args.reverse:
        depth_map = 255 - depth_map
        stereogram = 255 - stereogram

    result = np.zeros((img.shape[0], img.shape[1] * 2), dtype=np.uint8)
    np.copyto(result[:,:img.shape[1]], depth_map)
    np.copyto(result[:,img.shape[1]:], stereogram)

    fig = plt.figure('The result')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(result, cm.Greys_r)
    plt.axis('off')
    plt.show()

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('\nReceived Keyboard Interrupt...Aborting')
    except Exception:
        sys.exit(traceback.format_exc())

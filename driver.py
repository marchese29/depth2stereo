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
    '''
    # This keeps track of all attempts to write to a specific pixel in the resulting stereogram.
    cumulative_stereogram = np.zeros((depth_map.shape[0], depth_map.shape[1], cmd_args.num_layers),
        dtype=np.uint8)

    # This keeps track of the number of attempts to write to a specific pixel in the image.
    manip_counts = np.zeros((depth_map.shape[0], depth_map.shape[1]), dtype=np.uint8)

    # The intensity depth of a layer (range of depths mapping to the same layer).
    layer_depth = 255.0 / float(cmd_args.num_layers)

    # Tells us how far to shift stuff.
    shift_map = np.zeros(256, dtype=int)
    for i in xrange(256):
        shift_map[i] = int(math.floor(float(i) / layer_depth)) - cmd_args.center_plane

    for i in xrange(depth_map.shape[0]):
        stop = depth_map.shape[1] - 1
        start = stop - 1

        # Calculate the interval in which tearing will occur.
        while stop >= 0:
            while start >= 0 and shift_map[depth_map[i, start]] <= shift_map[depth_map[i, start+1]]:
                start -= 1
            start += 1

            # Get a reference to the range of original values we are checking.
            original = depth_map[i, start:stop+1]

            # Create a representation of the new range being created.
            left = start + shift_map[depth_map[i, start]]
            shift_left = start - left
            right = stop + shift_map[depth_map[i, stop]]
            shifted = np.zeros(right - left + 1, dtype=np.uint8)

            for j in xrange(original.shape[0]):
                shifted[j + shift_map[original[j]] + shift_left] = original[j]

            # Correct for missing data in the shifted segment.
            orig_ptr = start
            for j in xrange(shifted.shape[0]):
                if shifted[j] == depth_map[i, orig_ptr]:
                    # This entry is good to go.
                    orig_ptr += 1
                else:
                    # This entry is inside of a tear, copy the previous data point.
                    # TODO: Handle missing data according to user preference.
                    shifted[j] = shifted[j-1]

            # Input the data into the cumulative stereogram.
            for j in xrange(shifted.shape[0]):
                idx = start - shift_left + j
                if idx > 0 and idx < depth_map.shape[1]:
                    cumulative_stereogram[i, idx, manip_counts[i, idx]] = shifted[j]
                    manip_counts[i, idx] += 1

            # Shift the indices for the next chunk of data from this line.
            stop = start - 1
            start = stop - 1

    # Use a median filter to cut down a little bit on the edge noise.
    return cv2.medianBlur(cumulative_stereogram.max(axis=2), 7)


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

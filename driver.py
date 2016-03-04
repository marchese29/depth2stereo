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
import multiprocessing
from multiprocessing import sharedctypes
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
    parser.add_argument('--jpeg', action='store_true',
        help='Indicates that we should attempt to squash interpolation.')

    return validate_args(parser.parse_args())


def stereogram_worker(indices):
    '''Works on a chunk of the stereogram denoted by the given indices.'''
    try:
        # Configure global shared memory.
        cumulative_stereogram = np.ctypeslib.as_array(global_stereogram)
        depth_map = np.ctypeslib.as_array(global_depth_map)

        # Additional top-level information
        manip_counts = np.zeros(cumulative_stereogram.shape, dtype=np.uint8)
        layer_depth = 255.0 / float(cmd_args.num_layers)
        shift_map = np.zeros(256, dtype=int)
        for i in xrange(256):
            shift_map[i] = int(math.floor(float(i) / layer_depth)) - cmd_args.center_plane

        for i in xrange(*indices):
            stop = depth_map.shape[1] - 1
            start = stop - 1

            # Calculate the interval in which tearing will occur.
            while stop >= 0:
                while start >= 0 and shift_map[depth_map[i, start]] <= shift_map[depth_map[i, start+1]]:
                    start -= 1
                start += 1

                # Get a copy of the range of original values we are checking.
                original = np.copy(depth_map[i, start:stop+1])

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
                        shifted[j] = shifted[j-1]

                # Input the data into the cumulative stereogram.
                for j in xrange(shifted.shape[0]):
                    idx = start - shift_left + j + cmd_args.num_layers
                    if idx >= 0 and idx < cumulative_stereogram.shape[1]:
                        cumulative_stereogram[i, idx, manip_counts[i, idx]] = shifted[j]
                        manip_counts[i, idx] += 1
                    else:
                        raise Exception('This should not happen')

                # Shift the indices for the next chunk of data from this line.
                stop = start - 1
                start = stop - 1
        return None
    except Exception as ex:
        print str(os.getpid()) + ': ' + traceback.format_exc()
        return ex


def _init_process(shared_stereogram, shared_depth_map, args):
    '''Initializes the shared memory for a new process.'''
    global global_stereogram
    global_stereogram = shared_stereogram

    global global_depth_map
    global_depth_map = shared_depth_map

    global cmd_args
    cmd_args = args


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

    # Find the edges that might have been interpolated, squash them.
    if args.jpeg:
        edge_mask = (cv2.Canny(img, 100, 200) == 255)
        idxs_i, idxs_j = np.nonzero(edge_mask)
        for i, j in zip(idxs_i, idxs_j):
            # Only do this for things not already on the edge of the image.
            if i > 1 and i < img.shape[0]-2 and j > 1 and j < img.shape[1]-2:
                # The pixel that is one to the left is given to the pixel on the far left.
                depth_map[i,j-1] = depth_map[i,j-2]
                # Same with the pixel on the right.
                depth_map[i,j+1] = depth_map[i,j+2]
                # We need to pick a suitable color for the middle pixel
                depth_map[i,j] = depth_map[i,j+2]

    # Configure shared memory
    c_stereogram = np.ctypeslib.as_ctypes(
        np.zeros((depth_map.shape[0], depth_map.shape[1] + (2 * args.num_layers), args.num_layers),
            dtype=np.uint8))
    c_depth_map = np.ctypeslib.as_ctypes(depth_map)
    shared_stereogram = sharedctypes.Array(c_stereogram._type_, c_stereogram, lock=False)
    shared_depth_map = sharedctypes.Array(c_depth_map._type_, c_depth_map, lock=False)

    # Configure the chunks.
    chunksize = depth_map.shape[0] / multiprocessing.cpu_count()
    idx = 0
    chunks = []
    for i in range(multiprocessing.cpu_count()):
        chunks.append((idx, idx + chunksize))
        idx += chunksize
    if chunks[-1][1] < depth_map.shape[0]:
        chunks[-1] = (chunks[-1][0], depth_map.shape[0])

    # Configure the processes.
    process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(),
        initializer=_init_process, initargs=(shared_stereogram, shared_depth_map, args))

    # Start working.
    try:
        result = process_pool.map(stereogram_worker, chunks)

        # Check for failures
        for item in result:
            if isinstance(item, Exception):
                raise item
    except KeyboardInterrupt as ki:
        raise ki
    except Exception as ex:
        raise ex
    finally:
        process_pool.close()

    # Recover the stereogram from the stuff that the workers used.
    stereogram = np.ctypeslib.as_array(shared_stereogram).max(axis=2)

    # Un-reverse if necessary
    if args.reverse:
        depth_map = 255 - depth_map
        stereogram = 255 - stereogram

    result = np.zeros((img.shape[0], img.shape[1] * 2 + (2 * args.num_layers)), dtype=np.uint8)
    np.copyto(result[:, :img.shape[1]], depth_map)
    np.copyto(result[:, img.shape[1]:], stereogram)

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

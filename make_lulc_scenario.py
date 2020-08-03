"""Create scenarios from base LULC given raster masks of new forest."""
import argparse

import numpy
import pygeoprocessing

FOREST_LULC = 50


def replace_where(base_array, mask_array, replacement_val):
    """Replace in `base_arry_where `mask_array==1` with `replacement_val`."""
    result = numpy.copy(base_array)
    result[mask_array == 1] = replacement_val
    return result


def main(args):
    """Entry point."""
    base_raster_info = pygeoprocessing.get_raster_info(
        args.base_lulc_raster_path)
    pygeoprocessing.raster_calculator(
        [(args.base_lulc_raster_path, 1),
         (args.forest_mask_raster_path, 1), (FOREST_LULC, )], replace_where,
        args.target_raster_path, base_raster_info['datatype'],
        base_raster_info['nodata'][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Create scenarios from base LULC given raster masks of new forest.'))
    parser.add_argument(
        'base_lulc_raster_path', help='path to base lulc raster')
    parser.add_argument(
        'forest_mask_raster_path', help='path to additional forest raster')
    parser.add_argument('target_raster_path', help='path to target raster')
    parser.add_argument(
        '--replacement_val', type=int, default=FOREST_LULC,
        help=f'replacement value defaults to {FOREST_LULC} if not provided')
    args = parser.parse_args()
    main(args)

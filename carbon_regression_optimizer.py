"""Used to optimize results from carbon regression.

Run like this: git pull && docker run --name optimize -v `pwd`:/usr/local/workspace -it --rm therealspring/inspring:latest carbon_regression_optimizer.py --target_dir optimize_regression_results --path_to_forest_mask_data carbon_regression_scenario_workspace/data/-61.0,0.0,-60.0,1.0/ --marginal_value_raster carbon_regression_scenario_workspace/marginal_values/marginal_value_regression_-61.0,0.0,-60.0,1.0.tif --sum

target_dir
forest_masks
marginal_value_raster
path_to_forest_mask_data

"""
import argparse
import glob
import logging
import os
import shutil
import sys
import tempfile

from osgeo import gdal
import pygeoprocessing
import numpy
import taskgraph


gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

NODATA = -1


def sum_of_masked_op(mask_path, value_raster_path, churn_dir):
    temp_dir = tempfile.mkdtemp(dir=churn_dir)
    mask_align_path = os.path.join(temp_dir, 'align_mask.tif')
    value_align_path = os.path.join(temp_dir, 'value_align.tif')
    target_pixel_size = pygeoprocessing.get_raster_info(
        value_raster_path)['pixel_size']

    pygeoprocessing.align_and_resize_raster_stack(
        [mask_path, value_raster_path],
        [mask_align_path, value_align_path], ['near']*2,
        target_pixel_size, 'intersection')

    mask_raster = gdal.OpenEx(mask_align_path, gdal.OF_RASTER)
    value_raster = gdal.OpenEx(value_align_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)
    value_band = value_raster.GetRasterBand(1)

    sum_val = 0.0
    for offset_dict in pygeoprocessing.iterblocks(
            (mask_align_path, 1), offset_only=True):
        mask_array = mask_band.ReadAsArray(**offset_dict)
        value_array = value_band.ReadAsArray(**offset_dict)
        sum_val += numpy.sum(value_array[mask_array == 1])

    mask_band = None
    value_band = None
    mask_raster = None
    value_raster = None
    shutil.rmtree(temp_dir)
    return sum_val


def efficiency_op(average_marginal_value, average_forest_coverage):
    result = numpy.zeros(average_marginal_value.shape)
    valid_mask = (average_marginal_value > 0) & (average_forest_coverage > 0)
    result[valid_mask] = (
        average_marginal_value[valid_mask] /
        average_forest_coverage[valid_mask])
    return result


def new_forest_mask_op(restoration_forest_mask, base_forest_mask):
    result = numpy.zeros(restoration_forest_mask.shape, dtype=numpy.uint8)
    new_forest = (restoration_forest_mask == 1) & (base_forest_mask == 0)
    result[new_forest] = 1
    return result


def calc_raster_sum(raster_path):
    """Return the sum of the values in raster_path."""
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    raster_sum = 0.0
    for _, raster_array in pygeoprocessing.iterblocks((raster_path, 1)):
        raster_sum += numpy.sum(
            raster_array[~numpy.isclose(raster_array, nodata)])
    return raster_sum


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Carbon regression scenario maker')
    parser.add_argument(
        '--target_dir', help="path to output dir")
    parser.add_argument(
        '--marginal_value_raster', help='path to marginal value raster')
    parser.add_argument(
        '--path_to_forest_mask_data',
        help='path to the scenario clipped dir of the forest masks.')
    parser.add_argument(
        '--sum', action='store_true',
        help='if set, report sum of marignal value raster')
    parser.add_argument(
        '--target_val', type=float, default=None,
        help='if set use this as the goal met cutoff')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph(args.target_dir, -1)

    churn_dir = os.path.join(args.target_dir, 'churn')
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass

    raster_sum_task = task_graph.add_task(
            func=calc_raster_sum,
            args=(args.marginal_value_raster,),
            task_name=f'calc sum for {args.marginal_value_raster}')
    raster_sum = raster_sum_task.get()
    if args.sum:
        LOGGER.info(f'{args.marginal_value_raster}: {raster_sum}')

    # sum marginal value to 30km pixel
    pixel_size_30km = (30/111, -30/111)
    # warp marginal value map to this pixel size using average
    marginal_value_id = os.path.basename(
        os.path.splitext(args.marginal_value_raster)[0])
    marginal_value_30km_average_raster_path = os.path.join(
        churn_dir, f'{marginal_value_id}_30km_average.tif')
    marginal_value_30km_task = task_graph.add_task(
        func=pygeoprocessing.warp_raster,
        args=(
            args.marginal_value_raster, pixel_size_30km,
            marginal_value_30km_average_raster_path, 'average'),
        target_path_list=[marginal_value_30km_average_raster_path],
        task_name='marginal value average to 30km')

    # create count of difference of forest masks from esa to restoration
    restoration_mask_raster_path = os.path.join(
        args.path_to_forest_mask_data,
        'mask_of_forest_10sec_restoration_limited.tif')
    esa_mask_raster_path = os.path.join(
        args.path_to_forest_mask_data, 'mask_of_forest_10sec_esa2014.tif')
    new_forest_mask_raster_path = os.path.join(
        churn_dir, f'{marginal_value_id}_new_forest_mask.tif')
    new_forest_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(restoration_mask_raster_path, 1), (esa_mask_raster_path, 1)],
            new_forest_mask_op, new_forest_mask_raster_path, gdal.GDT_Float32,
            NODATA),
        target_path_list=[new_forest_mask_raster_path],
        task_name='new forest mask')

    # sum count to 30km pixel
    # then warp this difference to 30km size using average

    new_forest_mask_30km_raster_path = os.path.join(
        churn_dir, f'{marginal_value_id}_new_forest_mask_30km.tif')

    new_forest_30km_task = task_graph.add_task(
        func=pygeoprocessing.warp_raster,
        args=(
            new_forest_mask_raster_path, pixel_size_30km,
            new_forest_mask_30km_raster_path, 'average'),
        target_path_list=[new_forest_mask_30km_raster_path],
        dependent_task_list=[new_forest_mask_task],
        task_name='forest mask value average to 30km')

    # divide marginal sum by average count
    # then divide 30km marginal average by 30km difference average
    efficiency_marginal_value_raster_path = os.path.join(
        churn_dir, f'{marginal_value_id}_efficiency.tif')

    _ = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(marginal_value_30km_average_raster_path, 1),
             (new_forest_mask_30km_raster_path, 1)],
            efficiency_op, efficiency_marginal_value_raster_path,
            gdal.GDT_Float32, NODATA),
        dependent_task_list=[new_forest_30km_task, marginal_value_30km_task],
        target_path_list=[efficiency_marginal_value_raster_path],
        task_name='calc efficiency_op')

    # optimize
    optimize_dir = os.path.join(args.target_dir, 'optimize_rasters')
    task_graph.join()
    task_graph.add_task(
        func=pygeoprocessing.raster_optimization,
        args=(
            [(efficiency_marginal_value_raster_path, 1)], churn_dir,
            optimize_dir),
        kwargs={
            'target_suffix': marginal_value_id,
            'goal_met_cutoffs': [float(x)/100.0 for x in range(1, 101)],
            'heap_buffer_size': 2**28,
            'ffi_buffer_size': 2**10,
            },
        task_name='optimize')

    # TODO: evaluate the optimization rasters for total carbon
    sum_task_list = []
    task_graph.join()
    for optimization_raster_mask in sorted(glob.glob(
            os.path.join(optimize_dir, 'working_mask*.tif'))):
        sum_of_masked_task = task_graph.add_task(
            func=sum_of_masked_op,
            args=(
                optimization_raster_mask, args.marginal_value_raster,
                churn_dir),
            task_name=f'sum of {optimization_raster_mask}')
        sum_task_list.append((optimization_raster_mask, sum_of_masked_task))
    target_table_path = os.path.join(
        args.target_dir, f'total_carbon_{marginal_value_id}.csv')
    with open(target_table_path, 'w') as target_table_file:
        for raster_mask_path, sum_task in sum_task_list:
            target_table_file.write(
                f'{sum_task.get()}, {os.path.basename(raster_mask_path)}\n')


    task_graph.join()
    task_graph.close()
    task_graph = None



if __name__ == '__main__':
    main()

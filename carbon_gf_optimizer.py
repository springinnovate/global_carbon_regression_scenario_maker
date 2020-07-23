"""Pick optimization based on gaussian filter pipeline.

Run like this: git pull && docker run --name optimize -v `pwd`:/usr/local/workspace -it --rm therealspring/inspring:latest carbon_gf_optimizer.py --target_dir optimize_regression_results --path_to_forest_mask_data carbon_regression_scenario_workspace/data/-61.0,0.0,-60.0,1.0/ --marginal_value_raster carbon_regression_scenario_workspace/marginal_values/marginal_value_regression_-61.0,0.0,-60.0,1.0.tif --sum

"""
import argparse
import glob
import logging
import multiprocessing
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


def mask_with_range_op(
        value_array, mask_array, min_max_range, target_nodata):
    """Mask value by mask in a valid range.

    Args:
        value_array (numpy.array): arbitrary values
        mask_array (numpy.array): keep values wehre mask_array == 1
        min_max_range (tuple): keep values in value array that are >=
            min_max_range[0] and <= min_max_range[1]
        target_nodata (float): target output nodata.

    Returns:
        value masked by mask in range of min/max.

    """
    result = numpy.empty(value_array.shape)
    result[:] = target_nodata
    valid_mask = (
        (value_array >= min_max_range[0]) &
        (value_array <= min_max_range[1]) &
        (mask_array == 1))
    result[valid_mask] = value_array[valid_mask]
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
        '--n_workers', type=int, default=multiprocessing.cpu_count(),
        help='number of workers to taskgraph')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph(args.target_dir, args.n_workers, 5.0)
    churn_dir = os.path.join(args.target_dir, 'churn')
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass

    # sum marginal value to 30km pixel
    pixel_size_30km = (30/111, -30/111)

    # 1a) mask forest from marginal value map and set values outside of reasonable ranges to nodata
    #   - marginal_value_map, forest_mask -> mv_forest_only.tif
    marginal_value_raster_path = args.marginal_value_raster
    forest_mask_raster_path = os.path.join(
        args.path_to_forest_mask_data,
        'mask_of_forest_10sec_restoration_limited.tif')
    mv_forest_only_raster_path = os.path.join(churn_dir, 'mv_forest_only.tif')
    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(marginal_value_raster_path, 1), (forest_mask_raster_path, 1),
             ((0, 2e5), 'raw'), (NODATA, 'raw')], mask_with_range_op,
            mv_forest_only_raster_path, gdal.GDT_Float32, NODATA),
        target_path_list=[mv_forest_only_raster_path],
        task_name=f'calculate marginal value masked by forest only')

    task_graph.join()
    task_graph.close()
    # 2a) make 30km gaussian kernel
    # 3a) gaussian filter mv_forest_only.tif to 30km
    #   - mv_forest_only.tif, kernel -> mv_forest_only_gf.tif

    # 1b) new_forest_mask
    #   - reforest_forest_mask - esa_forest_mask -> new_forest_mask.tif

    # 1c) mask spatial filter by new forest
    #   -  new_forest_mask.tif * mv_forest_only_gf.tif -> weighted_mv_forest_only.tif

    # 1d) optimize weighted_mv_forest_only.tif up to 100% and save 1% increments
    #   - optimal_masks_{percent}.tif
    # 2d) create new landcover map for each mask
    #   - optimal_landcover_{percent}.tif
    # 3d) "evaluate" optimial_landcover_{percent}.tif with 'mult-by-column'?
    #   - optimal_carbon_{percent}.tif
    # 4d) subtract optimal-carbon_{percent}.tif from esa_carbon
    #   - optimal_carbon_increase_{percent}.tif
    # 5d) sum optimal_carbon_increase_{percent}.tif and print to table.


    return
    # ALL OLD BELOW

    # warp marginal value map to this pixel size using average
    marginal_value_id = os.path.basename(
        os.path.splitext(args.marginal_value_raster)[0])

    # create count of difference of forest masks from esa to restoration
    restoration_mask_raster_path = os.path.join(
        args.path_to_forest_mask_data,
        'mask_of_forest_10sec_restoration_limited.tif')
    esa_mask_raster_path = os.path.join(
        args.path_to_forest_mask_data, 'mask_of_forest_10sec_esa2014.tif')
    new_forest_mask_raster_path = os.path.join(
        churn_dir, f'{marginal_value_id}_new_forest_mask.tif')
    LOGGER.info('count difference of forest mask from esa to restoration')
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

    LOGGER.debug('warp new forest mask to 30km')
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

    LOGGER.debug(
        'divide marginal value by new forest average to get efficiency')
    efficiency_task = task_graph.add_task(
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
    LOGGER.debug('run that optimization on efficiency')
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
        dependent_task_list=[efficiency_task],
        task_name='optimize')

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
    LOGGER.debug('writing result')
    with open(target_table_path, 'w') as target_table_file:
        for raster_mask_path, sum_task in sum_task_list:
            LOGGER.debug(f'waiting for result of {raster_mask_path}')
            target_table_file.write(
                f'{sum_task.get()}, {os.path.basename(raster_mask_path)}\n')

    task_graph.join()
    task_graph.close()
    task_graph = None


if __name__ == '__main__':
    main()

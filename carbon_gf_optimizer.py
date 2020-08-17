"""Pick optimization based on gaussian filter pipeline.

Run like this: git pull && docker run --name optimize -v `pwd`:/usr/local/workspace -it --rm therealspring/inspring:latest carbon_gf_optimizer.py --target_dir optimize_regression_results --path_to_forest_mask_data carbon_regression_scenario_workspace/data/-61.0,0.0,-60.0,1.0/ --marginal_value_raster carbon_regression_scenario_workspace/marginal_values/marginal_value_regression_-61.0,0.0,-60.0,1.0.tif --sum

"""
import argparse
import logging
import multiprocessing
import os
import sys

from osgeo import gdal
import pygeoprocessing
import numpy
import scipy
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
EDGE_EFFECT_DIST_KM = 3  # 1stdev range expected edge effect of carbon forest edge


def where_zero_op(mask_array, value_array, value_nodata):
    """Set value array to nodata where mask is not 1."""
    result = numpy.copy(value_array)
    result[mask_array != 1] = value_nodata
    return result


def mask_new_a(array_a, array_b, nodata):
    """Calc array_a-array_b but ignore nodata."""
    valid_mask = (
        ~numpy.isclose(array_a, nodata) &
        ~numpy.isclose(array_b, nodata))
    result = numpy.empty_like(array_a)
    result[:] = nodata
    result[valid_mask] = (
        (array_a[valid_mask] == 1) &
        (array_b[valid_mask] == 0))
    return result


def mult_const(base_array, constant, nodata):
    """Multiply base by constant but skip nodata."""
    result = numpy.copy(base_array)
    result[~numpy.isclose(base_array, nodata)] *= constant
    return result


def normalize_raster(base_raster_path, constant, target_raster_path):
    """Multiply constant by base raster path."""
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (constant, 'raw'), (base_nodata, 'raw')],
        mult_const, target_raster_path, gdal.GDT_Float32, base_nodata)


def sum_raster(raster_path):
    """Sum raster and return result."""
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    sum_val = 0.0
    for _, data_array in pygeoprocessing.iterblocks((raster_path, 1)):
        sum_val += numpy.sum(data_array[~numpy.isclose(data_array, nodata)])
    return sum_val


def make_kernel_raster(pixel_radius, target_path):
    """Create kernel with given radius to `target_path`."""
    truncate = 4
    size = int(pixel_radius * 2 * truncate + 1)
    step_fn = numpy.zeros((size, size))
    step_fn[size//2, size//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(
        step_fn, pixel_radius, order=0, mode='reflect', cval=0.0,
        truncate=truncate)
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1., -1.), (0.,  0.), None,
        target_path)


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
    nan_inf_mask = numpy.isnan(value_array) | numpy.isinf(value_array)
    value_array[nan_inf_mask] = target_nodata
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
        '--path_to_scenario_forest_mask', help='path_to_scenario_forest_mask',
        required=True)
    parser.add_argument(
        '--path_to_base_forest_mask', help='path_to_base_forest_mask',
        required=True)
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

    # 1a) mask forest from marginal value map and set values outside of reasonable ranges to nodata
    #   - marginal_value_map, forest_mask -> mv_forest_only.tif
    marginal_value_raster_path = args.marginal_value_raster
    forest_mask_raster_path = os.path.join(
        args.path_to_forest_mask_data,
        'mask_of_forest_10sec_restoration_limited.tif')
    marginal_value_forest_raster_path = os.path.join(
        churn_dir, 'marginal_value_forest.tif')
    marginal_value_forest_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(marginal_value_raster_path, 1), (forest_mask_raster_path, 1),
             ((0, 2e5), 'raw'), (NODATA, 'raw')], mask_with_range_op,
            marginal_value_forest_raster_path, gdal.GDT_Float32, NODATA),
        target_path_list=[marginal_value_forest_raster_path],
        task_name=f'calculate marginal value masked by forest only')

    pixel_length = pygeoprocessing.get_raster_info(
        marginal_value_raster_path)['pixel_size'][0]

    # 2a) make 3km gaussian kernel
    # pixel_length is in degrees and we want about a 30km decay so do that:
    # (deg/pixel  * km/deg * 1/30km)^-1
    # ~111km / degree
    pixel_radius = (pixel_length * 111 / EDGE_EFFECT_DIST_KM)**-1
    kernel_raster_path = os.path.join(churn_dir, f'kernel_{pixel_radius}.tif')
    kernel_task = task_graph.add_task(
        func=make_kernel_raster,
        args=(pixel_radius, kernel_raster_path),
        target_path_list=[kernel_raster_path],
        task_name=f'make kernel of radius {pixel_radius}')

    # 3a) gaussian filter mv_forest_only.tif to 30km
    #   - mv_forest_only.tif, kernel -> mv_forest_only_gf.tif
    marginal_value_forest_gf_raster_path = os.path.join(
        churn_dir, 'marginal_value_forest_gf.tif')
    convolution_task = task_graph.add_task(
        func=pygeoprocessing.convolve_2d,
        args=(
            (marginal_value_forest_raster_path, 1), (kernel_raster_path, 1),
            marginal_value_forest_gf_raster_path),
        dependent_task_list=[marginal_value_forest_task, kernel_task],
        target_path_list=[marginal_value_forest_gf_raster_path],
        task_name=(
            f'gaussian filter the marginal value'))

    # 4a) normalize marginal_value_forest_gf by *
    # sum(marginal_value_forest)/sum(marginal_value_forest_gf)

    mv_sum_map = {}
    for mv_raster_id, mv_raster_path, dependent_task_list in [
            ('mv_forest', marginal_value_forest_raster_path,
             [marginal_value_forest_task]),
            ('mv_forest_gf', marginal_value_forest_gf_raster_path,
             [convolution_task])]:
        mv_sum_map[mv_raster_id] = task_graph.add_task(
            func=sum_raster,
            args=(mv_raster_path,),
            dependent_task_list=dependent_task_list,
            task_name=f'sum raster {mv_raster_id}')

    norm_marginal_value_forest_gf_path = os.path.join(
        churn_dir, 'norm_marginal_value_forest_gf.tif')
    norm_mv_gf_task = task_graph.add_task(
        func=normalize_raster,
        args=(
            marginal_value_forest_gf_raster_path,
            mv_sum_map['mv_forest'].get()/mv_sum_map['mv_forest_gf'].get(),
            norm_marginal_value_forest_gf_path),
        dependent_task_list=[convolution_task],
        target_path_list=[norm_marginal_value_forest_gf_path],
        task_name='normalize marginal value forest gf')

    # 1b) new_forest_mask
    #   - reforest_forest_mask - esa_forest_mask -> new_forest_mask.tif
    mask_info = pygeoprocessing.get_raster_info(
        args.scenario_forest_mask_raster_path)
    mask_nodata = mask_info['nodata'][0]

    new_forest_mask_raster_path = os.path.join(
        churn_dir, 'new_forest_mask.tif')

    new_forest_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(args.path_to_scenario_forest_mask, 1),
             (args.path_to_base_forest_mask, 1),
             (mask_nodata, 'raw')], mask_new_a,
            new_forest_mask_raster_path, mask_info['datatype'], mask_nodata),
        target_path_list=[new_forest_mask_raster_path],
        task_name=f'calculate new forest mask')

    # * (mult new_forest_mask, norm_marginal_value_forest_gf) norm_marginal_value_new_forest_gf
    norm_mv_gf_task.join()
    norm_mv_nodata = pygeoprocessing.get_raster_info(
        norm_marginal_value_forest_gf_path)['nodata'][0]
    norm_marginal_value_new_forest_gf = os.path.join(
        churn_dir, 'norm_marginal_value_new_forest_gf.tif')
    set_non_new_forest_to_zero_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(new_forest_mask_raster_path, 1),
             (norm_marginal_value_forest_gf_path, 1),
             (norm_mv_nodata, 'raw')], where_zero_op,
            norm_marginal_value_new_forest_gf, gdal.GDT_Float32,
            norm_mv_nodata),
        dependent_task_list=[new_forest_mask_task, norm_mv_gf_task],
        target_path_list=[norm_marginal_value_new_forest_gf],
        task_name='norm_marginal_value_new_forest_gf')

    # optimize
    LOGGER.debug('run that optimization on efficiency')
    optimize_dir = os.path.join(args.target_dir, 'optimize_rasters_v2')
    try:
        os.makedirs(optimize_dir)
    except OSError:
        pass
    with open(os.path.join(
            optimize_dir,
            f'''sum_of_{os.path.basename(os.path.splitext(
                norm_marginal_value_new_forest_gf)[0])}'''), 'w') as sum_file:
        sum_file.write(f"{mv_sum_map['mv_forest'].get()}\n")
    task_graph.join()
    task_graph.add_task(
        func=pygeoprocessing.raster_optimization,
        args=(
            [(norm_marginal_value_new_forest_gf, 1)], churn_dir,
            optimize_dir),
        kwargs={
            'goal_met_cutoffs': [float(x)/100.0 for x in range(1, 101)],
            'heap_buffer_size': 2**28+2,
            'target_suffix': f'{EDGE_EFFECT_DIST_KM}km',
            'ffi_buffer_size': 2**10,
            },
        transient_run=True,
        dependent_task_list=[set_non_new_forest_to_zero_task],
        task_name='optimize')

    task_graph.join()
    task_graph.close()
    task_graph = None


if __name__ == '__main__':
    main()

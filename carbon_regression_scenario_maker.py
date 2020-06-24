"""Create carbon regression scenarios."""
import collections
import logging
import multiprocessing
import os
import subprocess
import sys

from osgeo import gdal
import ecoshard
import pygeoprocessing
import numpy
import taskgraph

import justin_gaussian_kernel
import mult_by_columns_library

gdal.SetCacheMax(2**27)

# treat this one column name as special for the y intercept
N_CPUS = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

WORKSPACE_DIR = 'carbon_regression_scenario_workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')

CONVOLUTION_PIXEL_DIST_LIST = [1, 2, 3, 5, 10, 20, 30, 50, 100]

CROPLAND_LULC_CODES = range(10, 41)
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

SCENARIO_LIST = [
    ('is_cropland', CROPLAND_LULC_CODES, ''),
    ('is_urban', URBAN_LULC_CODES, ''),
    ('not_forest', FOREST_CODES, 'inv')]

BASE_DATA_BUCKET_ROOT = 'gs://ecoshard-root/global_carbon_regression/inputs/'
LULC_URL_LIST = [
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/PNV_jsmith_060420_md5_8dd464e0e23fefaaabe52e44aa296330.tif']

LASSO_TABLE_URI = 'gs://ecoshard-root/global_carbon_regression/lasso_interacted_not_forest_gs1to100_nonlinear_alpha0-0001_params_namefix.csv'
LASSO_TABLE_PATH = os.path.join(
    ECOSHARD_DIR, os.path.basename(LASSO_TABLE_URI))
# The following is the base in the pattern found in the lasso table
# [base]_[mask_type]_gs[kernel_size]
BASE_LASSO_CONVOLUTION_RASTER_NAME = 'lulc_esa_smoothed_2014_10sec'

def make_kernel_raster(pixel_radius, target_path):
    """Create kernel with given radius to `target_path`."""
    kernel_array = justin_gaussian_kernel.get_array_from_two_dim_first_order_kernel_function(
        pixel_radius, 1, 5)
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1., -1.), (0.,  0.), None,
        target_path)


def _mask_vals_op(array, nodata, valid_1d_array, inverse, target_nodata):
    result = numpy.zeros(array.shape, dtype=numpy.uint8)
    if nodata is not None:
        nodata_mask = numpy.isclose(array, nodata)
    else:
        nodata_mask = numpy.zeros(array.shape, dtype=numpy.bool)
    valid_mask = numpy.in1d(array, valid_1d_array).reshape(result.shape)
    if inverse:
        valid_mask = ~valid_mask
    result[valid_mask & ~nodata_mask] = 1
    result[nodata_mask] = target_nodata
    return result


def mask_ranges(base_raster_path, range_tuple, inverse, target_raster_path):
    """Mask all values in the given inclusive range to 1, the rest to 0."""
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    target_nodata = 2
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (base_nodata, 'raw'),
         (range_tuple, 'raw'), (inverse, 'raw'),
         (target_nodata, 'raw')], _mask_vals_op,
        target_raster_path, gdal.GDT_Byte, target_nodata)


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR, CHURN_DIR, DATA_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(CHURN_DIR, N_CPUS, 5.0)

    # download inputs
    download_inputs_task = task_graph.add_task(
        func=subprocess.run,
        args=('/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp -nr %s %s' % (
            BASE_DATA_BUCKET_ROOT, f'{DATA_DIR}/'),),
        kwargs={'shell': True, 'check': True},
        task_name='download inputs')

    # download lasso table
    download_lasso_task = task_graph.add_task(
        func=subprocess.run,
        args=(
            f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp -n '
            f'{LASSO_TABLE_URI} {LASSO_TABLE_PATH}',),
        kwargs={'shell': True, 'check': True},
        target_path_list=[LASSO_TABLE_PATH],
        task_name='download lasso table')

    lulc_scenario_raster_path_list = []
    dl_lulc_task_map = {}
    for lulc_url in LULC_URL_LIST:
        LOGGER.debug(f'download {lulc_url}')
        lulc_raster_path = os.path.join(
            ECOSHARD_DIR, os.path.basename(lulc_url))
        dl_lulc_task_map[lulc_raster_path] = task_graph.add_task(
            func=ecoshard.download_url,
            args=(lulc_url, lulc_raster_path),
            target_path_list=[lulc_raster_path],
            task_name=f'download {lulc_url}')
        lulc_scenario_raster_path_list.append(lulc_raster_path)

    scenario_mask_task_map = {}

    for lulc_scenario_raster_path in lulc_scenario_raster_path_list:
        lulc_basename = os.path.splitext(os.path.basename(
            lulc_scenario_raster_path))[0]
        for scenario_id, lulc_codes, inverse_mode in SCENARIO_LIST:
            scenario_lulc_mask_raster_path = os.path.join(
                CHURN_DIR, f'{scenario_id}_{lulc_basename}.tif')
            mask_task = task_graph.add_task(
                func=mask_ranges,
                args=(
                    lulc_scenario_raster_path, lulc_codes,
                    inverse_mode == 'inv', scenario_lulc_mask_raster_path),
                dependent_task_list=[
                    dl_lulc_task_map[lulc_scenario_raster_path]],
                target_path_list=[scenario_lulc_mask_raster_path],
                task_name=f'make {scenario_id}_{lulc_basename}')
            scenario_mask_task_map[scenario_id] = (
                scenario_lulc_mask_raster_path, mask_task)
            LOGGER.debug(
                f'this is the scenario lulc mask target: '
                f'{scenario_lulc_mask_raster_path}')

        kernel_raster_path_map = {}

        for pixel_radius in CONVOLUTION_PIXEL_DIST_LIST:
            kernel_raster_path = os.path.join(
                CHURN_DIR, f'{pixel_radius}_kernel.tif')
            kernel_task = task_graph.add_task(
                func=make_kernel_raster,
                args=(pixel_radius, kernel_raster_path),
                target_path_list=[kernel_raster_path],
                task_name=f'make kernel of radius {pixel_radius}')
            kernel_raster_path_map[pixel_radius] = kernel_raster_path
            convolution_task_list = []
            for scenario_id in scenario_mask_task_map:
                scenario_mask_path, mask_task = \
                    scenario_mask_task_map[scenario_id]
                LOGGER.debug(
                    f'this is the scenario mask about to convolve: '
                    f'{scenario_mask_path} {mask_task}')
                convolution_mask_raster_path = os.path.join(
                    DATA_DIR,
                    f'{lulc_basename}_{scenario_id}_{pixel_radius}.tif')
                convolution_task = task_graph.add_task(
                    func=pygeoprocessing.convolve_2d,
                    args=(
                        (scenario_mask_path, 1), (kernel_raster_path, 1),
                        convolution_mask_raster_path),
                    dependent_task_list=[mask_task, kernel_task],
                    target_path_list=[convolution_mask_raster_path],
                    task_name=f'convolve {scenario_id}_{lulc_basename}')
                convolution_task_list.append(convolution_task)

        target_result_path = os.path.join(
            WORKSPACE_DIR, f'lasso_eval_{lulc_basename}.tif')
        lasso_mult_workspace_dir = os.path.join(
            WORKSPACE_DIR, lulc_basename)
        task_graph.add_task(
            func=mult_by_columns_library.mult_by_columns,
            args=(
                LASSO_TABLE_PATH, DATA_DIR, lasso_mult_workspace_dir,
                BASE_LASSO_CONVOLUTION_RASTER_NAME, lulc_basename, None,
                0.002777777777777777884, target_result_path),
            dependent_task_list=convolution_task_list + [
                download_inputs_task, download_lasso_task],
            target_path_list=[target_result_path],
            task_name=f'lasso eval of {lulc_basename}')

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

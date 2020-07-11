"""Create carbon regression scenarios."""
import argparse
import collections
import glob
import logging
import multiprocessing
import os
import re
import subprocess
import sys

from osgeo import gdal
import ecoshard
import pygeoprocessing
import numpy
import scipy.ndimage
import taskgraph

import mult_by_columns_library

gdal.SetCacheMax(2**27)

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

MASK_TYPES = [
    ('is_cropland_10sec', CROPLAND_LULC_CODES, ''),
    ('is_urban_10sec', URBAN_LULC_CODES, ''),
    ('not_forest_10sec', FOREST_CODES, 'inv'),
    ('forest_10sec', FOREST_CODES, '')]
MASK_NODATA = 2
MULT_BY_COLUMNS_NODATA = numpy.finfo('float32').min

BASE_DATA_BUCKET_ROOT = 'gs://ecoshard-root/global_carbon_regression/inputs/'

LULC_SCENARIO_URI_MAP = {
    'esa2014': 'gs://ecoshard-root/global_carbon_regression/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif',
    'pnv_jsmith_060420': 'gs://ecoshard-root/global_carbon_regression/PNV_jsmith_060420_md5_8dd464e0e23fefaaabe52e44aa296330.tif',
    'restoration_limited': 'gs://nci-ecoshards/scenarios050420/restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif',
}
TARGET_PIXEL_SIZE = (10./3600., -10./3600.)
FOREST_REGRESSION_LASSO_TABLE_URI = 'gs://ecoshard-root/global_carbon_regression/lasso_interacted_not_forest_gs1to100_nonlinear_alpha0-0001_params_namefix.csv'
NON_FOREST_REGRESSION_LASSO_TABLES_URL = 'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/results_2020-07-07b_md5_fe89d44cf181486b384beb432c253d47.zip'
NON_FOREST_REGRESSION_LASSO_TABLES_DIR = os.path.join(
    ECOSHARD_DIR, 'non_forest_regression_tables')
FOREST_REGRESSION_LASSO_TABLE_PATH = os.path.join(
    ECOSHARD_DIR, os.path.basename(FOREST_REGRESSION_LASSO_TABLE_URI))
# The following is the base in the pattern found in the lasso table
# [base]_[mask_type]_gs[kernel_size]
BASE_LASSO_CONVOLUTION_RASTER_NAME = 'lulc_esa_smoothed_2014_10sec'
LULC_SCENARIO_RASTER_PATH_MAP = {}


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


def mult_op(array_a, array_b, nodata_a, nodata_b, target_nodata):
    """Mult a*b and account for nodata."""
    result = numpy.empty(array_a.shape)
    result[:] = target_nodata
    valid_mask = (
        ~numpy.isclose(array_a, nodata_a) & ~numpy.isclose(array_b, nodata_b))
    result[valid_mask] = array_a[valid_mask] * array_b[valid_mask]
    return result


def mask_ranges(
        base_raster_path, mask_value_list, inverse, target_raster_path):
    """Mask all values in the given inclusive range to 1, the rest to 0.

    Args:
        base_raster_path (str): path to an integer raster
        mask_value_list (list): lits of integer codes to set output to 1 or 0
            if it is ccontained in the list.
        inverse (bool): if true the inverse of the `mask_value_list` is used
            so that values not in the list are set to 1 and those within are
            set to 0.
        target_raster_path (str): path to output mask, will contain 1 or 0
            whether the base had a pixel value contained in the
            `mask_value_list` while accounting for `inverse`.

    Returns:
        None.

    """
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (base_nodata, 'raw'),
         (mask_value_list, 'raw'), (inverse, 'raw'),
         (MASK_NODATA, 'raw')], _mask_vals_op,
        target_raster_path, gdal.GDT_Byte, MASK_NODATA)


def download_and_clip(file_uri, download_dir, bounding_box, target_file_path):
    """Download file uri, then clip it. Will hardlink if no clip is necessary.

    Will download and keep original files to `download_dir`.

    Args:
        file_uri (str): uri to file to download
        download_dir (str): path to download directory that can will be used
            to hold and keep the base downloaded file before clipping.
        bounding_box (list): if not none, clip the result to target_file_path,
            otherwise copy the result to target_file_path.
        target_file_path (str): desired target of clipped file

    Returns:
        None.

    """
    try:
        os.makedirs(download_dir)
    except OSError:
        pass

    base_filename = os.path.basename(file_uri)
    base_file_path = os.path.join(download_dir, base_filename)

    # Wrapping this in a taskgraph prevents us from re-downloading a large
    # file if it's already been clipped before.
    LOGGER.debug(f'download {file_uri} to {base_file_path}')
    subprocess.run(
        f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp -nr '
        f'{file_uri} {download_dir}/', shell=True, check=True)

    raster_info = pygeoprocessing.get_raster_info(base_file_path)
    if bounding_box != raster_info['bounding_box']:
        LOGGER.debug(
            f'bounding box and desired target differ '
            f"{bounding_box} {raster_info['bounding_box']}")
        pygeoprocessing.warp_raster(
            base_file_path, raster_info['pixel_size'], target_file_path,
            'near', target_bb=bounding_box)
    else:
        # it's already the same size so no need to warp it
        LOGGER.debug('already the same size, so no need to warp')
        os.link(base_file_path, target_file_path)


def fetch_data(bounding_box, clipped_data_dir, task_graph):
    """Download all the global data needed to run this analysis.

    Args:
        bounding_box (list): minx, miny, maxx, maxy list to clip to
        clipped_data_dir (str): path to directory to copy clipped rasters
            to
        task_graph (TaskGraph): taskgraph object to schedule work.

    Returns:
        None.

    """
    files_to_download = subprocess.check_output([
        '/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil ls '
        'gs://ecoshard-root/global_carbon_regression/inputs'],
        shell=True).decode('utf-8').splitlines()

    LOGGER.debug(f'here are the files to download: {files_to_download}')

    try:
        os.makedirs(clipped_data_dir)
    except OSError:
        pass

    for file_uri in files_to_download:
        if not file_uri.endswith('tif'):
            continue
        clipped_file_path = os.path.join(
            clipped_data_dir, os.path.basename(file_uri))
        _ = task_graph.add_task(
            func=download_and_clip,
            args=(
                file_uri, DATA_DIR, bounding_box, clipped_file_path),
            target_path_list=[clipped_file_path],
            task_name=(
                f'download and clip contents of {file_uri} to '
                f'{clipped_data_dir}'))
    task_graph.join()

    download_forest_regression_lasso_task = task_graph.add_task(
        func=subprocess.run,
        args=(
            f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp -n '
            f'{FOREST_REGRESSION_LASSO_TABLE_URI} '
            f'{FOREST_REGRESSION_LASSO_TABLE_PATH}',),
        kwargs={'shell': True, 'check': True},
        target_path_list=[FOREST_REGRESSION_LASSO_TABLE_PATH],
        task_name='download forest regresstion lasso table')

    try:
        os.makedirs(NON_FOREST_REGRESSION_LASSO_TABLES_DIR)
    except OSError:
        pass

    download_non_forest_regression_lasso_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(
            NON_FOREST_REGRESSION_LASSO_TABLES_URL,
            NON_FOREST_REGRESSION_LASSO_TABLES_DIR),
        task_name='download non forest regresstion lasso table')

    global LULC_SCENARIO_RASTER_PATH_MAP
    for scenario_id, lulc_uri in LULC_SCENARIO_URI_MAP.items():
        LOGGER.debug(f'download {lulc_uri}')
        lulc_raster_path = os.path.join(
            ECOSHARD_DIR, os.path.basename(lulc_uri))
        clipped_file_path = os.path.join(
            clipped_data_dir, os.path.basename(lulc_raster_path))
        _ = task_graph.add_task(
            func=download_and_clip,
            args=(
                lulc_uri, ECOSHARD_DIR, bounding_box, clipped_file_path),
            target_path_list=[clipped_file_path],
            task_name=(
                f'download and clip contents of {lulc_uri} to '
                f'{clipped_data_dir}'))
        LULC_SCENARIO_RASTER_PATH_MAP[scenario_id] = clipped_file_path

    task_graph.join()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Global carbon regression scenario maker')
    parser.add_argument(
        '--bounding_box', type=float, nargs=4, default=[-180, -90, 180, 90],
        help=(
            "manual bounding box in the form of four consecutive floats: "
            "min_lng, min_lat, max_lng, max_lat, ex: "
            "-180.0, -58.3, 180.0, 81.5"))
    parser.add_argument(
        '--keyfile', help='path to keyfile that authorizes bucket access')
    parser.add_argument(
        '--n_workers', type=int, default=multiprocessing.cpu_count(),
        help='how many workers to allocate to taskgraph')
    args = parser.parse_args()

    if args.keyfile:
        subprocess.run(
            f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gcloud auth '
            f'activate-service-account --key-file={args.keyfile}',
            shell=True, check=True)

    for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR, CHURN_DIR, DATA_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    bounding_box_str = ','.join([str(x) for x in args.bounding_box])
    clipped_data_dir = os.path.join(DATA_DIR, bounding_box_str)
    # Step 0: Download data
    task_graph = taskgraph.TaskGraph(CHURN_DIR, args.n_workers, 5.0)
    LOGGER.info("Step 0: Download data")
    fetch_data(args.bounding_box, clipped_data_dir, task_graph)

    # FOREST REGRESSION

    # 1) Make convolutions with custom kernel of 1, 2, 3, 5, 10, 20, 30, 50,
    #    and 100 pixels for not_forest (see forest lulc codes), is_cropland
    #    (classes 10-40), and is_urban (class 190) for LULC maps

    LOGGER.info("Forest Regression step 1")
    mask_path_task_map = collections.defaultdict(dict)
    for scenario_id, lulc_scenario_raster_path in \
            LULC_SCENARIO_RASTER_PATH_MAP.items():
        for mask_type, lulc_codes, inverse_mode in MASK_TYPES:
            scenario_lulc_mask_raster_path = os.path.join(
                clipped_data_dir, f'mask_of_{mask_type}_{scenario_id}.tif')
            mask_task = task_graph.add_task(
                func=mask_ranges,
                args=(
                    lulc_scenario_raster_path, lulc_codes,
                    inverse_mode == 'inv', scenario_lulc_mask_raster_path),
                target_path_list=[scenario_lulc_mask_raster_path],
                task_name=f'make {mask_type}_{scenario_id}')
            mask_path_task_map[scenario_id][mask_type] = (
                scenario_lulc_mask_raster_path, mask_task)
            LOGGER.debug(
                f'this is the scenario lulc mask target: '
                f'{scenario_lulc_mask_raster_path}')

        kernel_raster_path_map = {}

        for pixel_radius in reversed(sorted(CONVOLUTION_PIXEL_DIST_LIST)):
            kernel_raster_path = os.path.join(
                CHURN_DIR, f'{pixel_radius}_kernel.tif')
            kernel_task = task_graph.add_task(
                func=make_kernel_raster,
                args=(pixel_radius, kernel_raster_path),
                target_path_list=[kernel_raster_path],
                task_name=f'make kernel of radius {pixel_radius}')
            kernel_raster_path_map[pixel_radius] = kernel_raster_path
            convolution_task_list = []
            for mask_type in mask_path_task_map[scenario_id]:
                scenario_mask_path, mask_task = \
                    mask_path_task_map[scenario_id][mask_type]
                LOGGER.debug(
                    f'this is the scenario mask about to convolve: '
                    f'{scenario_mask_path} {mask_task}')
                convolution_mask_raster_path = os.path.join(
                    clipped_data_dir,
                    f'{scenario_id}_{mask_type}_gs{pixel_radius}.tif')
                convolution_task = task_graph.add_task(
                    func=pygeoprocessing.convolve_2d,
                    args=(
                        (scenario_mask_path, 1), (kernel_raster_path, 1),
                        convolution_mask_raster_path),
                    dependent_task_list=[mask_task, kernel_task],
                    target_path_list=[convolution_mask_raster_path],
                    task_name=(
                        f'convolve {pixel_radius} {mask_type}_'
                        f'{scenario_id}'))
                convolution_task_list.append(convolution_task)
    task_graph.join()

    # 2) Apply the mult_rasters_by_columns.py script to `FOREST_REGRESSION_LASSO_TABLE_PATH` and
    #    the new convolution rasters for forest classes only.
    LOGGER.info("Forest Regression step 2")

    mult_by_columns_workspace = os.path.join(WORKSPACE_DIR, bounding_box_str)
    try:
        os.makedirs(mult_by_columns_workspace)
    except OSError:
        pass
    task_graph.join()
    lasso_eval_for_esa2014_path = os.path.join(
        CHURN_DIR, f'pre_mask_forest_regression_esa2014_eval_{bounding_box_str}.tif')
    mult_by_columns_library.mult_by_columns(
        FOREST_REGRESSION_LASSO_TABLE_PATH, clipped_data_dir,
        mult_by_columns_workspace,
        'lulc_esa_smoothed_2014_10sec', 'esa2014',
        args.bounding_box, TARGET_PIXEL_SIZE, lasso_eval_for_esa2014_path,
        task_graph, zero_nodata=False,
        target_nodata=MULT_BY_COLUMNS_NODATA)

    # TODO -- multiply lasso_eval for_esa2014 by scenario_lulc_mask_raster_path
    # _ = task_graph.add_task(
    LOGGER.info('mask forest values')
    scenario_lulc_forest_mask_raster_path = os.path.join(
        clipped_data_dir, f'mask_of_forest_10sec_esa2014.tif')
    forest_regression_esa2014_eval_raster_path = os.path.join(
        WORKSPACE_DIR, f'forest_regression_esa2014_eval_{bounding_box_str}.tif')
    pygeoprocessing.raster_calculator(
        [(scenario_lulc_forest_mask_raster_path, 1),
         (lasso_eval_for_esa2014_path, 1),
         (MASK_NODATA, 'raw'),
         (MULT_BY_COLUMNS_NODATA, 'raw'),
         (MULT_BY_COLUMNS_NODATA, 'raw'), ],
        mult_op, forest_regression_esa2014_eval_raster_path, gdal.GDT_Float32,
        MULT_BY_COLUMNS_NODATA)

    # NON-FOREST REGRESSION

    # Apply non-forest regression to non-forest classes using
    # mult_rasters_by_columns. Each of the non-forest LULC classes are
    # interacted with each of the predictor variables, so in order to run the
    # mult_rasters script on the ESA or PNV scenario maps, masks of each of
    # those LULC classes will need to be made as files called
    # lulc_esacci_2014_smoothed_class_10 - _220 (collapsing all the 11, 12 etc
    # back to its 10's place).

    LOGGER.info('create the lulc_esacci_2014_smoothed_class_* rasters')

    # TODO: run the mult_by_columns on esa2014 and PNG and maybe the other
    # scenario but collapose their landcover rasters down to the right class
    # or whatever with the masks collapsed
    for scenario_id, lulc_raster_path in LULC_SCENARIO_URI_MAP.items():
        for class_id in range(10, 221, 10):
            # this path will ensure it's picked up later by the mult_by_columns
            # function
            collapsed_class_raster_path = os.path.join(
                clipped_data_dir,
                f'{scenario_id}_{class_id}.tif')
            mask_task = task_graph.add_task(
                func=mask_ranges,
                args=(
                    lulc_raster_path,
                    list(range(class_id, class_id+11)), '',
                    collapsed_class_raster_path),
                target_path_list=[collapsed_class_raster_path],
                task_name=f'eval for collapsed_class_raster_path')
        for lasso_table_path in glob.glob(
                os.path.join(NON_FOREST_REGRESSION_LASSO_TABLES_DIR, '*.csv')):
            alpha = re.match(
                '.*alpha(.*)_.*', os.path.basename(lasso_table_path)).group(1)
            pre_mask_non_forest_regression_eval_raster_path = os.path.join(
                CHURN_DIR,
                f'pre_mask_non_forest_regression_{scenario_id}_{alpha}_'
                f'{bounding_box_str}.tif')
            mult_by_columns_library.mult_by_columns(
                lasso_table_path, clipped_data_dir,
                mult_by_columns_workspace,
                'lulc_esacci_2014_smoothed_class', scenario_id,
                args.bounding_box, TARGET_PIXEL_SIZE,
                pre_mask_non_forest_regression_eval_raster_path,
                task_graph, zero_nodata=False,
                target_nodata=MULT_BY_COLUMNS_NODATA)

            non_forest_regression_eval_raster_path = os.path.join(
                WORKSPACE_DIR,
                f'non_forest_regression_{scenario_id}_{alpha}_'
                f'{bounding_box_str}.tif')

            not_forest_mask_raster_path = os.path.join(
                clipped_data_dir,
                f'mask_of_not_forest_10sec_{scenario_id}.tif')

            pygeoprocessing.raster_calculator(
                [(not_forest_mask_raster_path, 1),
                 (pre_mask_non_forest_regression_eval_raster_path, 1),
                 (MASK_NODATA, 'raw'),
                 (MULT_BY_COLUMNS_NODATA, 'raw'),
                 (MULT_BY_COLUMNS_NODATA, 'raw'), ],
                mult_op, non_forest_regression_eval_raster_path,
                gdal.GDT_Float32, MULT_BY_COLUMNS_NODATA)

    task_graph.join()

    # SCENARIOS/OPTIMIZATION

    # 1) Standard approach: the IPCC approach will be applied for ESA 2014 and
    #    to the forest pixels only of a Potential Natural Vegetation (PNV) map.
    #    An IPCC-based marginal value map will be created as the difference
    #    between the two, and pixels selected by the largest marginal value
    #    until the 3 Pg target is reached.

    # 2) For the regression approach, the forest regression model will be
    #    applied to the forest pixels and the non-forest regression model will
    #    be applied to the non-forest pixels. The regression will also be
    #    applied to the same PNV map for forest pixels only. The difference
    #    between the two will create a regression-based marginal value map. In
    #    this case, because the aim is to select for areas not only of high
    #    marginal value for reforestation but also regeneration, a 30 km
    #    resolution grid will be used to summarize values with edge effects
    #    (since 30 km was the largest scale over which edge effects were seen to
    #    operate). The marginal values will be summed and divided by the
    #    difference in the number of forest pixels between PNV and ESA 2014-this
    #    ratio can be seen as the "efficiency" of intervention in that 30 km
    #    grid cell. Highest efficiency grid cells will be selected first, with
    #    all viable non-forest pixels within them restored, until the 3 Pg
    #    target is reached.


    # target_result_path = os.path.join(
    #     WORKSPACE_DIR, f'lasso_eval_{lulc_basename}.tif')
    # lasso_mult_workspace_dir = os.path.join(
    #     WORKSPACE_DIR, lulc_basename)
    # task_graph.add_task(
    #     func=mult_by_columns_library.mult_by_columns,
    #     args=(
    #         FOREST_REGRESSION_LASSO_TABLE_PATH, clipped_data_dir, lasso_mult_workspace_dir,
    #         BASE_LASSO_CONVOLUTION_RASTER_NAME, lulc_basename, None,
    #         0.002777777777777777884, target_result_path),
    #     dependent_task_list=convolution_task_list + [
    #         download_inputs_task, download_lasso_task],
    #     target_path_list=[target_result_path],
    #     task_name=f'lasso eval of {lulc_basename}')

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

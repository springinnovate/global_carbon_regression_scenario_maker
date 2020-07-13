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
MULT_BY_COLUMNS_NODATA = -1

ZERO_NODATA_SYMBOLS = {
    'population_2015_5min',
    'population_2015_30sec',
    'clay_10sec',
    'bdod_10sec',
    'ocd_10sec',
    'soc_10sec',
    'silt_10sec',
    'sand_10sec',
    }

CARBON_ZONES_VECTOR_URI = 'gs://ecoshard-root/global_carbon_regression/carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d.gpkg'
CARBON_ZONES_VECTOR_PATH = os.path.join(ECOSHARD_DIR, 'carbon_zones.gpkg')
BASE_DATA_BUCKET_ROOT = 'gs://ecoshard-root/global_carbon_regression/inputs/'

LULC_SCENARIO_URI_MAP = {
    'esa2014': 'gs://ecoshard-root/global_carbon_regression/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif',
    'restoration_limited': 'gs://nci-ecoshards/scenarios050420/restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif',
}
TARGET_PIXEL_SIZE = (10./3600., -10./3600.)
FOREST_REGRESSION_LASSO_TABLE_URI = 'gs://ecoshard-root/global_carbon_regression/lasso_interacted_not_forest_gs1to100_nonlinear_alpha0-0001_params_namefix.csv'
NON_FOREST_REGRESSION_LASSO_TABLES_URL = 'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/results_2020-07-07b_md5_fe89d44cf181486b384beb432c253d47.zip'
NON_FOREST_REGRESSION_LASSO_TABLES_DIR = os.path.join(
    ECOSHARD_DIR, 'non_forest_regression_tables')
IPCC_CARBON_TABLE_URI = 'gs://ecoshard-root/global_carbon_regression/IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7.csv'
IPCC_CARBON_TABLE_PATH = os.path.join(
    ECOSHARD_DIR, os.path.basename(IPCC_CARBON_TABLE_URI))
FOREST_REGRESSION_LASSO_TABLE_PATH = os.path.join(
    ECOSHARD_DIR, os.path.basename(FOREST_REGRESSION_LASSO_TABLE_URI))
# The following is the base in the pattern found in the lasso table
# [base]_[mask_type]_gs[kernel_size]
BASE_LASSO_CONVOLUTION_RASTER_NAME = 'lulc_esa_smoothed_2014_10sec'
LULC_SCENARIO_RASTER_PATH_MAP = {}


def sub_pos_op(array_a, array_b):
    """Assume nodata value is negative and the same for a and b."""
    result = array_a.copy()
    mask = array_b > 0
    result[mask] -= array_b[mask]
    return array_b


def where_op(condition_array, if_true_array, else_array):
    result = numpy.copy(else_array)
    mask = condition_array == 1
    result[mask] = if_true_array[mask]
    return result


def raster_where(
        condition_raster_path, if_true_raster_path, else_raster_path,
        target_raster_path):
    pygeoprocessing.raster_calculator(
        [(condition_raster_path, 1), (if_true_raster_path, 1),
         (else_raster_path, 1)], where_op, target_raster_path,
        gdal.GDT_Float32,
        pygeoprocessing.get_raster_info(if_true_raster_path)['nodata'][0])


def ipcc_carbon_op(
        lulc_array, zones_array, zone_lulc_to_carbon_map, conversion_factor):
    result = numpy.zeros(lulc_array.shape)
    for zone_id in numpy.unique(zones_array):
        zone_mask = zones_array == zone_id
        result[zone_mask] = \
            zone_lulc_to_carbon_map[zone_id][lulc_array[zone_mask]] * conversion_factor
    return result


def parse_carbon_lulc_table(ipcc_carbon_table_path):
    """Custom func to parse out the IPCC carbon table by zone and lulc."""
    with open(IPCC_CARBON_TABLE_PATH, 'r') as carbon_table_file:
        header_line = carbon_table_file.readline()
        lulc_code_list = [int(lucode) for lucode in header_line.split(',')[1:]]
        max_code = max(lulc_code_list)

        zone_lucode_to_carbon_map = {}
        for line in carbon_table_file:
            split_line = line.split(',')
            if split_line[0] == '':
                continue
            zone_id = int(split_line[0])
            zone_lucode_to_carbon_map[zone_id] = numpy.zeros(max_code+1)
            for lucode, carbon_value in zip(lulc_code_list, split_line[1:]):
                zone_lucode_to_carbon_map[zone_id][lucode] = float(carbon_value)
    return zone_lucode_to_carbon_map


def rasterize_carbon_zones(
        base_raster_path, carbon_vector_path, rasterized_zones_path):
    """Rasterize carbon zones, expect 'CODE' as the parameter in the vector."""
    pygeoprocessing.new_raster_from_base(
        base_raster_path, rasterized_zones_path, gdal.GDT_Int32,
        [-1])
    pygeoprocessing.rasterize(
        carbon_vector_path, rasterized_zones_path,
        option_list=['ATTRIBUTE=CODE'])


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

    for data_uri, data_path in [
            (CARBON_ZONES_VECTOR_URI, CARBON_ZONES_VECTOR_PATH),
            (FOREST_REGRESSION_LASSO_TABLE_URI,
             FOREST_REGRESSION_LASSO_TABLE_PATH),
            (IPCC_CARBON_TABLE_URI, IPCC_CARBON_TABLE_PATH)]:
        _ = task_graph.add_task(
            func=subprocess.run,
            args=(
                f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp -n '
                f'{data_uri} {data_path}',),
            kwargs={'shell': True, 'check': True},
            target_path_list=[data_path],
            task_name=f'download {data_uri}')

    try:
        os.makedirs(NON_FOREST_REGRESSION_LASSO_TABLES_DIR)
    except OSError:
        pass

    _ = task_graph.add_task(
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

    # IPCC Approach
    # Create carbon stocks for ESA 2014 and restoration scenario
    rasterize_carbon_zone_task = None
    ipcc_carbon_scenario_raster_map = {}
    IPCC_CARBON_DIR = os.path.join(WORKSPACE_DIR, 'ipcc_carbon')
    try:
        os.makedirs(IPCC_CARBON_DIR)
    except OSError:
        pass

    for scenario_id, lulc_raster_path in LULC_SCENARIO_RASTER_PATH_MAP.items():
        if rasterize_carbon_zone_task is None:
            rasterized_zones_raster_path = os.path.join(
                clipped_data_dir, 'carbon_zones.tif')
            rasterize_carbon_zone_task = task_graph.add_task(
                func=rasterize_carbon_zones,
                args=(
                    lulc_raster_path, CARBON_ZONES_VECTOR_PATH,
                    rasterized_zones_raster_path),
                target_path_list=[rasterized_zones_raster_path],
                task_name='rasterize carbon zones')
            zone_lucode_to_carbon_map = parse_carbon_lulc_table(
                IPCC_CARBON_TABLE_PATH)

        ipcc_carbon_scenario_raster_map[scenario_id] = os.path.join(
            IPCC_CARBON_DIR,
            f'ipcc_carbon_{scenario_id}_{bounding_box_str}.tif')
        # Units are in Mg/Ha but pixel area is in degrees^2 so multiply result
        # by (111120 m/deg)**2*1 ha / 10000m^2
        # TODO: I can convert this to varying area later if we want
        conversion_factor = (
            pygeoprocessing.get_raster_info(
                lulc_raster_path)['pixel_size'][0]**2 *
            111120**2 *
            (1/100000)**2)

        task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(lulc_raster_path, 1), (rasterized_zones_raster_path, 1),
                 (zone_lucode_to_carbon_map, 'raw'),
                 (conversion_factor, 'raw')],
                ipcc_carbon_op, ipcc_carbon_scenario_raster_map[scenario_id],
                gdal.GDT_Float32, MULT_BY_COLUMNS_NODATA),
            dependent_task_list=[rasterize_carbon_zone_task],
            target_path_list=[ipcc_carbon_scenario_raster_map[scenario_id]],
            task_name=f'''create carbon for {
                ipcc_carbon_scenario_raster_map[scenario_id]}''')

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

    # 2) Evalute the forest regression for each scenario
    LOGGER.info("Forest Regression step 2")

    mult_by_columns_workspace = os.path.join(
        WORKSPACE_DIR, 'mult_by_columns_workspace', bounding_box_str)
    try:
        os.makedirs(mult_by_columns_workspace)
    except OSError:
        pass
    task_graph.join()

    FOREST_REGRESSION_RESULT_DIR = os.path.join(
        WORKSPACE_DIR, 'forest_regression_rasters')
    try:
        os.makedirs(FOREST_REGRESSION_RESULT_DIR)
    except OSError:
        pass

    forest_regression_scenario_raster_map = {}
    for scenario_id, lulc_scenario_raster_path in \
            LULC_SCENARIO_RASTER_PATH_MAP.items():
        conversion_factor = (
            pygeoprocessing.get_raster_info(
                lulc_scenario_raster_path)['pixel_size'][0]**2 *
            111120**2 *
            (1/100000)**2) * 0.47  # IPCC value to convert biomass to carbon
        forest_regression_scenario_raster_map[scenario_id] = os.path.join(
            FOREST_REGRESSION_RESULT_DIR,
            f'forest_regression_{scenario_id}_{bounding_box_str}.tif')

        mult_by_columns_library.mult_by_columns(
            FOREST_REGRESSION_LASSO_TABLE_PATH, clipped_data_dir,
            mult_by_columns_workspace,
            'lulc_esa_smoothed_2014_10sec', scenario_id,
            args.bounding_box, TARGET_PIXEL_SIZE,
            forest_regression_scenario_raster_map[scenario_id],
            task_graph, zero_nodata_symbols=ZERO_NODATA_SYMBOLS,
            target_nodata=MULT_BY_COLUMNS_NODATA,
            conversion_factor=conversion_factor)

    # NON-FOREST REGRESSION
    NON_FOREST_REGRESSION_RESULT_DIR = os.path.join(
        WORKSPACE_DIR, 'non_forest_regression_rasters')
    try:
        os.makedirs(NON_FOREST_REGRESSION_RESULT_DIR)
    except OSError:
        pass

    LOGGER.info('evalute non-forest regression')
    non_forest_regression_scenario_raster_map = {}
    for scenario_id, lulc_raster_path in LULC_SCENARIO_RASTER_PATH_MAP.items():
        conversion_factor = (
            pygeoprocessing.get_raster_info(
                lulc_raster_path)['pixel_size'][0]**2 *
            111120**2 *
            (1/100000)**2)
        for class_id in range(10, 221, 10):
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

        alpha = 'alpha0-0001'
        lasso_table_path = os.path.join(
            NON_FOREST_REGRESSION_LASSO_TABLES_DIR,
            f'lasso_not_forest_interacted_dummies_equation_string_{alpha}_'
            f'params.csv')

        non_forest_regression_scenario_raster_map[scenario_id] = os.path.join(
            NON_FOREST_REGRESSION_RESULT_DIR,
            f'non_forest_regression_{scenario_id}_{alpha}_'
            f'{bounding_box_str}.tif')

        mult_by_columns_library.mult_by_columns(
            lasso_table_path, clipped_data_dir,
            mult_by_columns_workspace,
            'lulc_esacci_2014_smoothed_class', scenario_id,
            args.bounding_box, TARGET_PIXEL_SIZE,
            non_forest_regression_scenario_raster_map[scenario_id],
            task_graph, zero_nodata_symbols=ZERO_NODATA_SYMBOLS,
            target_nodata=MULT_BY_COLUMNS_NODATA,
            conversion_factor=conversion_factor)

    task_graph.join()

    # combine both the non-forest and forest into one map for each
    # scenario based on their masks
    regression_carbon_scenario_path_map = {}
    REGRESSION_TOTAL_DIR = os.path.join(WORKSPACE_DIR, 'regression_total')
    try:
        os.makedirs(REGRESSION_TOTAL_DIR)
    except OSError:
        pass
    for scenario_id in LULC_SCENARIO_RASTER_PATH_MAP:
        regression_carbon_scenario_path_map[scenario_id] = os.path.join(
            REGRESSION_TOTAL_DIR,
            f'regression_carbon_{scenario_id}_{bounding_box_str}.tif')
        task_graph.add_task(
            func=raster_where,
            args=(
                mask_path_task_map[scenario_id]['forest_10sec'][0],
                forest_regression_scenario_raster_map[scenario_id],
                non_forest_regression_scenario_raster_map[scenario_id],
                regression_carbon_scenario_path_map[scenario_id]),
            target_path_list=[
                regression_carbon_scenario_path_map[scenario_id]],
            task_name=f'combine forest/nonforest for {scenario_id}')

    task_graph.join()

    # SCENARIOS/OPTIMIZATION

    # 1) Standard approach: the IPCC approach will be applied for ESA 2014 and
    #    to the forest pixels only of a Potential Natural Vegetation (PNV) map.
    #    An IPCC-based marginal value map will be created as the difference
    #    between the two, and pixels selected by the largest marginal value
    #    until the 3 Pg target is reached.

    # mask ipcc_carbon_scenario_raster_map to forest only from
    # restoration scenario
    masked_ipcc_carbon_raster_map = {}
    ipcc_mask_task_list = []
    for scenario_id in LULC_SCENARIO_RASTER_PATH_MAP:
        masked_ipcc_carbon_raster_map[scenario_id] = os.path.join(
            WORKSPACE_DIR,
            f'ipcc_carbon_forest_only_{scenario_id}_{bounding_box_str}.tif')

        # specifically masking to 'restoration limited'
        mask_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(ipcc_carbon_scenario_raster_map[scenario_id], 1),
                 (mask_path_task_map['restoration_limited']['forest_10sec'][0],
                  1), (MULT_BY_COLUMNS_NODATA, 'raw'), (MASK_NODATA, 'raw'),
                 (MULT_BY_COLUMNS_NODATA, 'raw')],
                mult_op, masked_ipcc_carbon_raster_map[scenario_id],
                gdal.GDT_Float32, MULT_BY_COLUMNS_NODATA),
            target_path_list=[masked_ipcc_carbon_raster_map[scenario_id]],
            task_name=f'mask out forest only ipcc {scenario_id}')
        ipcc_mask_task_list.append(mask_task)

    # subtract
    #   masked_ipcc_carbon_raster_map[esa2014]
    #   masked_ipcc_carbon_raster_map[restoration_limited]

    marginal_value_dir = os.path.join(WORKSPACE_DIR, 'marginal_values')
    try:
        os.makedirs(marginal_value_dir)
    except OSError:
        pass
    ipcc_carbon_marginal_value_raster = os.path.join(
        marginal_value_dir, f'marginal_value_ipcc_{bounding_box_str}.tif')
    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (masked_ipcc_carbon_raster_map['restoration_limited'], 1),
            (masked_ipcc_carbon_raster_map['esa2014'], 1),
            ],
            sub_pos_op, ipcc_carbon_marginal_value_raster, gdal.GDT_Float32,
            MULT_BY_COLUMNS_NODATA),
        dependent_task_list=ipcc_mask_task_list,
        target_path_list=[ipcc_carbon_marginal_value_raster],
        task_name='make ipcc marginal value raster')

    # TODO: mask out forest from IPCC to have a forest only map
    # TODO: set up raster calculation to subtract IPCC forest only from

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

    # mask the regression rasters
    masked_regression_carbon_raster_map = {}
    regression_mask_task_list = []
    for scenario_id in LULC_SCENARIO_RASTER_PATH_MAP:
        masked_regression_carbon_raster_map[scenario_id] = os.path.join(
            WORKSPACE_DIR,
            f'regression_carbon_forest_only_{scenario_id}_'
            f'{bounding_box_str}.tif')

        # specifically masking to 'restoration limited'
        mask_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(regression_carbon_scenario_path_map[scenario_id], 1),
                 (mask_path_task_map['restoration_limited']['forest_10sec'][0],
                  1), (MULT_BY_COLUMNS_NODATA, 'raw'), (MASK_NODATA, 'raw'),
                 (MULT_BY_COLUMNS_NODATA, 'raw')],
                mult_op, masked_regression_carbon_raster_map[scenario_id],
                gdal.GDT_Float32, MULT_BY_COLUMNS_NODATA),
            target_path_list=[masked_regression_carbon_raster_map[scenario_id]],
            task_name=f'mask out forest only regression {scenario_id}')
        regression_mask_task_list.append(mask_task)

    regression_carbon_marginal_value_raster = os.path.join(
        marginal_value_dir, f'marginal_value_regression_{bounding_box_str}.tif')
    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (masked_regression_carbon_raster_map['restoration_limited'], 1),
            (masked_regression_carbon_raster_map['esa2014'], 1),
            ],
            sub_pos_op, regression_carbon_marginal_value_raster,
            gdal.GDT_Float32, MULT_BY_COLUMNS_NODATA),
        dependent_task_list=regression_mask_task_list,
        target_path_list=[regression_carbon_marginal_value_raster],
        task_name='make regression marginal value raster')

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

"""Create carbon regression scenarios."""
import argparse
import logging
import multiprocessing
import os
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
    ('is_cropland', CROPLAND_LULC_CODES, ''),
    ('is_urban', URBAN_LULC_CODES, ''),
    ('not_forest', FOREST_CODES, 'inv')]

BASE_DATA_BUCKET_ROOT = 'gs://ecoshard-root/global_carbon_regression/inputs/'

LULC_URL_LIST = [
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/PNV_jsmith_060420_md5_8dd464e0e23fefaaabe52e44aa296330.tif',
    'https://storage.googleapis.com/nci-ecoshards/scenarios050420/restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif']

LASSO_TABLE_URI = 'gs://ecoshard-root/global_carbon_regression/lasso_interacted_not_forest_gs1to100_nonlinear_alpha0-0001_params_namefix.csv'
LASSO_TABLE_PATH = os.path.join(
    ECOSHARD_DIR, os.path.basename(LASSO_TABLE_URI))
# The following is the base in the pattern found in the lasso table
# [base]_[mask_type]_gs[kernel_size]
BASE_LASSO_CONVOLUTION_RASTER_NAME = 'lulc_esa_smoothed_2014_10sec'
LULC_SCENARIO_RASTER_PATH_LIST = []


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
    target_nodata = 2
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (base_nodata, 'raw'),
         (mask_value_list, 'raw'), (inverse, 'raw'),
         (target_nodata, 'raw')], _mask_vals_op,
        target_raster_path, gdal.GDT_Byte, target_nodata)


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

    # download lasso table
    download_lasso_task = task_graph.add_task(
        func=subprocess.run,
        args=(
            f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp -n '
            f'{LASSO_TABLE_URI} {LASSO_TABLE_PATH}',),
        kwargs={'shell': True, 'check': True},
        target_path_list=[LASSO_TABLE_PATH],
        task_name='download lasso table')

    global LULC_SCENARIO_RASTER_PATH_LIST
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
        LULC_SCENARIO_RASTER_PATH_LIST.append(lulc_raster_path)

    task_graph.close()
    task_graph.join()
    task_graph = None


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
        '--n_workers', default=multiprocessing.cpu_count(),
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
    mask_path_task_map = {}  # keep track of masks and tasks that made them
    for lulc_scenario_raster_path in LULC_SCENARIO_RASTER_PATH_LIST:
        lulc_basename = os.path.splitext(os.path.basename(
            lulc_scenario_raster_path))[0]
        for mask_type, lulc_codes, inverse_mode in MASK_TYPES:
            scenario_lulc_mask_raster_path = os.path.join(
                CHURN_DIR, f'{mask_type}_{lulc_basename}.tif')
            mask_task = task_graph.add_task(
                func=mask_ranges,
                args=(
                    lulc_scenario_raster_path, lulc_codes,
                    inverse_mode == 'inv', scenario_lulc_mask_raster_path),
                dependent_task_list=[
                    dl_lulc_task_map[lulc_scenario_raster_path]],
                target_path_list=[scenario_lulc_mask_raster_path],
                task_name=f'make {mask_type}_{lulc_basename}')
            mask_path_task_map[mask_type] = (
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
            for mask_type in mask_path_task_map:
                scenario_mask_path, mask_task = \
                    mask_path_task_map[mask_type]
                LOGGER.debug(
                    f'this is the scenario mask about to convolve: '
                    f'{scenario_mask_path} {mask_task}')
                convolution_mask_raster_path = os.path.join(
                    clipped_data_dir,
                    f'{lulc_basename}_{mask_type}_{pixel_radius}.tif')
                convolution_task = task_graph.add_task(
                    func=pygeoprocessing.convolve_2d,
                    args=(
                        (scenario_mask_path, 1), (kernel_raster_path, 1),
                        convolution_mask_raster_path),
                    dependent_task_list=[mask_task, kernel_task],
                    target_path_list=[convolution_mask_raster_path],
                    task_name=(
                        f'convolve {pixel_radius} {mask_type}_'
                        f'{lulc_basename}'))
                convolution_task_list.append(convolution_task)


    # 2) Apply the mult_rasters_by_columns.py script to existing inputs and
    #    the new convolution rasters for forest classes only.
    LOGGER.info("Forest Regression step 2")

    # NON-FOREST REGRESSION

    # Apply non-forest regression to non-forest classes using
    # mult_rasters_by_columns. Each of the non-forest LULC classes are
    # interacted with each of the predictor variables, so in order to run the
    # mult_rasters script on the ESA or PNV scenario maps, masks of each of
    # those LULC classes will need to be made as files called
    # lulc_esacci_2014_smoothed_class_10 - _220 (collapsing all the 11, 12 etc
    # back to its 10's place).


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
    #         LASSO_TABLE_PATH, clipped_data_dir, lasso_mult_workspace_dir,
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

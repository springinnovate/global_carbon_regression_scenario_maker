"""Create carbon regression scenarios."""
import argparse
import logging
import multiprocessing
import os
import subprocess
import sys

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

WORKSPACE_DIR = 'becky_ipcc_for_you'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')

CROPLAND_LULC_CODES = range(10, 41)
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

MASK_NODATA = 2
MULT_BY_COLUMNS_NODATA = -1

CARBON_ZONES_VECTOR_URI = 'gs://ecoshard-root/global_carbon_regression/carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d.gpkg'
CARBON_ZONES_VECTOR_PATH = os.path.join(ECOSHARD_DIR, 'carbon_zones.gpkg')
BASE_DATA_BUCKET_ROOT = 'gs://ecoshard-root/global_carbon_regression/inputs/'

LULC_SCENARIO_URI_MAP = {
    'esa2014': 'gs://ecoshard-root/global_carbon_regression/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif',
}

TARGET_PIXEL_SIZE = (10./3600., -10./3600.)
IPCC_CARBON_TABLE_URI = 'gs://ecoshard-root/global_carbon_regression/IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7.csv'
IPCC_CARBON_TABLE_PATH = os.path.join(
    ECOSHARD_DIR, os.path.basename(IPCC_CARBON_TABLE_URI))
BACCINI_10s_2014_BIOMASS_URI = (
    'gs://ecoshard-root/global_carbon_regression/baccini_10s_2014'
    '_md5_5956a9d06d4dffc89517cefb0f6bb008.tif')

# The following is the base in the pattern found in the lasso table
# [base]_[mask_type]_gs[kernel_size]
BASE_LASSO_CONVOLUTION_RASTER_NAME = 'lulc_esa_smoothed_2014_10sec'
LULC_SCENARIO_RASTER_PATH_MAP = {}


def ipcc_carbon_op(
        lulc_array, zones_array, zone_lulc_to_carbon_map, conversion_factor):
    """Map carbon to LULC/zone values and multiply by conversion map."""
    result = numpy.zeros(lulc_array.shape)
    for zone_id in numpy.unique(zones_array):
        if zone_id in zone_lulc_to_carbon_map:
            zone_mask = zones_array == zone_id
            result[zone_mask] = (
                zone_lulc_to_carbon_map[zone_id][lulc_array[zone_mask]] *
                conversion_factor)
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
                zone_lucode_to_carbon_map[zone_id][lucode] = float(
                    carbon_value)
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
        shell=True).decode('utf-8').splitlines() + [
            BACCINI_10s_2014_BIOMASS_URI]

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
    global BACCINI_10s_2014_BIOMASS_RASTER_PATH
    BACCINI_10s_2014_BIOMASS_RASTER_PATH = os.path.join(
        clipped_data_dir, os.path.basename(BACCINI_10s_2014_BIOMASS_URI))

    for data_uri, data_path in [
            (CARBON_ZONES_VECTOR_URI, CARBON_ZONES_VECTOR_PATH),
            (IPCC_CARBON_TABLE_URI, IPCC_CARBON_TABLE_PATH)]:
        _ = task_graph.add_task(
            func=subprocess.run,
            args=(
                f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp -n '
                f'{data_uri} {data_path}',),
            kwargs={'shell': True, 'check': True},
            target_path_list=[data_path],
            task_name=f'download {data_uri}')

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
        description='Becky\'s IPCC maker')
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
        # Units are in Mg/Ha
        conversion_factor = 1.0

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

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

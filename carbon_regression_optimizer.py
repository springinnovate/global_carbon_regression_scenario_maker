"""Used to optimize results from carbon regression."""
import argparse
import glob
import os

import pygeoprocessing
import numpy
import taskgraph

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
        '--target_dir', nargs=1, help="path to output dir")
    parser.add_argument(
        'base_rasters', nargs='+',
        help=("glob to base rasters to optimize"))
    parser.add_argument(
        '--sum', action='store_true', help='if set, report sum of raster')
    parser.add_argument(
        '--target_val', type=float, default=None,
        help='if set use this as the goal met cutoff')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph(args.target_dir)

    churn_dir = os.path.join(args.target_dir, 'churn')
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass

    for raster_path in [glob.glob(path) for path in args.base_rasters]:
        raster_sum_task = task_graph.add_task(
            func=calc_raster_sum,
            args=(raster_path,),
            task_name=f'calc sum for {raster_path}')
        raster_sum = raster_sum_task.get()
        if args.sum:
            print(f'{raster_path}: {raster_sum}')
        elif args.target_val is not None:
            raster_id = os.path.basename(os.path.splitext(raster_path)[0])
            output_dir = os.path.join(args.target_dir, raster_id)
            try:
                os.makedirs(output_dir)
            except OSError:
                pass
            target_threshold = args.target_val / raster_sum
            pygeoprocessing.raster_optimization(
                [(raster_path, 1)], churn_dir, output_dir,
                target_suffix=raster_id,
                goal_met_cutoffs=numpy.linspace(0, target_threshold, 5)[1:],
                heap_buffer_size=2**28, ffi_buffer_size=2**10)


if __name__ == '__main__':
    main()

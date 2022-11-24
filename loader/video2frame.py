
import os
import argparse
import logging
from typing import List, Union, Optional, Tuple

from sync import align_media_by_soundtrack, encode_videos
from sync.utils import GetWorkingDir



def process_videos(
        video_files: List[Union[os.PathLike, str]],
        output_dir: Union[os.PathLike, str],
        working_dir: Optional[Union[os.PathLike, str]] = None,
        fps: Optional[float] = None,
        resolution: Optional[Tuple[int, int]] = None,
):

    with GetWorkingDir(working_dir) as work_dir:
        align_result = align_media_by_soundtrack(
            video_files,
            work_dir,
            output_json=os.path.join(work_dir, 'align_output.json') if working_dir else None
        )
        outputs = encode_videos(
            video_files,
            output_dir,
            align_info=align_result,
            fps=fps,
            resolution=resolution,
        )

    return outputs


def main():
    parser = argparse.ArgumentParser()

    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    parser.add_argument(
        'input_dir',
        help='input videos directory',
    )
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='./extracted',
        help='output directory',
    )
    parser.add_argument(
        '--fps',
        help='target_fps',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--working_dir',
        help='working directory',
    )
    parser.add_argument(
        '--loglevel',
        default='debug',
        help='Logging level',
        choices=list(levels.keys())
    )
    args = parser.parse_args()

    logging.basicConfig(level=levels[args.loglevel])

    target_files = list(map(
        lambda fname: os.path.join(args.input_dir, fname),
        os.listdir(args.input_dir)
    ))
    # return
    process_videos(
        target_files,
        args.output_dir,
        args.working_dir,
        fps=args.fps,
        resolution=(960, 540)
    )


if __name__ == '__main__':
    main()




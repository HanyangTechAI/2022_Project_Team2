
import os
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
        align_result = align_media_by_soundtrack(video_files, work_dir)
        outputs = encode_videos()


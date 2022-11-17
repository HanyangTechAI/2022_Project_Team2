
import os
import logging
import subprocess

from typing import Optional, Tuple, Dict, Any, List, Union


_logger = logging.getLogger(__name__)

__all__ = [
    'encode_video_single',
    'encode_videos'
]


def _build_filename(
        basename,
        video_only=False,
        start_offset=None,
        duration=None,
        fps=None,
        resolution=None,
        **kwargs
):
    base_wo_ext, extension = os.path.splitext(basename)

    features = []
    if video_only:
        features.append(f'v')
    if start_offset:
        features.append(f'ss{start_offset:.2f}')
    if duration:
        features.append(f't{duration:.2f}')
    if fps:
        features.append(f'fps{fps:.2f}')
    if resolution:
        features.append(f'{resolution[0]}x{resolution[1]}')

    if features:
        ret = f'{base_wo_ext}_[{"_".join(features)}]{extension}'
    else:
        ret = f'{base_wo_ext}{extension}'
    return ret



def encode_video_single(
        video,
        output_path,
        overwrite: bool = False,
        video_only: bool = True,
        start_offset: Optional[float] = None,
        duration: Optional[float] = None,
        fps: Optional[float] = None,
        resolution: Optional[Tuple[int, int]] = None,
        silent: bool = True,
) -> Union[os.PathLike, str]:

    """
    encode single video using ffmpeg
    :return: output path
    """

    if not overwrite and os.path.exists(output_path):
        _logger.info(f'encode {video} - output {output_path} already exists')
        return output_path

    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-y',
        '-i', video,
    ]
    if video_only:
        cmd.extend(['-map', '0:v'])
    if start_offset and start_offset > 0:
        cmd.extend(['-ss', f'{start_offset:.3f}'])
    if duration:
        cmd.extend(['-t', f'{duration:.3f}'])

    filters = []
    if fps and fps > 0:
        filters.append(f'fps={fps:.3f}')
    if resolution:
        filters.append(f'scale={resolution[0]}:{resolution[1]}')
    if filters:
        cmd.extend(['-filter:v', f'\"{", ".join(filters)}\"'])

    cmd.append(output_path)

    cmd_str = ' '.join(cmd)
    _logger.info(f'encode {video} to {output_path}')
    _logger.debug(f'cmd {cmd_str}')
    if silent:
        subprocess.check_call(cmd_str, stderr=subprocess.DEVNULL)
    else:
        subprocess.check_call(cmd_str)
    return output_path


def encode_videos(
        video_files,
        output_dir,
        overwrite: bool = False,
        video_only: bool = True,
        align_info: Optional[List[Dict[str, Any]]] = None,
        fps: Optional[float] = None,
        resolution: Optional[Tuple[int, int]] = None,
        silent: bool = True,
) -> List[Union[os.PathLike, str]]:

    """
    encode videos using ffmpeg
    :return: list of output file paths
    """

    n_files = len(video_files)
    if n_files == 0:
        return []

    base_kwargs = dict(
        overwrite=overwrite,
        video_only=video_only,
        fps=fps,
        resolution=resolution,
        silent=silent,
    )

    file_kwargs = []
    durations = []
    for i in range(n_files):
        video_file = video_files[i]
        os.path.getatime(video_file)

        video_basename = os.path.basename(video_file)

        if align_info:
            video_info = align_info[i]

            assert video_basename == os.path.basename(video_info['file']), \
                "the files and align info seem to be out of order"

            durations.append(video_info['orig_duration'] - video_info['trim'])
            st_offset = video_info['trim']
        else:
            st_offset = None

        file_kwargs.append((
            video_file,
            dict(
                start_offset=st_offset,
            )
        ))

    if align_info:
        min_duration = min(durations)
        base_kwargs.update(duration=min_duration)

    outputs = []
    for i, (video_file, f_kwarg) in enumerate(file_kwargs):
        _logger.info(f'({i+1}/{n_files}) encode')

        video_output = os.path.join(
            output_dir,
            _build_filename(
                os.path.basename(video_file),
                **f_kwarg,
                **base_kwargs
            )
        )
        outputs.append(encode_video_single(video_file, video_output, **f_kwarg, **base_kwargs))

    return outputs


if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 2:
        target = sys.argv[1]
        if len(sys.argv) >= 3:
            out = sys.argv[2]
        else:
            out = 'out.mp4'

        encode_video_single(
            target,
            out,
            video_only=True,
            start_offset=3.,
            duration=190,
            # fps=2,
            resolution=(800, 450),
        )

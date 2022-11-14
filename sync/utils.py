
# This module is based on align-videos-by-sound, see
# https://github.com/align-videos-by-sound/align-videos-by-sound

import shutil
import time
import tempfile
import os
import re
import logging
import scipy.io.wavfile
import subprocess
from typing import List, Dict



__all__ = [
    'GetWorkingDir',
    'extract_audio',
    'validate_filenames',
    'load_wav_data',
    'get_media_info',
    'trim_video',
]
_logger = logging.getLogger(__name__)


class GetWorkingDir:
    def __init__(self, working_dir=None):
        self.working_dir = working_dir
        self.use_temp_dir = self.working_dir is None
        if self.use_temp_dir:
            self.working_dir = tempfile.mkdtemp()

        os.makedirs(self.working_dir, exist_ok=True)

        _logger.debug(f'working dir {self.working_dir}')

    def __enter__(self):
        return self.working_dir

    def __exit__(self, type, value, tb):
        if self.use_temp_dir:
            _logger.debug(f'clean tmp dir {self.working_dir}')
            retry = 3
            while retry:
                try:
                    shutil.rmtree(self.working_dir)
                    break
                except PermissionError:
                    _logger.debug(f'retrying clean tmp dir {self.working_dir}')
                time.sleep(0.5)
                retry -= 1

def _duration_to_hhmmss(duration):
    duration = round(duration*1000)
    hours, remainder = divmod(duration, 3600*1000)
    minutes, remainder = divmod(remainder, 60*1000)
    seconds, remainder = divmod(remainder, 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{remainder:03d}'


def extract_audio(
    media_file,
    output_dir,
    params,
):
    """
    extract audiotrack as wav from media (using ffmpeg)\n
    :param media_file:
    path of media to extract audiotrack
    :param output_dir:
    path where extracted audio files will be saved
    :param params:
    ffmpeg arguments
    :return:
    path of extracted audio file
    """
    os.path.getatime(media_file)

    cmd = [
        'ffmpeg',
        '-i', media_file,
        '-hide_banner',
        '-y',
        '-vn',
        '-ar', str(params.sample_rate),
        '-ac', '1',
        '-f', 'wav',
    ]

    if params.start_offset > 0:
        cmd.extend(['-ss', _duration_to_hhmmss(params.start_offset)])
    if params.duration > 0:
        cmd.extend(['-t', _duration_to_hhmmss(params.duration)])
    if params.afilter:
        cmd.extend(['-af', params.afilter])

    track_name = os.path.basename(media_file)
    audio_output = track_name + f'_[{params.start_offset:.2f}-{params.duration:.2f}-{params.sample_rate}].wav'

    output = os.path.join(output_dir, audio_output)
    cmd.append(output)

    if not os.path.exists(output):
        subprocess.check_call(cmd, stderr=open(os.devnull, 'w'))

    return output


def load_wav_data(wav_file):
    rate, data = scipy.io.wavfile.read(wav_file, mmap=True)
    return rate, data


def validate_filenames(files, min_num_files=0):
    result = list(map(os.path.abspath, files))
    nf_files = [path for path in result if not os.path.isfile(path)]

    if nf_files:
        for nf in nf_files:
            _logger.error("{}: No such file.".format(nf))
        raise FileNotFoundError

    if len(result) < min_num_files:
        _logger.error(
            "At least {} files are necessary.".format(
                min_num_files))
        raise FileNotFoundError

    return result

def _parse_time(s):
    if isinstance(s, (float, int)):
        return s
    else:
        rgx = r"(\d+):([0-5]\d):([0-5]\d)(\.\d+)?"
        m = re.match(rgx, s)
        if not m:
            raise ValueError("'{}' is not valid time.".format(s))
        hms = list(map(int, m.group(1, 2, 3)))
        ss = m.group(4)
        ss = ss[1:] if ss else "0"

        result = hms[0] * 60 * 60 + hms[1] * 60 + hms[2]
        result += int(ss) / (10**len(ss))
        return result


def _parse_ffprobe_output(output_str):
    def _split_csv(s):
        ss = s.split(", ")
        _res = []
        i = 0
        while i < len(ss):
            _res.append(ss[i])
            while i < len(ss) - 1 and \
                    _res[-1].count("(") != _res[-1].count(")"):
                i += 1
                _res[-1] = ", ".join((_res[-1], ss[i]))
            i += 1
        return _res

    result = {"streams": []}
    lines = output_str.split('\n')
    rgx = r"Duration: (\d+:\d{2}:\d{2}\.\d+)"
    for line in lines:
        m = re.search(rgx, line)
        if m:
            result["duration"] = _parse_time(m.group(1))
            break
    #
    rgx = r"Stream #(\d+):(\d+)\[0[xX][0-9a-fA-F]\](?:\(\w+\)): ([^:]+): (.*)$"
    streams_ = {}
    for line in lines:
        m = re.search(rgx, line)
        if not m:
            continue
        ifidx, strmidx, strmtype, rest = m.group(1, 2, 3, 4)
        if strmtype == "Video":
            spl = _split_csv(rest)
            resol = list(filter(lambda item: re.search(r"[1-9]\d*x[1-9]\d*", item), spl))[0]
            fps = list(filter(lambda item: re.search(r"[\d.]+ fps", item), spl))[0]
            streams_[int(strmidx)] = {
                "type": strmtype,
                "resolution": [
                    list(map(int, s.split("x"))) if i == 0 else s
                    for i, s in enumerate(resol.partition(" ")[0::2])
                ],
                "fps": float(fps.split(" ")[0]),
            }
        elif strmtype == "Audio":
            spl = _split_csv(rest)
            ar = list(filter(lambda item: re.search(r"\d+ Hz", item), spl))[0]
            streams_[int(strmidx)] = {
                "type": strmtype,
                "sample_rate": int(re.match(r"(\d+) Hz", ar).group(1)),
            }
    for i in sorted(streams_.keys()):
        result["streams"].append(streams_[i])
    return result


def _summarize_streams(streams: List[Dict]):
    result = dict(
        max_resol_width=0,
        max_resol_height=0,
        max_sample_rate=0,
        max_fps=0.0,
        num_video_streams=0,
        num_audio_streams=0
    )

    result["num_video_streams"] = sum([st["type"] == "Video" for st in streams])
    result["num_audio_streams"] = sum([st["type"] == "Audio" for st in streams])
    for st in streams:
        if st["type"] == "Video":
            new_w, new_h = st["resolution"][0]
            result["max_resol_width"] = max(
                result["max_resol_width"], new_w)
            result["max_resol_height"] = max(
                result["max_resol_height"], new_h)
            if "fps" in st:
                result["max_fps"] = max(
                    result["max_fps"], st["fps"])
        elif st["type"] == "Audio":
            result["max_sample_rate"] = max(
                result["max_sample_rate"], st["sample_rate"])

    return result


def get_media_info(filename):
    """
    return the information extracted by ffprobe.
    """
    os.path.getatime(filename)

    cmd = ["ffprobe", "-hide_banner", filename]
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = out.communicate()
    err_str = err.decode('utf-8')
    result = _parse_ffprobe_output(err_str)
    result["streams_summary"] = _summarize_streams(result["streams"])
    return result


def _trim_single_video(video, output_path, start_offset, duration):
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video,
        '-t', f'{duration:.3f}'
    ]
    if start_offset > 0:
        cmd.extend(['-ss', _duration_to_hhmmss(start_offset)])

    cmd.append(output_path)
    _logger.debug(f'trim video: {" ".join(cmd)}')
    subprocess.check_call(cmd)


def trim_video(video_files, align_info, output_dir, overwrite_ok=True):
    n_files = len(video_files)
    if n_files == 0:
        return

    targets = []
    durations = []
    for i in range(n_files):
        video_file = video_files[i]
        video_info = align_info[i]
        os.path.getatime(video_file)

        video_basename = os.path.basename(video_file)

        assert video_basename == os.path.basename(video_info['file']), \
            "the files and align info seem to be out of order"

        video_output = os.path.join(output_dir, video_basename)
        if not overwrite_ok and os.path.exists(video_output):
            raise FileExistsError(f'{video_output} seem to be exists already')

        targets.append((video_file, video_output, video_info['trim']))
        durations.append(video_info['orig_duration'] - video_info['trim'])

    os.makedirs(output_dir, exist_ok=True)
    min_duration = min(durations)

    for orig_video, out, trim in targets:
        _trim_single_video(orig_video, out, trim, min_duration)


if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 2:
        target = sys.argv[1]
        res = get_media_info(target)
        print(res)

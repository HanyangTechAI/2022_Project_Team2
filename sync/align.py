
# This script based on alignment_by_row_channels.py by Allison Deal, see
# https://github.com/allisonnicoledeal/VideoSync/blob/master/alignment_by_row_channels.py


import time
import logging
import numpy as np
from typing import Dict, Tuple

from sync.summarizer import summarize_media_files
from sync.utils import validate_filenames, get_media_info


__all__ = [
    'align',
    'build_result',
]
_logger = logging.getLogger(__name__)


def align(files, freq_dicts):
    """
    Find time delays between video files
    """
    st_time = time.time()

    _result1: Dict[Tuple, float] = dict()
    _result2: Dict[Tuple, float] = dict()

    _result2[(0, 0)] = 0.0
    for i in range(len(files) - 1):
        if (0, i + 1) in _result1:
            _result2[(0, i + 1)] = _result1[(0, i + 1)]
        elif (i + 1, 0) in _result1:
            _result2[(0, i + 1)] = -_result1[(i + 1, 0)]
        else:
            _result2[(0, i + 1)] = -summarizer.find_delay(freq_dicts[0], freq_dicts[i + 1])

    for ib, it in sorted(_result1.keys()):
        for i in range(len(files) - 1):
            if it == i + 1 and (0, i + 1) not in _result1 and (i + 1, 0) not in _result1:
                if files[0] != files[it]:
                    _result2[(0, it)] = _result2[(0, ib)] - _result1[(ib, it)]
            elif ib == i + 1 and (0, i + 1) not in _result1 and (i + 1, 0) not in _result1:
                if files[0] != files[ib]:
                    _result2[(0, ib)] = _result2[(0, it)] + _result1[(ib, it)]

    # build result
    result = np.array([_result2[k] for k in sorted(_result2.keys())])
    pad_pre = result - np.min(result)
    trim_pre = np.max(pad_pre) - pad_pre


    ed_time = time.time()
    _logger.info(f'total aligning time cost: {ed_time-st_time:.3f}s')

    return pad_pre, trim_pre




def build_result(files, align_result):
    """
    Find time delays between video files
    """
    files = validate_filenames(files)
    pad_pre, trim_pre = align_result

    info = [get_media_info(fn) for fn in files]
    orig_dur = np.array([info_["duration"] for info_ in info])
    streams_info = [(info_["streams"], info_["streams_summary"]) for info_ in info]
    pad_post = np.max(pad_pre + orig_dur) - (pad_pre + orig_dur)
    trim_post = (orig_dur - trim_pre) - np.min(orig_dur - trim_pre)

    result = [
        {
            "file": files[i],
            "trim": trim_pre[i],
            "pad": pad_pre[i],
            "orig_duration": orig_dur[i],
            "trim_post": trim_post[i],
            "pad_post": pad_post[i],
            "orig_streams": streams_info[i][0],
            "orig_streams_summary": streams_info[i][1],
        }
        for i in range(len(files))
    ]
    return result




import os
import ray
import time
import math
import logging
import heapq
import numpy as np
from collections import defaultdict
from typing import List, Tuple

from sync.align_params import SummarizerParams
from sync.utils import extract_audio, load_wav_data



__all__ = [
    'FreqTransSummarizer',
    'summarize_media_files',
]
_logger = logging.getLogger(__name__)


class FreqTransSummarizer:
    def __init__(self, working_dir, params: SummarizerParams):
        self.working_dir = working_dir
        self.params = params

    def _summarize(self, data):
        """
        Return characteristic frequency transition's summary.

        The dictionaries to be returned are as follows:
        * key: The frequency appearing as a peak in any time zone.
        * value: A list of the times at which specific frequencies occurred.
        """

        xrange = range(0, len(data), self.params.resolution)
        n_box_x = math.ceil(len(xrange) / self.params.box_width)
        n_box_y = math.ceil((self.params.fft_bin_size // 2) / self.params.box_height)

        boxes: List[List[List[Tuple]]] \
            = [[[] for _ in range(n_box_y)] for _ in range(n_box_x)]

        for x, j in enumerate(xrange):
            sample_data = data[max(0, j):max(0, j) + self.params.fft_bin_size]
            if len(sample_data) == self.params.fft_bin_size:
                intensities = np.abs(np.fft.fft(sample_data))

                box_x = x // self.params.box_width
                for y in range(self.params.fft_bin_size // 2):
                    # x: corresponding to time
                    # y: corresponding to freq

                    box_y = y // self.params.box_height

                    box_target = boxes[box_x][box_y]
                    val = (intensities[y], x, y)
                    if len(box_target) < self.params.maxes_per_box:
                        heapq.heappush(box_target, val)
                    elif val > box_target[0]:
                        heapq.heappop(box_target)
                        heapq.heappush(box_target, val)


        freq_dict = defaultdict(list)
        for box_row in boxes:
            for box in box_row:
                for _, x, y in box:
                    freq_dict[y].append(x)
        return freq_dict

    def _secs_to_x(self, secs):
        j = secs * self.params.sample_rate
        x = (j + self.params.overlap) / self.params.resolution
        return x

    def _x_to_secs(self, x):
        j = x * self.params.resolution - self.params.overlap
        return j / self.params.sample_rate

    def _summarize_wav(self, wav_file):
        rate, data = load_wav_data(wav_file)
        result = self._summarize(data)
        return rate, result

    def summarize_audiotrack(self, media):
        media_basename = os.path.basename(media)

        _logger.info(f"extracting audio for {media_basename} begin")
        wav_file = extract_audio(media, output_dir=self.working_dir, params=self.params)
        _logger.info(f"extracting audio for {media_basename} end")

        _logger.info(f"summarizing audio for {media_basename} begin")
        rate, ft_dict = self._summarize_wav(wav_file)
        _logger.info(f"summarizing audio for {media_basename} end")

        return ft_dict

    def find_delay(
            self,
            freq_origin, freq_sample,
            min_delay=float('nan'),
            max_delay=float('nan')
    ):
        #
        min_delay, max_delay = self._secs_to_x(min_delay), self._secs_to_x(max_delay)
        keys = set(freq_sample.keys()) & set(freq_origin.keys())
        #
        if not keys:
            raise Exception(
                """Cannot find a match. Consider giving a large value to \
"duration" if the target medias are sure to shoot the same event.""")

        if freq_origin == freq_sample:
            return 0.

        t_diffs = defaultdict(int)
        for key in keys:
            for x_i in freq_sample[key]:  # determine time offset
                for x_j in freq_origin[key]:
                    delta_t = x_i - x_j
                    min_ok = math.isnan(min_delay) or delta_t >= min_delay
                    max_ok = math.isnan(max_delay) or delta_t <= max_delay
                    if min_ok and max_ok:
                        t_diffs[delta_t] += 1

        try:
            delay_ = max(t_diffs.items(), key=lambda x: x[1])
            delay = self._x_to_secs(delay_[0])
        except IndexError:
            raise Exception(
                """Cannot find a match. \
Are the target medias sure to shoot the same event?""")
        return delay


@ray.remote
def _summarize_single_media_ray(
        media_file,
        summarizer: FreqTransSummarizer
):
    return summarizer.summarize_audiotrack(media_file)


def _summarize_media_ray(
        media_files,
        summarizer: FreqTransSummarizer
):
    ray.init()
    _logger.debug(f'summarizing {len(media_files)} files with ray distribution')
    st_time = time.time()

    summarizer_ = ray.put(summarizer)
    out_ids = [_summarize_single_media_ray.remote(media, summarizer_) for media in media_files]
    results = ray.get(out_ids)

    ed_time = time.time()
    _logger.debug(f'time cost {ed_time-st_time:.3f}s')
    ray.shutdown()
    return results


def _summarize_media(
        media_files,
        summarizer: FreqTransSummarizer
):
    _logger.debug(f'summarizing {len(media_files)} files sequentially')
    st_time = time.time()

    results = [summarizer.summarize_audiotrack(media) for media in media_files]

    ed_time = time.time()
    _logger.debug(f'time cost {ed_time-st_time:.3f}s')
    return results


def summarize_media_files(
        media_files,
        summarizer: FreqTransSummarizer,
        ray_threshold=4
):
    _logger.debug(f'start summarizing {len(media_files)} files')
    st_time = time.time()

    if len(media_files) < ray_threshold:
        result = _summarize_media(media_files, summarizer)
    else:
        result = _summarize_media_ray(media_files, summarizer)

    ed_time = time.time()
    _logger.debug(f'total summarizing time {ed_time-st_time:.3f}s')
    return result





# This module is based on align-videos-by-sound, see
# https://github.com/align-videos-by-sound/align-videos-by-sound

"""
This module contains only class for parameters of the detector class
for knowing the offset difference for audio and video files,
containing audio recordings from the same event.
"""


__all__ = [
    'SummarizerParams',
]


class SummarizerParams:
    """
    Parameter used by SyncDetector for summarizing audio track.
    It affects the behavior until find_delay return. Conversely,
    known_delay_map affecting interpretation of find_delay result is not
    included here.

    * max_misalignment:
        When handling media files with long playback time,
        it may take a huge amount of time and huge memory.
        In such a case, by changing this value to a small value,
        it is possible to indicate the scanning range of the media
        file to the program.

    * sample_rate:
        In this program, delay is examined by unifying all the sample
        rates of media files into the same one. If this value is the
        value itself of the media file itself, the result will be more
        precise. However, this wastes a lot of memory, so you can
        reduce memory consumption by downsampling (instead losing
        accuracy a bit). The default value uses quite a lot of memory,
        but if it changes to a value of, for example, 44100, 22050,
        etc., although a large error of about several tens of
        milliseconds  increases, the processing time is greatly
        shortened.

    * fft_bin_size, overlap:
        "fft_bin_size" is the number of audio samples passed to the FFT.
        If it is small, it means "fine" in the time domain viewpoint,
        whereas the larger it can be resolved into more kinds of
        frequencies. There is a possibility that it becomes difficult
        to be deceived as the frequency is examined finely, but instead
        the time step width of the delay detection becomes "coarse".
        "overlap" is in order to solve this dilemma. That is, windows
        for FFT are examined by overlapping each other. "overlap" must
        be less than "fft_bin_size".

    * box_height, box_width, maxes_per_box:
        This program sees the characteristics of the audio track by
        adopting a representative which has high strength in a small
        box divided into the time axis and the frequency axis.
        These parameters are those.

        Be careful as to how to give "box_height" is not easy to
        understand. It depends on the number of samples given to the
        FFT. That is, it depends on fft_bin_size - overlap. For
        frequencies not to separate, ie, not to create a small box,
        box_height should give (fft_bin_size - overlap) / 2.

    * afilter:
        This program begins by first extracting audio tracks from the
        media with ffmpeg. In this case, it is an audio filter given to
        ffmpeg. If the media is noisy, for example, it may be good to
        give a bandpass filter etc.
    """

    def __init__(self, **kwargs):
        self.sample_rate = kwargs.get("sample_rate", 44100)

        self.fft_bin_size = kwargs.get("fft_bin_size", 1024 * 3)
        self.overlap = kwargs.get("overlap", 1024 * 2)
        self.resolution = self.fft_bin_size - self.overlap

        self.box_height = kwargs.get("box_height", self.fft_bin_size // 2)
        self.box_width = kwargs.get("box_width", 40)
        self.maxes_per_box = kwargs.get("maxes_per_box", 8)

        self.start_offset = kwargs.get("start_offset", 20)
        self.duration = kwargs.get("duration", 100)

        self.afilter = kwargs.get("afilter")



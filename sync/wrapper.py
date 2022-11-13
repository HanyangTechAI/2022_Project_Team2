
import logging
from typing import Optional, List, Dict

from sync.align_params import SummarizerParams
from sync.align import align, build_result
from sync.summarizer import FreqTransSummarizer
from sync.utils import GetWorkingDir


__all__ = [
    'align_media_by_soundtrack',
]
_logger = logging.getLogger(__name__)


def align_media_by_soundtrack(
        media_files: List[str],
        working_dir: str = None,
        **kwargs,
) -> List[Dict]:
    """
    align sync of media files\n
    :param media_files:
    list of media files
    :param working_dir:
    directory to store temporary files (if not set, tempfile.mkdtemp is used)
    :param kwargs:
    summarizer param
    :return:
    list of dictionaries containing the information of each file
    """
    params = SummarizerParams(**kwargs)

    with GetWorkingDir(working_dir) as _dir:
        summarizer = FreqTransSummarizer(_dir, params)

        align_result = align(media_files, summarizer)
        result = build_result(media_files, align_result)

    return result


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG)

    media = [
        'attention_mc.mp4',
        'attention_mb.mp4',
    ]
    res = align_media_by_soundtrack(media, working_dir='temp')
    print('res')
    for rr in res:
        print(rr)



import logging
import json
from typing import Optional, List, Dict

from sync.align_params import SummarizerParams
from sync.align import align, build_result
from sync.summarizer import FreqTransSummarizer, summarize_media_files
from sync.utils import GetWorkingDir


__all__ = [
    'align_media_by_soundtrack',
]
_logger = logging.getLogger(__name__)


def align_media_by_soundtrack(
        media_files: List[str],
        working_dir: str = None,
        ray_threshold=2,
        output_json=None,
        **kwargs,
) -> List[Dict]:
    """
    align sync of media files\n
    :param media_files:
    list of media files
    :param working_dir:
    directory to store temporary files (if not set, tempfile.mkdtemp is used)
    :param ray_threshold
    minimum number of files for distributed processing
    :param output_json
    json output path (if set, save result in json format)
    :param kwargs:
    summarizer param
    :return:
    list of dictionaries containing the information of each file
    """
    params = SummarizerParams(**kwargs)

    with GetWorkingDir(working_dir) as _dir:
        summarizer = FreqTransSummarizer(_dir, params)
        freq_dicts = summarize_media_files(media_files, summarizer, ray_threshold=ray_threshold)

        align_result = align(media_files, freq_dicts, summarizer)
        result = build_result(media_files, align_result)

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=4)

    return result


if __name__ == '__main__':
    import logging
    import sys

    logging.basicConfig(level=logging.DEBUG)

    media = [
        'attention_mc.mp4',
        'attention_mb.mp4',
    ]
    if len(sys.argv) >= 2:
        targets = sys.argv[1:]
    else:
        targets = media
    res = align_media_by_soundtrack(
        targets,
        working_dir='temp',
        output_json='out.json',
        ray_threshold=4,
    )
    print('res')
    for rr in res:
        print(rr)


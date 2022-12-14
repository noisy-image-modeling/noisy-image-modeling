from dataclasses import dataclass, field
from pathlib import Path

from umei.args import AugArgs, UMeIArgs, SegArgs

@dataclass
class AmosArgs(AugArgs, SegArgs, UMeIArgs):
    monitor: str = field(default='val/dice/avg')
    monitor_mode: str = field(default='max')
    output_root: Path = field(default=Path('output/amos'))
    conf_root: Path = field(default=Path('conf/amos'))
    use_test_fold: bool = field(default=False)
    per_device_eval_batch_size: int = field(default=1)  # unable to batchify the whole image without resize
    # val_post: bool = field(default=False, metadata={'help': 'whether to perform post-processing during validation'})
    task_id: int = field(default=2, metadata={'choices': [1, 2]})
    use_monai: bool = field(default=False, metadata={'help': 'run validation for models produced by '
                                                             'official monai implementation'})

    @property
    def num_seg_classes(self) -> int:
        return 16

    @property
    def num_input_channels(self) -> int:
        return 1

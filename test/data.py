import torch.utils.data as data
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import DatasetFolder


class CustomImageDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = (
                    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(CustomImageDataset, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)
        classes, class_to_idx = DatasetFolder.find_classes(self, self.root)
        samples = DatasetFolder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample=sample)
        if self.target_transform is not None:
            target = self.target_transform(target=target)

        return sample, target

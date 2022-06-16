from torchgeo.datasets import BigEarthNet
import torch
import numpy as np

class BigEarthNetGoldi(BigEarthNet):
    # (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    # min/max band statistics computed on 100k random samples
    band_mins_raw = torch.tensor(
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    band_maxs_raw = torch.tensor(
        [
            31.0,
            35.0,
            18556.0,
            20528.0,
            18976.0,
            17874.0,
            16611.0,
            16512.0,
            16394.0,
            16672.0,
            16141.0,
            16097.0,
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    band_mins = torch.tensor(
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    band_maxs = torch.tensor(
        [
            6.0,
            16.0,
            9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            5897.0,
            5544.0,
        ]
    )
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "s2",
        num_classes: int = 19,
        transforms = None,
        download: bool = False,
        checksum: bool = False,
        **kwargs
    ) -> None:
        super(BigEarthNetGoldi, self).__init__(root, 
                                               split=split, 
                                               bands=bands, 
                                               num_classes=num_classes,
                                               transforms=transforms,
                                               download= False,
                                               checksum=False,
                                               **kwargs)
        if split=="train":
            self.split = "val"
            self.folders += self._load_folders()
            self.split = "train"
        self.sequence = np.arange(len(self.folders))
    
    def __len__(self):
        return len(self.sequence)
    
    def _verify(self):
        pass
    
    def preprocess1(self, image):
        self.mins = self.band_mins[2:, None, None]
        self.maxs = self.band_maxs[2:, None, None]
        image = image.float()
        image = (image - self.mins) / (self.maxs - self.mins)
        image = torch.clip(image, min=0.0, max=1.0)
        return image
    
    def preprocess(self, image):
        self.mins = self.band_mins[[4,5,6,10], None, None]
        self.maxs = self.band_maxs[[4,5,6,10], None, None]
        image = image[[4,5,6,10],:,:].float()
        image = (image - self.mins) / (self.maxs - self.mins)
        image = torch.clip(image, min=0.0, max=1.0)
        return image
    
    def __getitem__(self, index: int):
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        idx = self.sequence[index]
        image = self._load_image(idx)
        label = self._load_target(idx)
        
        image = self.preprocess(image)
        if self.transforms is not None:
            image = self.transforms(image)

        return idx, image, label
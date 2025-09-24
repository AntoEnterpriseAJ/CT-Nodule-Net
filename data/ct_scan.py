import numpy as np
import SimpleITK as sitk  # noqa: N813

from config import MHD_INDEX
from utils.utils import xyz_to_irc


class CTScan:
    def __init__(self, series_uid: str) -> None:
        mhd_path = MHD_INDEX[f"{series_uid}"]
        ct_mhd = sitk.ReadImage(mhd_path)

        ct_array = np.array(sitk.GetArrayFromImage(image=ct_mhd), dtype=np.float32)
        ct_array.clip(min=-1000, max=1000, out=ct_array)  # clip irrelevant information

        self.series_uid = series_uid
        self.ct_array = ct_array

        self.origin_xyz = np.array(ct_mhd.GetOrigin())
        self.vxSize_xyz = np.array(ct_mhd.GetSpacing())
        self.direction = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(
        self, candidate_xyz: tuple[float, float, float], width_irc: tuple[int, int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_irc = xyz_to_irc(candidate_xyz, self.origin_xyz, self.vxSize_xyz, self.direction)

        slices: list[slice] = []
        for axis, length in enumerate(width_irc):
            start_pos = round(candidate_irc[axis] - length / 2)
            stop_pos = start_pos + length

            assert length >= 0 and length < self.ct_array.shape[axis]

            if start_pos < 0:
                start_pos = 0
                stop_pos = length

            if stop_pos > self.ct_array.shape[axis]:
                stop_pos = self.ct_array.shape[axis]
                start_pos = stop_pos - length

            assert stop_pos - start_pos == length, "length mismatch"

            slices.append(slice(start_pos, stop_pos))

        return self.ct_array[*slices], candidate_irc

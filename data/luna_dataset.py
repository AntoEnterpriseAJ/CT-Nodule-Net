from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from config import ANNOTATIONS_PATH, CANDIDATES_PATH
from data.cache import get_ct_scan
from utils.utils import coord_distance


class CandidateInfoTuple(NamedTuple):
    is_nodule: bool
    diameter_mm: float
    series_uid: str
    center_xyz: tuple[float, float, float]


class AnnotationTuple(NamedTuple):
    series_uid: str
    coord: tuple[float, float, float]
    diameter: float


class LunaDataset(Dataset):
    def __init__(
        self,
        validate_stride: int = 0,
        validate: bool = False,
        series_uid: str | None = None,
    ) -> None:
        self.candidate_info_list = self._get_candidate_info(
            annotations_path=ANNOTATIONS_PATH, candidates_path=CANDIDATES_PATH
        )

        if series_uid:
            self.candidate_info_list = [
                candidate
                for candidate in self.candidate_info_list
                if candidate.series_uid == series_uid
            ]

        if validate:
            assert validate_stride > 0 and validate_stride < len(self.candidate_info_list), (
                f"invalid validation stride: {validate_stride}"
            )
            self.candidate_info_list = self.candidate_info_list[::validate_stride]
        elif validate_stride > 0:
            del self.candidate_info_list[::validate_stride]

        assert self.candidate_info_list

    def __len__(self) -> int:
        return len(self.candidate_info_list)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, tuple[float, float, float]]:
        assert index >= 0 and index < len(self.candidate_info_list), "index out of range"

        series_uid = self.candidate_info_list[index].series_uid
        xyz = self.candidate_info_list[index].center_xyz
        is_nodule = self.candidate_info_list[index].is_nodule

        ct_scan = get_ct_scan(series_uid)
        candidate, candidate_irc = ct_scan.get_raw_candidate(
            candidate_xyz=xyz, width_irc=(32, 48, 48)
        )

        candidate_t = torch.tensor(candidate, dtype=torch.float32).unsqueeze(0)
        is_nodule_t = torch.tensor(is_nodule, dtype=torch.long)

        return (candidate_t, is_nodule_t, series_uid, candidate_irc)

    @staticmethod
    def _get_candidate_info(
        annotations_path: str, candidates_path: str
    ) -> list[CandidateInfoTuple]:
        annotations_df = pd.read_csv(filepath_or_buffer=annotations_path)
        candidates_df = pd.read_csv(filepath_or_buffer=candidates_path)

        annotations_dict: dict[list[AnnotationTuple]] = {}
        for row in annotations_df.itertuples():
            annotations_dict.setdefault(row.seriesuid, []).append(
                AnnotationTuple(
                    series_uid=row.seriesuid,
                    coord=(row.coordX, row.coordY, row.coordZ),
                    diameter=row.diameter_mm,
                )
            )

        candidate_info_list: list[CandidateInfoTuple] = []
        for row in candidates_df.itertuples():
            series_uid = row.seriesuid
            coord_x, coord_y, coord_z = row.coordX, row.coordY, row.coordZ
            is_nodule = row[5]  # .class is a reserved keyword in Python
            diameter_mm = 0

            if series_uid in annotations_dict:
                current_coord = (coord_x, coord_y, coord_z)
                for annotation in annotations_dict[series_uid]:
                    if coord_distance(annotation.coord, current_coord) < annotation.diameter / 4:
                        diameter_mm = annotation.diameter
                        break

            candidate_info_list.append(
                CandidateInfoTuple(
                    is_nodule=bool(is_nodule),
                    diameter_mm=diameter_mm,
                    series_uid=series_uid,
                    center_xyz=(coord_x, coord_y, coord_z),
                )
            )

        return candidate_info_list

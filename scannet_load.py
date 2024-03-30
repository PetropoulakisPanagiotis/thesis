import numpy as np
import json
import cv2
from dataclasses import dataclass
from typing import Tuple, Union
from scipy.spatial.transform import Rotation

SCANNET_COLOR_MAP_20 = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

SCANNET_COLOR_MAP_20[13] = (178, 76, 76)
SCANNET_COLOR_MAP_20[31] = (120, 185, 128)

@dataclass(frozen=True)
class _LabelBase:
    class_name: str

@dataclass(frozen=True)
class SemanticLabel(_LabelBase):
    is_thing: Union[bool, None]
    use_orientations: Union[bool, None]
    color: Tuple[int]

class _LabelListBase:
    def __init__(
        self,
        label_list: Tuple[_LabelBase] = ()
    ) -> None:
        self.label_list = list(label_list)
        # a copy of a the class names list for faster name to idx lookup
        self._class_names = ()
        self._update_internal_lists()
        # for iterator
        self._idx = 0

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        return self.label_list[idx]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            el = self[self._idx]
            self._idx += 1
            return el
        except IndexError:
            self._idx = 0
            raise StopIteration     # done iterating

    def add_label(self, label: _LabelBase):
        self.label_list.append(label)
        self._update_internal_lists()

    def _update_internal_lists(self):
        self._class_names = tuple(item.class_name for item in self.label_list)

    def _name_to_idx(self, name: str) -> int:
        return self._class_names.index(name)

    def index(self, value: Union[_LabelBase, str]) -> int:
        if isinstance(value, _LabelBase):
            return self.label_list.index(value)
        else:
            return self._name_to_idx(value)

    def __contains__(self, value: Union[_LabelBase, str]) -> bool:
        if isinstance(value, _LabelBase):
            return value in self.label_list
        else:
            return value in self._class_names

    @property
    def class_names(self) -> Tuple[str]:
        return self._class_names

class SemanticLabelList(_LabelListBase):
    @property
    def colors(self) -> Tuple[Tuple[int]]:
        return tuple(item.color for item in self.label_list)

    @property
    def colors_array(self) -> np.ndarray:
        return np.array(self.colors, dtype=np.uint8)

    @property
    def classes_is_thing(self) -> Tuple[bool]:
        return tuple(item.is_thing for item in self.label_list)

    @property
    def classes_use_orientations(self) -> Tuple[bool]:
        return [item.use_orientations for item in self.label_list]

SEMANTIC_CLASS_COLORS_SCANNET_40 = tuple(
        tuple(int(val) for val in SCANNET_COLOR_MAP_20[idx])
        for idx in sorted(SCANNET_COLOR_MAP_20.keys())
    )

# set missing colors in dictionary, for source of colors see:

# https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util.py


SEMANTIC_LABEL_LIST_40 = SemanticLabelList([
        # class_name, is_thing, use orientations, color
        SemanticLabel('void',           False, None, SEMANTIC_CLASS_COLORS_SCANNET_40[0]),  # unannotated
        SemanticLabel('wall',           False, None, SEMANTIC_CLASS_COLORS_SCANNET_40[1]),
        SemanticLabel('floor',          False, None, SEMANTIC_CLASS_COLORS_SCANNET_40[2]),
        SemanticLabel('cabinet',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[3]),
        SemanticLabel('bed',            True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[4]),
        SemanticLabel('chair',          True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[5]),
        SemanticLabel('sofa',           True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[6]),
        SemanticLabel('table',          True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[7]),
        SemanticLabel('door',           True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[8]),
        SemanticLabel('window',         True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[9]),
        SemanticLabel('bookshelf',      True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[10]),
        SemanticLabel('picture',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[11]),
        SemanticLabel('counter',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[12]),
        SemanticLabel('blinds',         True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[13]),
        SemanticLabel('desk',           True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[14]),
        SemanticLabel('shelves',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[15]),
        SemanticLabel('curtain',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[16]),
        SemanticLabel('dresser',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[17]),
        SemanticLabel('pillow',         True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[18]),
        SemanticLabel('mirror',         True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[19]),
        SemanticLabel('floor mat',      True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[20]),
        SemanticLabel('clothes',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[21]),
        SemanticLabel('ceiling',        False, None, SEMANTIC_CLASS_COLORS_SCANNET_40[22]),
        SemanticLabel('books',          True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[23]),
        SemanticLabel('refrigerator',   True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[24]),  # renamed from refridgerator
        SemanticLabel('television',     True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[25]),
        SemanticLabel('paper',          True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[26]),
        SemanticLabel('towel',          True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[27]),
        SemanticLabel('shower curtain', True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[28]),  # sometimes also referred to as: 'shower'
        SemanticLabel('box',            True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[29]),
        SemanticLabel('whiteboard',     True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[30]),
        SemanticLabel('person',         True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[31]),
        SemanticLabel('night stand',    True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[32]),
        SemanticLabel('toilet',         True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[33]),
        SemanticLabel('sink',           True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[34]),
        SemanticLabel('lamp',           True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[35]),
        SemanticLabel('bathtub',        True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[36]),
        SemanticLabel('bag',            True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[37]),
        SemanticLabel('otherstructure', True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[38]),
        SemanticLabel('otherfurniture', True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[39]),
        SemanticLabel('otherprop',      True,  None, SEMANTIC_CLASS_COLORS_SCANNET_40[40])
    ])


VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
class ScanNet():
    def __init__(self, dataset_path: str, split: str = 'train') -> None:

        self.dataset_path = dataset_path
        self.split = split
        self.semantic_n_classes = 20

        file_names_path = self.dataset_path + "/" + split + ".txt"
        with open(file_names_path, 'r') as f:
            self.filenames = f.read().splitlines()


        self.intr = np.eye(3)
        self.intr[0][0] = 577.8706114969136
        self.intr[0][2] = 319.87654320987656
        self.intr[1][1] = 577.8706114969136
        self.intr[1][2] = 238.88888888888889


        SEMANTIC_LABEL_LIST_20 = SemanticLabelList([
            SemanticLabel('void', False, None, SEMANTIC_CLASS_COLORS_SCANNET_40[0])
        ])
        for idx in VALID_CLASS_IDS_20:
            label = SEMANTIC_LABEL_LIST_40[idx]
            SEMANTIC_LABEL_LIST_20.add_label(label)

        print(SEMANTIC_LABEL_LIST_20[5].color)
        exit()
    def get_item(self, idx):
        assert idx < len(self.filenames) and idx >= 0

        # rgb #
        rgb_path = self.dataset_path + "/" + self.split + "/rgb/" + self.filenames[idx] + ".jpg"
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # depth #
        depth_path = self.dataset_path + "/" + self.split + "/depth/" + self.filenames[idx] + ".png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # extr #
        extr_path = self.dataset_path + "/" + self.split + "/extrinsics/" + self.filenames[idx] + ".json"

        with open(extr_path, 'r') as f:
            extr = json.load(f)

        w, x, y, z = extr['quat_w'], extr['quat_x'], extr['quat_y'], extr['quat_z']
        # Assuming you have translation values (tx, ty, tz)
        tx = extr['x']
        ty = extr['y']
        tz = extr['z']
        print(extr)
        # Create rotation matrix from quaternions
        rotation_matrix = Rotation.from_quat([x, y, z, w]).as_matrix()

        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [tx, ty, tz] 
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [tx, ty, tz]
        extr = transformation_matrix

        sem, inst = None, None
        if self.split != 'test':
            # sem #
            sem_path = self.dataset_path + "/" + self.split + "/semantic_refined_" + str(self.semantic_n_classes) + "/" + self.filenames[idx] + ".png"
            sem = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)

            # inst #
            inst_path = self.dataset_path + "/" + self.split + "/instance_refined/" + self.filenames[idx] + ".png"
            inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED).astype('int32')

        return rgb, depth, extr, sem, inst

if __name__ == '__main__':

    dataset = ScanNet('/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted')
    print(dataset.get_item(0))

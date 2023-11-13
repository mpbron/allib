# coding=utf-8

"""
The implementation is based on
[1] Satopaa, Ville, et al. "Finding a" kneedle" in a haystack: Detecting knee points in system behavior." 2011 31st international conference on distributed computing systems workshops. IEEE, 2011.
[2] Gordon V. Cormack and Maura R. Grossman. 2016. Engineering Quality and Reliability in Technology-Assisted Review. In Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (Pisa, Italy) (SIGIR ’16). ACM, New York, NY, USA, 75–84.
[3] 
"""

from enum import Enum, auto
from typing import Any, Dict, Sequence
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from ..activelearning.base import ActiveLearner
from instancelib.utils.func import list_unzip
from .others import KneeStoppingRule
import numpy.typing as npt
import matplotlib.pyplot as plt


def detect_knee(data: npt.NDArray[Any], window_size=1, s=10) -> Sequence[int]:
    """
    Detect the so-called knee in the data.

    The implementation is based on paper [1] and code here (https://github.com/jagandecapri/kneedle).

    @param data: The 2d data to find an knee in.
    @param window_size: The data is smoothed using Gaussian kernel average smoother, this parameter is the window used for averaging (higher values mean more smoothing, try 3 to begin with).
    @param s: How many "flat" points to require before we consider it a knee.
    @return: The knee values.
    """
    assert len(data.shape) == 2
    data_size = data.shape[0]
    if data_size == 1:
        return list()

    # smooth
    smoothed_data = []
    for i in range(data_size):
        if 0 < i - window_size:
            start_index = i - window_size
        else:
            start_index = 0
        if i + window_size > data_size - 1:
            end_index = data_size - 1
        else:
            end_index = i + window_size

        sum_x_weight = 0
        sum_y_weight = 0
        sum_index_weight = 0
        for j in range(start_index, end_index):
            index_weight = norm.pdf(abs(j - i) / window_size, 0, 1)
            sum_index_weight += index_weight
            sum_x_weight += index_weight * data[j][0]
            sum_y_weight += index_weight * data[j][1]

        smoothed_x = sum_x_weight / sum_index_weight
        smoothed_y = sum_y_weight / sum_index_weight

        smoothed_data.append((smoothed_x, smoothed_y))

    smoothed_data = np.array(smoothed_data)

    # normalize
    normalized_data = MinMaxScaler().fit_transform(smoothed_data)

    # difference
    differed_data = [(x, y - x) for x, y in normalized_data]

    # find indices for local maximums
    candidate_indices = []
    for i in range(1, data_size - 1):
        if (differed_data[i - 1][1] < differed_data[i][1]) and (
            differed_data[i][1] > differed_data[i + 1][1]
        ):
            candidate_indices.append(i)

    # threshold
    step = s * (normalized_data[-1][0] - data[0][0]) / (data_size - 1)

    # knees
    knee_indices = list()
    for i in range(len(candidate_indices)):
        candidate_index = candidate_indices[i]

        if i + 1 < len(candidate_indices):  # not last second
            end_index = candidate_indices[i + 1]
        else:
            end_index = data_size

        threshold = differed_data[candidate_index][1] - step

        for j in range(candidate_index, end_index):
            if differed_data[j][1] < threshold:
                knee_indices.append(candidate_index)
                break

    return knee_indices  # data[knee_indices]


def test_detect_knee():
    # data with knee at [0.2, 0.75]
    print("First example.")
    data = np.array(
        [
            [0, 0],
            [0.1, 0.55],
            [0.2, 0.75],
            [0.35, 0.825],
            [0.45, 0.875],
            [0.55, 0.9],
            [0.675, 0.925],
            [0.775, 0.95],
            [0.875, 0.975],
            [1, 1],
        ]
    )

    knees = detect_knee(data, window_size=1, s=1)
    for knee in knees:
        print(data[knee])

    # data with knee at [0.45  0.1  ], [0.775 0.2  ]
    print("Second example.")
    data = np.array(
        [
            [0, 0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.35, 0.1],
            [0.45, 0.1],
            [0.55, 0.1],
            [0.675, 0.2],
            [0.775, 0.2],
            [0.875, 0.2],
            [1, 1],
        ]
    )
    knees = detect_knee(data, window_size=1, s=1)
    for knee in knees:
        print(data[knee])


class RhoMode(str, Enum):
    DYNAMIC = auto()
    STATIC = auto()


def find_perpendicular_point(p1, p2, p3):
    # Calculate the direction vector of the line
    dir_vec = p3 - p2

    # Calculate the vector from p1 to the line
    vec = p2 - p1

    # Project vec onto dir_vec to find the distance from p1 to the line
    dist = np.dot(vec, dir_vec) / np.dot(dir_vec, dir_vec)

    # Calculate the point on the line that is perpendicular to p1
    p4 = p2 + dist * dir_vec

    return p4


def find_perpendicular_point2(p1, p2, p3):
    # Calculate the slope of the line l1
    # Calculate the slope of line l1
    m1 = (p3[1] - p2[1]) / (p3[0] - p2[0])

    # The intercept of line l1
    c1 = p2[1] - m1 * p2[1]

    # The x and y coordinates of the intersection point
    x4 = (p1[0] + m1 * p2[1] - m1 * c1) / (1 + m1**2)
    y4 = m1 * x4 + c1

    return np.array([x4, y4])


class KneeAutoSTOP(KneeStoppingRule):
    mode: RhoMode
    _rho_target: float
    stopping_beta: int
    rhos: Dict[int, float]

    def __init__(
        self,
        pos_label: Any,
        mode=RhoMode.DYNAMIC,
        stopping_beta: int = 100,
        rho_target: float = 6,
    ) -> None:
        super().__init__(pos_label)
        self.mode = mode
        self._rho_target = rho_target
        self.stopping_beta = stopping_beta
        self.rhos = dict()
        self.rhots = dict()

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, Any]) -> None:
        super().update(learner)
        current_it = self.stats.rounds
        if current_it not in self.rhos:
            self.rhos[current_it] = self.rho
            self.rhots[current_it] = self.rho_target

    @property
    def rho(self) -> float:
        pos_per_round = np.array(self.stats.label_per_round(self.pos_label))
        Lp = pos_per_round.cumsum()
        annotated_per_round = np.array(self.stats.annotations_per_round)
        L = annotated_per_round.cumsum()
        knee_data = np.column_stack((L, Lp))
        knee_indices = detect_knee(knee_data)
        if not knee_indices:
            return 0.0
        last_knee = knee_indices[-1]
        r1 = Lp[last_knee]
        rank1 = L[last_knee]
        r2 = Lp[-1]
        rank2 = L[-1]
        try:
            current_rho = float(r1 / rank1) / float((r2 - r1 + 1) / (rank2 - rank1))
        except:
            print(
                "(rank1, r1) = ({} {}), (rank2, r2) = ({} {})".format(
                    rank1, r1, rank2, r2
                )
            )
            current_rho = 0

        return current_rho

    @property
    def rho_target(self) -> float:
        pos_per_round = np.array(self.stats.label_per_round(self.pos_label))
        Lp = pos_per_round.cumsum()
        if self.mode == RhoMode.DYNAMIC:
            return 156 - min(Lp[-1], 150)
        return self._rho_target

    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 1:
            return False
        if self.stats.current_annotated < self.stopping_beta:
            return False
        return self.rho >= self.rho_target

    def plot_knee(self) -> None:
        rhos = [self.rhos[i] for i in range(self.stats.rounds + 1) if i in self.rhos]
        rhots = [self.rhots[i] for i in range(self.stats.rounds + 1) if i in self.rhots]
        annotated_per_round = np.array(self.stats.annotations_per_round)
        L = annotated_per_round.cumsum()[0 : len(rhos)]
        pos_per_round = np.array(self.stats.label_per_round(self.pos_label))
        Lp = pos_per_round.cumsum()[0 : len(rhos)]
        knee_data = np.column_stack((L, Lp))
        knee_indices = detect_knee(knee_data)

        plt.plot(L, rhos, linestyle="-.", label="Rhos")
        plt.plot(L, rhots, linestyle="-.", label="Rho Target")
        plt.plot(L, Lp, linestyle="-.", label="L+")
        plt.scatter(
            [L[idx] for idx in knee_indices],
            [Lp[idx] for idx in knee_indices],
            s=20,
            marker="^",  # type: ignore
        )
        plt.plot(
            L,
            (L / L[-1]) * Lp[-1],
            ":",
        )
        origin = np.array([0, 0])
        end = np.array([L[-1], Lp[-1]])
        knees = [np.array([L[idx], Lp[idx]]) for idx in knee_indices]
        ppds = np.abs(
            np.vstack([find_perpendicular_point2(knee, origin, end) for knee in knees])
        )
        plt.scatter(ppds[:, 0], ppds[:, 1], s=20, marker="o")  # type: ignore
        plt.plot()
        plt.legend()
        plt.show()

    def plot(self) -> None:
        rhos = [self.rhos[i] for i in range(self.stats.rounds + 1) if i in self.rhos]
        rhots = [self.rhots[i] for i in range(self.stats.rounds + 1) if i in self.rhots]
        annotated_per_round = np.array(self.stats.annotations_per_round)
        L = annotated_per_round.cumsum()[0 : len(rhos)]
        pos_per_round = np.array(self.stats.label_per_round(self.pos_label))
        Lp = pos_per_round.cumsum()[0 : len(rhos)]
        plt.plot(L, rhos, linestyle="-.", label="Rhos")
        plt.plot(L, rhots, linestyle="-.", label="Rho Target")
        plt.plot(L, Lp, linestyle="-.", label="L+")
        plt.legend()
        plt.show()


class BudgetAutoSTOP(KneeAutoSTOP):
    target_size: int

    def __init__(
        self,
        pos_label: Any,
        stopping_beta: int = 100,
        rho_target: float = 6,
        target_size: int = 10,
    ) -> None:
        super().__init__(pos_label, RhoMode.STATIC, stopping_beta, rho_target)
        self.target_size = target_size

    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 1 or self.stats.current_label_count(self.pos_label) < 1:
            return False

        pos_per_round = np.array(self.stats.label_per_round(self.pos_label))
        Lp = pos_per_round.cumsum()
        n_docs = self.stats.dataset_size
        condition_one = self.stats.current_annotated >= n_docs * 0.75
        condition_two_a = self.rho >= self.rho_target
        condition_two_b = self.stats.current_annotated >= self.target_size * (
            n_docs / Lp[-1]
        )
        return condition_one or (condition_two_a and condition_two_b)

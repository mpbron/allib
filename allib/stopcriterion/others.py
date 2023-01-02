from typing import Any, Generic, Mapping, Optional
from ..analysis.base import AbstractStatistics, StatsMixin, AnnotationStatistics
from ..activelearning.base import ActiveLearner
from ..typehints.typevars import KT, LT
from .base import AbstractStopCriterion
from scipy.stats import hypergeom
import numpy as np
import numpy.typing as npt
import instancelib as il


class StatsStoppingCriterion(AbstractStopCriterion[LT], Generic[KT, LT]):
    stats: AbstractStatistics[KT, LT]

    def __init__(self, pos_label: LT) -> None:
        super().__init__()
        self.pos_label = pos_label
        self.stats = AnnotationStatistics[KT, LT]()

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        if isinstance(learner, StatsMixin):
            self.stats = learner.stats
        else:
            self.stats.update(learner)


class StopAfterKNegative(StatsStoppingCriterion[KT, LT]):
    def __init__(self, pos_label: LT, k: int) -> None:
        super().__init__(pos_label)
        self.k = k

    @property
    def stop_criterion(self) -> bool:
        annotated_since_last_pos = self.stats.annotations_since_last(self.pos_label)
        return annotated_since_last_pos >= self.k


class KneeStoppingRule(StatsStoppingCriterion[KT, LT]):
    """
    .. seealso::
        .. [1] Gordon V. Cormack, and Maura R. Grossman. "Engineering quality and reliability in technology-assisted review."
               *Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval.* 2016.
               `<https://dl.acm.org/doi/10.1145/2911451.2911510>`__
    """

    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 1:
            return False

        pos_per_round = np.array(self.stats.label_per_round(self.pos_label))
        pos_found = pos_per_round.cumsum()

        rho_s = -1
        for i in range(self.stats.rounds):
            rho = (pos_found[i] / (i + 1)) / (
                (1 + pos_found[-1] - pos_found[i]) / (self.stats.rounds - i)
            )
            rho_s = max(rho_s, rho)

        return rho_s >= 156 - min(pos_found[-1], 150)


class BudgetStoppingRule(StatsStoppingCriterion[KT, LT]):
    """
    .. seealso::
        .. [2] Gordon V. Cormack, and Maura R. Grossman. "Engineering quality and reliability in technology-assisted review."
               *Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval.* 2016.
               `<https://dl.acm.org/doi/10.1145/2911451.2911510>`__
    """

    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 1:
            return False

        batchsize = self.stats.annotations_per_round[-1]
        pos_per_round = np.array([self.stats.label_per_round(self.pos_label)])
        pos_found = pos_per_round.cumsum()
        n_docs = self.stats.dataset_size
        n_rounds = self.stats.rounds
        rho_s = -1
        for i in range(n_rounds):
            rho = (pos_found[i] / (i + 1)) / (
                (1 + pos_found[-1] - pos_found[i]) / (n_rounds - i)
            )
            rho_s = max(rho_s, rho)

        return (
            rho_s >= 6
            and batchsize * self.stats.rounds + 1
            >= 10 * n_docs / pos_found[n_rounds - 1]
        ) or (self.stats.current_annotated >= n_docs * 0.75)


class ReviewHalfStoppingRule(StatsStoppingCriterion[KT, LT]):
    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 1:
            return False
        return self.stats.current_annotated >= self.stats.dataset_size // 2


class BatchPrecStoppingRule(StatsStoppingCriterion[KT, LT]):
    def __init__(self, pos_label: LT, prec_cutoff=5 / 200, slack=1) -> None:
        super().__init__(pos_label)
        self.prec_cutoff = prec_cutoff
        self.slack = slack

    @property
    def stop_criterion(self) -> bool:
        bprec = np.array(
            [
                len(batch[self.pos_label])
                / sum([len(batch[k]) for k in batch if k != self.pos_label])
                for batch in self.stats.per_round
            ]
        )
        counter = 0
        for prec in bprec:
            counter = (counter + 1) if prec <= self.prec_cutoff else 0
            if counter >= self.slack:
                return True
        return False


class Rule2399StoppingRule(StatsStoppingCriterion[KT, LT]):
    @property
    def stop_criterion(self) -> bool:
        return (
            self.stats.current_annotated
            >= 1.2 * self.stats.current_label_count(self.pos_label) + 2399
        )


class QuantStoppingRule(StatsStoppingCriterion[KT, LT]):
    """
    .. seealso::
        .. [3] Eugene Yang, David D. Lewis, and Ophir Frieder. "Heuristic stopping rules for technology-assisted review."
               *Proceedings of the 21st ACM Symposium on Document Engineering.* 2021.
               `<https://arxiv.org/abs/2106.09871>`__
    """

    scores: Mapping[KT, float]

    def __init__(self, pos_label: LT, target_recall: float, nstd: float = 0):
        super().__init__(pos_label)
        self.target_recall = target_recall
        self.nstd = nstd
        self.scores = dict()
        self.unknown_ps = float("inf")
        self.known_ps = float("inf")
        self.unknown_var = float("inf")
        self.all_var = float("inf")

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        super().update(learner)
        if hasattr(learner, "classifier") and isinstance(learner.classifier, il.AbstractClassifier):  # type: ignore
            clf: il.AbstractClassifier[Any, Any, Any, Any, Any, LT, npt.NDArray[np.int64], np.float64] = learner.classifier  # type: ignore
            self.scores = {
                key: dict(pred)[self.pos_label]
                for key, pred in clf.predict_proba(learner.env.dataset)
            }
            self.unknown_ps = sum([self.scores[k] for k in learner.env.unlabeled])
            self.known_ps = sum([self.scores[k] for k in learner.env.labeled])
            self.unknown_var = sum(
                [self.scores[k] * (1 - self.scores[k]) for k in learner.env.labeled]
            )
            self.all_var = sum([s * (1 - s) for s in self.scores.values()])

    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 2:
            return False

        # `ps` stands for probability sum

        est_recall = (self.known_ps) / (self.known_ps + self.unknown_ps)
        if self.nstd == 0:
            return est_recall >= self.target_recall

        est_var = (
            self.known_ps**2 / (self.known_ps + self.unknown_ps) ** 4 * self.all_var
        ) + (
            1
            / (self.known_ps + self.unknown_ps) ** 2
            * (self.all_var - self.unknown_var)
        )

        return est_recall - self.nstd * np.sqrt(est_var) >= self.target_recall


class CHMHeuristicsStoppingRule(StatsStoppingCriterion[KT, LT]):
    """
    .. seealso::
        .. [4] Max W. Callaghan, and Finn Müller-Hansen. "Statistical stopping criteria for automated screening in systematic reviews."
               *Systematic Reviews 9.1* (2020): 1-14.
               `<https://pubmed.ncbi.nlm.nih.gov/33248464/>`__
    """

    def __init__(self, pos_label: LT, target_recall: float, alpha: float) -> None:
        super().__init__(pos_label)
        self.target_recall = target_recall
        self.alpha = alpha

    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 2:
            return False

        pos_per_round = np.array([self.stats.label_per_round(self.pos_label)])
        pos_found = pos_per_round.cumsum()
        annotated_cumsum = np.array([self.stats.annotations_per_round]).cumsum()
        n_docs = self.stats.dataset_size

        for i in range(1, self.stats.rounds):
            if (
                hypergeom.cdf(
                    pos_found[-1] - pos_found[i],  # k
                    n_docs - annotated_cumsum[i],  # N
                    int(pos_found[-1] / self.target_recall - pos_found[i]),  # K_tar
                    annotated_cumsum[-1] - annotated_cumsum[i],  # n
                )
                < self.alpha
            ):
                return True
        return False

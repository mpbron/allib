import collections
from typing import Any, Deque, FrozenSet, Generic, Mapping, Sequence
from ..analysis.base import AbstractStatistics, StatsMixin, AnnotationStatistics
from ..activelearning.base import ActiveLearner
from ..typehints.typevars import KT, LT
from .base import AbstractStopCriterion
from scipy.stats import hypergeom
import numpy as np

     
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
            rho = (pos_found[i]/(i+1)) / ((1+pos_found[-1]-pos_found[i])/(self.stats.rounds-i))
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
            rho = (pos_found[i]/(i+1)) / ((1+pos_found[-1]-pos_found[i])/(n_rounds-i))
            rho_s = max(rho_s, rho)

        return  (rho_s >= 6 and batchsize*self.stats.rounds+1 >= 10 * n_docs / pos_found[n_rounds - 1]) or \
                (self.stats.current_annotated >= n_docs*0.75)


class ReviewHalfStoppingRule(StatsStoppingCriterion[KT, LT]):
    
    @property
    def stop_criterion(self) -> bool:
        if self.stats.rounds < 1:
            return False
        return self.stats.current_annotated >= self.stats.dataset_size // 2


class BatchPrecStoppingRule(StatsStoppingCriterion[KT, LT]):

    def __init__(self, pos_label: LT, prec_cutoff=5/200, slack=1) -> None:
        super().__init__(pos_label)
        self.prec_cutoff = prec_cutoff
        self.slack = slack
    
    @property
    def stop_criterion(self) -> bool:
        bprec = np.array([len(batch[self.pos_label]) / sum([len(batch[k]) for k in batch if k != self.pos_label]) for batch in self.stats.per_round])
        counter = 0
        for prec in bprec:
            counter = (counter+1) if prec <= self.prec_cutoff else 0
            if counter >= self.slack:
                return True
        return False


class Rule2399StoppingRule(StatsStoppingCriterion[KT, LT]):
    @property
    def stop_criterion(self) -> bool:
        return self.stats.current_annotated >= 1.2 * self.stats.current_label_count(self.pos_label) + 2399


# class QuantStoppingRule(StoppingRule):
#     """
#     .. seealso::
#         .. [3] Eugene Yang, David D. Lewis, and Ophir Frieder. "Heuristic stopping rules for technology-assisted review." 
#                *Proceedings of the 21st ACM Symposium on Document Engineering.* 2021.
#                `<https://arxiv.org/abs/2106.09871>`__
#     """
#     def __init__(self, target_recall: float, nstd: float = 0):
#         super().__init__(target_recall=target_recall)
#         self.nstd = nstd
    
#     def checkStopping(self, ledger: Ledger, workflow, **kwargs) -> bool:
#         if ledger.n_rounds < 2:
#             return False

#         scores = getOneDimScores(workflow.latest_scores)
            
#         assert (scores <= 1).all() and (scores >= 0).all(), \
#                 "Scores have to be probabilities to use Quant Rule."

#         # `ps` stands for probability sum
#         unknown_ps = scores[ ~ledger.annotated ].sum()
#         known_ps = scores[ ledger.annotated ].sum()
#         est_recall = (known_ps) / (known_ps + unknown_ps)
#         if self.nstd == 0:
#             return est_recall >= self.target_recall
        
#         prod = scores * (1-scores)
#         all_var = prod.sum()
#         unknown_var = prod[ ~ledger.annotated ].sum()

#         est_var = (known_ps**2 / (known_ps + unknown_ps)**4 * all_var) + (1 / (known_ps + unknown_ps)**2 * (all_var-unknown_var))
        
#         return est_recall - self.nstd*np.sqrt(est_var) >= self.target_recall

class CHMHeuristicsStoppingRule(StatsStoppingCriterion[KT, LT]):
    """
    .. seealso::
        .. [4] Max W. Callaghan, and Finn Müller-Hansen. "Statistical stopping criteria for automated screening in systematic reviews." 
               *Systematic Reviews 9.1* (2020): 1-14.
               `<https://pubmed.ncbi.nlm.nih.gov/33248464/>`__
    """
    def __init__(self, pos_label: LT, target_recall: float, alpha: float) -> None:
        super().__init__(pos_label)
        self.target_recall=target_recall
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
            if hypergeom.cdf( pos_found[-1] - pos_found[i], # k
                              n_docs-annotated_cumsum[i], # N
                              int(pos_found[-1]/self.target_recall - pos_found[i]), # K_tar
                              annotated_cumsum[-1] - annotated_cumsum[i] # n
                            ) < self.alpha:
                return True
        return False

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, List, Any

from sklearn.exceptions import NotFittedError # type: ignore
import numpy as np # type: ignore

DT = TypeVar("DT")
CT = TypeVar("CT")
LT = TypeVar("LT")

class BaseVectorizer(ABC, Generic[DT]):
    """This is the :class:`~abc.ABC` specifies a generic vectorizer.
    Vectorizers transform raw data examples into feature vectors.
    Given a data type `DT`, it specifies the methods :meth:`~.fit`
    that initializes or fits the vectorizer. The method :meth:`~.transform`
    transforms the data into vector form.
    """    
    _name = "BaseVectorizer"
    
    def __init__(self):
        self._fitted = False
    
    @property
    def fitted(self) -> bool:
        """Check if the vectorizer has been fitted

        Returns
        -------
        bool
            True if the vectorizer has been fitted
        """        
        return self._fitted

    @abstractmethod
    def fit(self, x_data: Sequence[DT], **kwargs: Any) -> BaseVectorizer[DT]:
        """Fit the vectorizer according to the data in the given 
        :class:`~collections.abc.Sequence`.

        Parameters
        ----------
        x_data : Sequence[DT]
            A Sequence of examples with type `DT`.

        Returns
        -------
        BaseVectorizer[DT]
            A fitted vectorizer for data with type `DT`

        Examples
        --------
        Assume the creation of a vectorizer and a sequence of data examples
        in the variable `data_list`
        
        >>> vectorizer = BaseVectorizer[DT]()
        >>> vectorizer = vectorizer.fit(data_list)
        """        
        pass

    @abstractmethod
    def transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        """Transform a list raw data points to a feature matrix 
        according to the fitted vectorizer

        Parameters
        ----------
        x_data : Sequence[DT]
            A sequence of raw data examples with length `n_examples`

        Returns
        -------
        np.ndarray
            A feature matrix with shape `(n_examples, n_features)`

        Examples
        --------
        Assume the vectorizer is fitted
        
        >>> x_mat = vectorizer.transform(x_data)
        """        
        pass

    @abstractmethod
    def fit_transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        """Transform a list of data to a feature matrix. The transformation
        is based on the data contained in the parameter `x_data`. Subsequent
        transformations with :meth:`~.transform()` will be based on the fit
        of the data provided in this call.

        Parameters
        ----------
        x_data : Sequence[DT]
            A sequence of raw data examples with length `n_examples`

        Returns
        -------
        np.ndarray
            A feature matrix with shape `(n_examples, n_features)`

        Examples
        --------
        Assume the vectorizer is fitted
        
        >>> x_mat = vectorizer.fit_transform(x_data)
        """
        pass

class SeparateContextVectorizer(ABC, Generic[DT, CT]):
    """This :class:`~abc.ABC` specifies a generic vectorizer for data types
    that consists of two parts that have to be fitted or configured according
    to different specifications. The feature vectors of the two parts
    are concatenated for each example.

    The two parts are referred to the `data` part and the `context` part.
    This vectorizer contains two inner vectorizer, one for the data part and 
    one for the context part respectively.

    Arguments
    ---------
    data_vectorizer : BaseVectorizer[DT]
            The vectorizer for the data part
    context_vectorizer : BaseVectorizer[CT]
            The vectorizer for the context part
    
    Examples
    --------
    Construction:

    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> data_vectorizer = SklearnVectorizer[str](TfidfVectorizer())
    >>> context_vectorizer = Doc2VecVectorizer[str]()
    >>> vectorizer = SeparateContextVectorizer[str, str](data_vectorizer, 
    ...     context_vectorizer)

    Fitting:

    >>> x_data = ["This...", "Another text...", ... "Last Text"]
    >>> x_context_data = ["Surrounding text", ... , "Another text"]
    >>> vectorizer = vectorizer.fit(x_data, x_context_data)

    Transforming:

    >>> x_mat = vectorizer.transform(x_data, x_context_data)
    """    
    
    _name = "SeparateContextVectorizer"
    
    def __init__(
            self,
            data_vectorizer: BaseVectorizer[DT],
            context_vectorizer: BaseVectorizer[CT]
        ):
        self.data_vectorizer = data_vectorizer
        self.context_vectorizer = context_vectorizer

    @property
    def fitted(self) -> bool:
        """Check if the vectorizer has been fitted

        Returns
        -------
        bool
            True if the vectorizer has been fitted
        """        
        return self.data_vectorizer.fitted and self.context_vectorizer.fitted

    def fit(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs: Any) -> SeparateContextVectorizer[DT, CT]:
        """Fit the vectorizer according to the data in the given 
        :class:`~collections.abc.Sequence` s.

        Parameters
        ----------
        x_data : Sequence[DT]
            The data parts
        context_data : Sequence[CT]
            The contexts parts

        Returns
        -------
        SeparateContextVectorizer[DT, CT]
            A fitted vectorizer

        Examples
        --------
        Fitting this vectorizer can be performed as follows:

        >>> x_data = ["This...", "Another text...", ... "Last Text"]
        >>> x_context_data = ["Surrounding text", ... , "Another text"]
        >>> data_vectorizer = SklearnVectorizer[str](TfidfVectorizer())
        >>> context_vectorizer = Doc2VecVectorizer[str]()
        >>> vectorizer = SeparateContextVectorizer[str, str](
        ...     data_vectorizer, 
        ...     context_vectorizer)
        >>> vectorizer = vectorizer.fit(x_data, x_context_data)

        Warning
        -------
        We assume that the variables `x_data` and `context_data` are sequences
        of equal length.
        """        
        self.data_vectorizer.fit(x_data, **kwargs)
        self.context_vectorizer.fit(context_data, **kwargs)
        return self

    def transform(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs: Any) -> np.ndarray: # type: ignore
        """Transform a list raw data points to a feature matrix 
        according to the fitted vectorizers


        Parameters
        ----------
        x_data : Sequence[DT]
            A sequence with data parts of the data points of length `n_docs`
        context_data : Sequence[CT]
            A sequence with context part of the data points of length `n_docs`

        Returns
        -------
        np.ndarray
            A feature matrix of concatenated vectors with shape
            `(n_docs, n_features_data + n_features_context)`

        Raises
        ------
        NotFittedError
            If the model is not fitted
        
        Warning
        -------
        We assume that the variables `x_data` and `context_data` are sequences
        of equal length and that the indices of the sequences correspond to the
        same data point.
        """        
        if self.fitted:
            data_part: np.ndarray = self.data_vectorizer.transform(x_data, **kwargs) # type: ignore
            context_part: np.ndarray = self.context_vectorizer.transform( # type: ignore
                context_data, **kwargs) # type: ignore
            return np.concatenate((data_part, context_part), axis=1) # type: ignore
        raise NotFittedError

    def fit_transform(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs: Any) -> np.ndarray: # type: ignore
        """Fit and transform a list raw data points to a feature matrix 
        according to the fitted vectorizers. Subsequent
        transformations with :meth:`~.transform()` will be based on the fit
        of the data provided in this call.


        Parameters
        ----------
        x_data : Sequence[DT]
            A sequence with data parts of the data points of length `n_docs`
        context_data : Sequence[CT]
            A sequence with context part of the data points of length `n_docs`

        Returns
        -------
        np.ndarray
            A feature matrix of concatenated vectors with shape
            `(n_docs, n_features_data + n_features_context)`
        """        
        self.fit(x_data, **kwargs)
        return self.transform(x_data, context_data, **kwargs) # type: ignore


class StackVectorizer(BaseVectorizer[DT], Generic[DT]):
    """This :class:`~abc.ABC` specifies a generic vectorizer that consists of 
    several vectorizers that are fitted on the same data points. 
    
    The feature vectors of the contained vectorizers are concatenated in the 
    transform step, according to the order they are specified in the 
    constructor (argument order).
    
    Arguments
    ----------
    vectorizer : BaseVectorizer[DT]
        At least one vectorizer is required
    *vectorizers: BaseVectorizer[DT]
        Any number of vectorizers for the same data type

    Examples
    --------
    Construction

    >>> tf_idf = SklearnVectorizer[str](TfidfVectorizer())
    >>> doc2vec = Doc2VecVectorizer[str]()
    >>> count = SklearnVectorizer[str](CountVectorizer())
    >>> vectorizer = StackVectorizer[str](tfidf, doc2vec, count)

    Fitting

    >>> x_data = ["This...", "Another text...", ... "Last Text"]
    >>> vectorizer = vectorizer.fit(x_data)

    Transforming

    >>> another_data = ["Another test text", ... , "Another text"]
    >>> x_mat = vectorizer.transform(x_data)    
    """    
    
    vectorizers: List[BaseVectorizer[DT]]
    """The internal vectorizers are stored in this list"""    
    
    _name = "StackVectorizer"

    def __init__(self,
                 vectorizer: BaseVectorizer[DT],
                 *vectorizers: BaseVectorizer[DT]) -> None:
        """[summary]

        Parameters
        ----------
        vectorizer : BaseVectorizer[DT]
            [description]
        """        
        super().__init__()
        self.vectorizers = [vectorizer, *vectorizers]

    def fit(self, x_data: Sequence[DT], **kwargs: Any) -> StackVectorizer[DT]:
        for vec in self.vectorizers:
            vec.fit(x_data, **kwargs)
        return self

    def fitted(self) -> bool:
        return all([vec.fitted for vec in self.vectorizers])

    def transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        if self.fitted:
            sub_vectors = [ # type: ignore
                vec.transform(x_data, **kwargs) # type: ignore
                for vec in self.vectorizers]
            return np.concatenate(sub_vectors, axis=1) # type: ignore
        raise NotFittedError

    def fit_transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        self.fit(x_data, **kwargs)
        return self.transform(x_data, **kwargs) # type: ignore

from multiprocessing import Process
import threading
import random
import logging
import time
from queue import Queue, Empty
from typing import Generic, TypeVar, Union, Iterator, Callable, List, Sequence
PT = TypeVar("PT")
CT = TypeVar("CT")

LOGGER = logging.getLogger(__name__)

class Sentinel(object):
    pass


def producer_func(input: Callable[..., Iterator[PT]], 
             output: Queue, 
             n_consumers) -> None:
    iterator = input()
    for item in iterator:
        output.put(item)
        print("Produced an item")
    for _ in range(n_consumers):
        output.put(Sentinel())
    return


def consumer_func(input: Queue,
             output: Queue,
             func: Callable[[PT], CT]) -> None:
    while True:
        if not input.empty():
            try:
                item = input.get()
            except Empty:
                continue
            else:
                if isinstance(item, Sentinel):
                    return
                result = func(item)
                output.put(result)
                input.task_done()
            print("Consumed an item")


class ProducerConsumer(Generic[PT, CT]):
    def __init__(self,
                 n_consumers: int = 1,
                 ):
        self.input_queue = Queue()
        self.result_queue = Queue()
        self.n_consumers = n_consumers

    def map_async(self,
                  function: Callable[[PT], Iterator[CT]],
                  iterator_builder: Callable[..., Iterator[CT]]) -> Sequence[CT]:
        producer = Process(target=producer_func, args=(iterator_builder, self.input_queue, self.n_consumers))
        producer.start()
        LOGGER.info("Producer has started")
        consumers = [
            Process(target=consumer_func, args=(self.input_queue, self.result_queue, function)) for _ in range(self.n_consumers)
        ]
        for c in consumers:
            c.start()
        LOGGER.info("Started consumers, waiting for, producer to join")
        producer.join()
        LOGGER.info("Producer has joined, waiting for consumers to join")
        for c in consumers:
            c.join()
            LOGGER.info("Consumer has joined")
        LOGGER.info("Consumers have joined")
        results: List[CT] = []
        LOGGER.info("Retrieving all info from result queue")
        while True:
            try:
                elem = self.result_queue.get()
                results.append(elem)
            except Empty:
                break
        LOGGER.info("Retrieved all info from result queue, returning results to caller")
        return results




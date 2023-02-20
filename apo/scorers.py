from typing import Any, Dict, Union
from abc import ABC
from dataclasses import dataclass, field
import os
import numpy as np
import logging
import io
import contextlib

import torch
import scrubadub
import scrubadub_spacy
import pycodestyle
from detoxify import Detoxify
import wandb

from .scorer_utils import AddressDetectorNoLibpostal, DateOfBirthDetectorNonNan


logger = logging.getLogger(__name__)


@dataclass
class LMSamples:
    prompts: list[Union[str, wandb.Html]] = field(default_factory=list)
    continuations: list[Union[str, wandb.Html]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        if self.prompts is not None:
            return ['prompt', 'continuation', 'score']
        else:
            return ['continuation', 'score']

    def __iter__(self):
        if self.prompts is not None:
            rows = zip(self.prompts, self.continuations, self.scores)
        else:
            rows = zip(self.continuations, self.scores)
        return iter(rows)

    def __len__(self):
        return len(self.continuations)

    def __add__(self, other):
        return LMSamples(
            prompts=self.prompts + other.prompts,
            continuations=self.continuations + other.continuations,
            scores=self.scores + other.scores
        )

    def display_as_html(self) -> 'LMSamples':
        """Return a new LMSamples instance with prompts and continuations embedded in HTML that makes code look
        nicely"""
        return LMSamples(
            prompts=[wandb.Html(self._generate_html(prompt)) for prompt in self.prompts],
            continuations=[wandb.Html(self._generate_html(continuation)) for continuation in self.continuations],
            scores=self.scores
        )

    def _generate_html(self, text: str) -> str:
        return f"""
        <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.0.3/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.0.3/highlight.min.js"></script>
        <script>hljs.initHighlightingOnLoad();</script>
        </head>
        <body><pre><code class="python">{text}</code></pre></body>"""


class Scorer(ABC):
    """
    Scorer is an abstraction of a computation needed for determining whether a piece of text is aligned or misaligned.
    A scorer can be implemented by a learned reward model or a simpler rule-based heuristic (using a blacklist of
    disallowed words).
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        class_name = config.pop('class_name')
        return globals()[class_name](**config)

    def score_text(self, text: str) -> float:
        raise NotImplementedError('A subclass of Scorer must implement score_text')

    def score_texts(self, texts: list[str]) -> list[float]:
        # Naive implementation that can be overridden by subclasses that can do smarter batch scoring
        return [self.score_text(text) for text in texts]

    def score_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a single HuggingFace dataset element with computed scores: a document-level `score` (float) and possibly
        `span_scores` (a list of dicts with `begin` and `end` keys and a `score` key)
        """
        # By default, only document score is computed but subclasses can override this method to compute span scores
        element['score'] = self.score_text(element['text'])
        return element

    def score_elements(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a batch of HuggingFace dataset elements with computed scores: for each element (document), a
        document-level `score` (float) and  possibly `span_scores` (a list of dicts with `begin` and `end` keys and a
        `score` key)
        """
        # By default, only document score is computed but subclasses can override this method to compute span scores
        element['score'] = self.score_texts(element['text'])
        return element

    def score_samples(self, samples: LMSamples, use_prompt_for_scoring: bool = False) -> LMSamples:
        """
        Update LMSamples with computed scores
        """
        if use_prompt_for_scoring:  # useful for e.g. code generation
            prompts = [prompt.replace('<|aligned|>', '') for prompt in samples.prompts]
            texts = [f'{prompt}{continuation}' for prompt, continuation in zip(prompts, samples.continuations)]
        else:
            texts = samples.continuations
        return LMSamples(
            prompts=samples.prompts,
            continuations=samples.continuations,
            scores=self.score_texts(texts=texts)
        )


class DetoxifyToxicityScorer(Scorer):

    def __init__(self, device: Union[str, int, torch.device] = 0, keep_on_device: bool = False):
        self.device = device
        self.detoxify = Detoxify('unbiased')
        self.keep_on_device = keep_on_device

    def score_text(self, text: str) -> float:
        self.detoxify.model.to(self.device)
        score = self.detoxify.predict(text)['toxicity']
        if not self.keep_on_device:
            self.detoxify.model.to('cpu')
        return score

    def score_texts(self, texts: list[str]) -> list[float]:
        self.detoxify.model.to(self.device)
        scores = self.detoxify.predict(texts)['toxicity']
        if not self.keep_on_device:
            self.detoxify.model.to('cpu')
        return scores


class PIIScorer(Scorer):
    """
    Scores text on PII: count number of PII objects in each document (as a float).
    If no PII is found, return 0.0.
    """

    def __init__(self):
        """
        Create scrubber and add all optional detectors.
        """

        self.scrubber = scrubadub.Scrubber()
        self.scrubber.add_detector(DateOfBirthDetectorNonNan)
        self.scrubber.add_detector(scrubadub.detectors.SkypeDetector)
        self.scrubber.add_detector(scrubadub_spacy.detectors.SpacyEntityDetector(model='en_core_web_sm'))
        self.scrubber.add_detector(AddressDetectorNoLibpostal)

    def score_text(self, text: str) -> float:
        """
        Return number of PII objects in text as a float.
        """
        try:
            return len(list(self.scrubber.iter_filth(text)))/len(text)
        except (ValueError, RecursionError, OverflowError, ZeroDivisionError) as exception:
            print(exception)
            return 0.0


class PEP8Scorer(Scorer):

    def score_text(self, text: str) -> float:
        """
        Return number of PEP8 violations per character in text as a float.
        """
        virtual_file = io.StringIO(text)
        checker = pycodestyle.Checker(lines=virtual_file.readlines(), show_source=True)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):  # keep stdout clean
            try:
                num_violations = checker.check_all()
            except (UnicodeEncodeError, IndexError):
                num_violations = 0   # this should be rare enough to not worry about
        try:
            score = num_violations/len(text)
        except ZeroDivisionError:
            score = 0   # this should be rare enough to not worry about
        return score


class PEP8LineScorer(Scorer):

    def score_text(self, text: str) -> list:
        """
        Return list of PEP8 violations per character in each line of text.
        """
        virtual_file = io.StringIO(text)
        checker = pycodestyle.Checker(lines=virtual_file.readlines(), show_source=True)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):  # keep stdout clean
            try:
                _ = checker.check_all()
                scores = np.zeros(len(checker.lines))
                for line_number, offset, code, text, doc in checker.report._deferred_print:
                    scores[line_number-1] += 1
                scores = scores/[len(line) for line in checker.lines]
            except (UnicodeEncodeError, ZeroDivisionError, IndexError):
                scores = np.zeros(len(checker.lines))  # this should be rare enough to not worry about
        return scores.tolist()

### Taken from https://github.com/bbuing9/ICLR24_SuRe/blob/main/data_utils.py

import numpy as np
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Sequence, Tuple, Union
from dataclasses import dataclass, field
import re
import unicodedata
import string
from collections import Counter
@dataclass
class Question:
    text: str
    answers: Union[Set[str], List[str]]
    id: Optional[str] = None
    tokens: Optional[List[str]] = field(default=None)
    acceptable_answers: Optional[List[str]] = field(default=None)
    unacceptable_answers: Optional[List[str]] = field(default=None)

    @property
    def has_answers(self) -> bool:
        return self.answers and len(self.answers) > 0

    @property
    def has_annotated_answers(self) -> bool:
        return len(self.gold_answers) > 0 or self.unacceptable_answers

    @property
    def tokenized_text(self) -> Optional[str]:
        return " ".join(self.tokens) if self.tokens is not None else None

    def update_answers(self, annotated_answers):
        if not annotated_answers:
            return

        self.acceptable_answers = annotated_answers["yes"]
        self.unacceptable_answers = annotated_answers["no"]

    def is_unacceptable(self, candidate_answer: str) -> bool:
        if self.unacceptable_answers:
            for ans in self.unacceptable_answers:
                if candidate_answer == ans or candidate_answer.lower() == ans.lower():
                    return True

        return False

    @property
    def gold_answers(self) -> Set[str]:
        answers = set(self.answers) if self.answers else set()

        if self.acceptable_answers:
            answers.update(self.acceptable_answers)

        if self.unacceptable_answers:
            for a in self.unacceptable_answers:
                if a in answers:
                    answers.remove(a)
                elif a.lower() in answers:
                    answers.remove(a.lower())

        return answers

    def to_json(self) -> Dict[str, Any]:
        json_dict = dict(
            question=self.text,
            id=self.id,
            answers=self.answers,
        )

        return json_dict

    @classmethod
    def from_json(cls, q_dict, idx: int = 0):
        return Question(
            q_dict["question"],
            q_dict.get("answer", q_dict.get("answers", None)),
            q_dict.get("id", idx),
        )

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            _normalize(pattern),
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(_normalize(text)) is not None

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def em_eval(question: Question, candidate_answer: str, match: str = "string") -> int:
    if not question.gold_answers:
        if question.is_unacceptable(candidate_answer):
            return 0
        else:
            return -1

    return int(
        metric_max_over_ground_truths(
            regex_match if match == "regex" else exact_match_score,
            candidate_answer,
            question.gold_answers,
        )
    )

def f1_eval(question: Question, candidate_answer: str) -> float:
    if not question.gold_answers:
        if question.is_unacceptable(candidate_answer):
            return 0
        else:
            return -1

    return metric_max_over_ground_truths(
        f1_score,
        candidate_answer,
        question.gold_answers,
    )


def get_em_f1(dataset, preds):
    res_em = []
    res_f1 = []
    for i, item in enumerate(dataset):
        q = Question(item['question'], item['answers'])
        if type(preds[i]) == list:
            preds_i = preds[i][0]
        else:
            preds_i = preds[i]
        em = em_eval(q, preds_i)
        f1 = f1_eval(q, preds_i)
        res_em.append(em)
        res_f1.append(f1)
    return np.array(res_em), np.array(res_f1)



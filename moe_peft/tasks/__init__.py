from . import glue_tasks, qa_tasks, arabic_tasks
from .common import (
    AutoMetric,
    BasicMetric,
    BasicTask,
    CasualTask,
    CommonSenseTask,
    MultiTask,
    SequenceClassificationTask,
    task_dict,
)
from .qa_tasks import QuestionAnswerTask
from .arabic_tasks import ArabicMCQTask

glue_tasks.update_task_dict(task_dict)
qa_tasks.update_task_dict(task_dict)
arabic_tasks.update_task_dict(task_dict)


__all__ = [
    "BasicMetric",
    "AutoMetric",
    "BasicTask",
    "CasualTask",
    "SequenceClassificationTask",
    "CommonSenseTask",
    "QuestionAnswerTask",
    "ArabicMCQTask",
    "MultiTask",
    "task_dict",
]

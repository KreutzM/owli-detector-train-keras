from owli_train.pseudo_label.teacher_tfhub import (
    TeacherModel,
    _find_input_batch_size,
    _resolve_effective_batch_size,
)


class _TensorSpec:
    def __init__(self, shape):
        self.shape = shape


class _Signature:
    structured_input_signature = ((), {"images": _TensorSpec((1, 640, 640, 3))})


def test_find_input_batch_size_reads_fixed_batch_dimension() -> None:
    assert _find_input_batch_size(_Signature()) == 1


def test_resolve_effective_batch_size_overrides_fixed_teacher_batch() -> None:
    teacher = TeacherModel(
        runner=object(),
        source="savedmodel",
        input_dtype_name="float32",
        signature_name="serving_default",
        input_batch_size=1,
    )

    assert _resolve_effective_batch_size(teacher=teacher, requested_batch_size=8) == 1


def test_resolve_effective_batch_size_keeps_requested_batch_when_unconstrained() -> None:
    teacher = TeacherModel(
        runner=object(),
        source="savedmodel",
        input_dtype_name="float32",
        signature_name="serving_default",
        input_batch_size=None,
    )

    assert _resolve_effective_batch_size(teacher=teacher, requested_batch_size=8) == 8

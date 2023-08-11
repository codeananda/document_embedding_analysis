from main import compare_documents_plans, compare_documents_sections


def is_approximately_equal(x, y, epsilon=1e-10):
    """Return True if two numbers are close in value. Cannot compare
    directly due to floating-point roundoff error."""
    return abs(x - y) < epsilon


def test_compare_documents_plans_same_document():
    result = compare_documents_plans(
        "output/wikipedia/Dual-phase evolution.json",
        "output/wikipedia/Dual-phase evolution.json",
    )

    for value in result["plan_total_similarity"].values():
        assert is_approximately_equal(value, 1.0)


def test_compare_documents_sections_same_document():
    result = compare_documents_sections(
        "output/wikipedia/Dual-phase evolution.json",
        "output/wikipedia/Dual-phase evolution.json",
    )

    for value in result["content_total_similarity"].values():
        assert is_approximately_equal(value, 1.0)


def test_compare_documents_plans_different_documents():
    result = compare_documents_plans(
        "output/wikipedia/Dual-phase evolution.json",
        "output/wikipedia/Climate Change.json",
    )

    for value in result["plan_total_similarity"].values():
        assert not is_approximately_equal(value, 1.0)


def test_compare_documents_sections_different_documents():
    result = compare_documents_sections(
        "output/wikipedia/Dual-phase evolution.json",
        "output/wikipedia/Climate Change.json",
    )

    for value in result["content_total_similarity"].values():
        assert not is_approximately_equal(value, 1.0)

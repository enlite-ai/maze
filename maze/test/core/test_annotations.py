"""Contains unit test for @override annotation."""

from __future__ import annotations

from maze.core.annotations import override

from pytest import raises


class A:
    """Base Class"""

    def method(self) -> None:
        """method to override"""
        pass


def test_override():
    """test override"""

    # valid override
    class B(A):
        """Sub Class"""

        @override(A)
        def method(self) -> None:
            """override possible"""
            pass

    # invalid override
    with raises(NameError):

        class C(A):
            """Sub Class"""

            @override(A)
            def another_method(self) -> None:
                """override not possible"""
                pass

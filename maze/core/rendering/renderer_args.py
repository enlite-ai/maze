"""Additional arguments exposed by the renderer for more complex envs."""

from abc import abstractmethod, ABC
from typing import Union, Any, Tuple, List


class RendererArg(ABC):
    """Interface for classes exposing arguments available at renderers.

    Example such argument: ID of a distribution center that we want to render a detail of.

    Subclasses of this class should also define how to convert these argument definitions into interactive
    controls that can be displayed to the user (currently only ipython widgets, though more types of controls
    can be added in the future).

    :param name:  Name of the argument as it should be passed to the renderer
    :param title: Name of the argument as it should be displayed to the user
    """

    def __init__(self, name: str, title: str):
        self.name = name
        self.title = title

    @abstractmethod
    def create_widget(self):
        """Build an ipython widget that can be used to control the value of this argument by the user."""
        raise NotImplementedError


class IntRangeArg(RendererArg):
    """Represents an argument which can take on a value of integer in a particular range.

    :param min_value: Min allowed value
    :param max_value: Max allowed value
    """

    def __init__(self, name: str, title: str, min_value: int, max_value: int):
        super().__init__(name, title)
        self.min_value = min_value
        self.max_value = max_value

    def create_widget(self):
        """Build an int slider widget."""
        import ipywidgets as widgets
        return widgets.IntSlider(
            description=self.title,
            min=self.min_value,
            max=self.max_value,
            step=1,
            continuous_update=False
        )


class OptionsArrayArg(RendererArg):
    """Represents an argument where a single value can be chosen from an array of allowed options.

    :param options: Array of allowed options. Either just a simple array of allowed argument values,
      or an array of tuples, each in the form of `(value_displayed_to_the_user, value_passed_to_renderer)`
    """

    def __init__(self, name: str, title: str, options: List[Union[Any, Tuple[str, Any]]]):
        super().__init__(name, title)
        self.options = options

    def create_widget(self):
        """Build a dropdown widget."""
        import ipywidgets as widgets
        return widgets.Dropdown(
            description=self.title,
            options=self.options
        )

# Coding Guidelines
In general we adhere to the coding style defined in [PEP 8](https://www.python.org/dev/peps/pep-0008/) as implemented in the PyCharm code inspections<sup id="anchor_1">[1](#footnote_1)</sup>. 

In addition to these basic Python style conventions, we extend this document by best practices, driven by the challenges we face in creating and maintaining a complex codebase for reinforcement learning.

## Type Hints

### Package “typing”
Type hints should be as specific as possible and also specify nested types. In the case of standard collections, this can be done by using the collection types from the typing package (Dict instead of the built-in lowercase dict).

```-> Dict[str, List[int]]```   
(instead of just using “```-> dict```“ or “```-> Dict[str, list]```”)

In the rare cases where the content type of a collection is not known in advance, we prefer to make this explicit by using   
```-> Dict[Any, List[Any]]``` or even ```-> Dict[Any, Any]```

The built-in collection types should not be used as type hints (lowercase ```list```, ```dict```, ```set```).

### Any or any
Always use to ```Any``` class from the typing package, as the built-in ```any``` is not recognized by the documentation system.

### Import conflicts
Due to regular confusions of ```gymnasium.spaces``` and ```typing``` imports we decided to follow the convention below:

For type hinting we use direct imports such and avoid built in types such as tuple or dict:

```python
from typing import Dict, Tuple
-> Tuple[int, int], -> Dict[str, int]
```

For gym spaces we follow this import scheme:

```python
from gymnasium import spaces
-> spaces.Dict, -> spaces.Tuple
```

### Configuration

No default config parameters

### Documentation Guidelines

**Stay consistent with the style prevalent in the files you touch, even if it violates the rules below!**

#### Documentation Scope

* All functions and methods must provide type hints for their arguments and their return values.
* All public functions and methods must provide docstrings with complete :param and :return blocks (except the function returns None)
* Instance attributes must be defined with type hints, if the type can not be inferred from the initial assignment.
* Use the **```@override```** annotation to document method overrides
* Provide a module docstring at the beginning of every Python file, describing why the file exists (a good place to motivate and provide background to the data structure or algorithm in question, as opposed to the more narrow class documentation)

#### Avoid redundant text in the documentation markup

As with source code, every line of documentation contributes refactoring complexity and maintenance overhead.

* Do not copy the text from the parent class implementation, instead describe what changed with respect to the parent method. The ‘:param’ and/or ‘:return:’ blocks can be skipped, if there is no new information to add to the existing parent method documentation.
* Make use of [Sphinx references](https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#ref-role).

#### Capitalization and Punctuation

While there is no strict policy on when to capitalize and punctuate argument/function/class/... descriptions, it is recommended to use full sentences (i.e. proper capitalization and punctuation) in paragraphs. If the text snippet in question is only one (full or partial) sentence, the description of parameter and return values can be shortened to code-comment style (i.e. without proper capitalization and punctuation). But this has to be consistent: If there is one full sentence in the description, all text needs to be phrased as full sentences.

#### Documentation Examples

```python
"""Provide a module docstring at the beginning of every Python file, describing why the file exists.

A good place to motivate and provide background to the data structure or algorithm in question, as opposed to
the more narrow class documentation
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from maze.core.annotations import override


class Interface(ABC):
   """One-line summary.

   Optional Paragraphs

   :param constructor_arg: Note that the __init__ arguments are documented in the class docstring.

   :ivar instance_variable: Optionally document instance variables.
   """

   def __init__(self, constructor_arg: str):
       # no docstring for __init__
       self.instance_variable = constructor_arg

       # provide type hints if the type can not be inferred from the initial assignment
       self.typed_instance_variable: Optional[str] = None

   @abstractmethod
   def interface_method(self, param1: List[int]) -> str:
       """One-line summary, not using variable names or the function name.

       All public functions and methods must provide docstrings with complete
       ``:param`` and ``:return`` blocks (except the function returns None)

       :param param1: Description of the parameter, mandatory.

       :return Description of the method return.
       """
       raise NotImplementedError()


class DerivedClass(Interface):
   """One-line summary.

   Optional description.
   """

   @override(Interface)  # override is mandatory
   def interface_method(self, param1: List[int]) -> str:
       """Do not copy the text from the parent class, instead describe what changed with respect to the parent method.

       The ``:param`` and/or ``:return:`` blocks can be skipped, if there is no new information to add to the existing
       parent method documentation.

       Other examples of valid docstrings:
       * "Implementation of ``Interface``."
         Bare minimum, in case there is absolutely nothing to be said about the
         implementation that is not already described in the super method.
       * "Forward call to :attr:`self.core_env <core_env>`."
         If the interface is implemented in some other method/class.
       """
       pass
```

## Reward Computation

Separation of the reward calculation from the actual simulation logic is highly desirable. To accomplish this, we utilize the _[Observer Design Pattern](https://gameprogrammingpatterns.com/observer.html)_ by introducing event interfaces.

In the simple case of only one event interface, the reward calculation can take place in a class derived from this interface. While this approach is appropriate to learn the basics of environment implementations, it is not suitable for modularized environments or multi-agent settings. Typically we encounter multiple environment components emitting events to multiple reward calculations. Therefore we extend the event interface concept by a publisher-subscriber model, where the environment publishes by invoking the event interface and an arbitrary number of reward aggregators receive and collect those messages in order to calculate the reward at the end of the step function, realizing referential and temporal decoupling<sup id="anchor_2">[2](#footnote_2)</sup>.

[More details on reward collection](https://docs.google.com/presentation/d/1NMEsP1Iu5895axBR3MMcjNyOXbmAgYy3PQuC1tzV87c/edit#slide=id.p). 

<b id="footnote_1">1</b> E.g. PEP 8 defines a max line length of 79, but we set this limit to 120, in line with the PyCharm default settings. [↩](#anchor_1)
<b id="footnote_2">2</b> https://arothuis.nl/posts/messaging-pub-sub/ [↩](#anchor_2)


## Observation and Action Spaces

When implementing new environments we currently (only) support flat dictionary spaces (gymnasium.spaces.Dict).

### Action Spaces
The action space dictionary holds the actual sub-action spaces (gymnasium.spaces.MultiBinary, gymnasium.spaces.Categorical, ...)
as mapping of string action keys to gym spaces:

```
action_dict = dict()
action_dict["action_0"] = gymnasium.spaces.MultiBinary(5)
action_dict["action_1"] = gymnasium.spaces.Categorical(5)

action_space = gymnasium.spaces.Dict(spaces=action_dict)
```

### Observation Spaces

Observation spaces are dictionary spaces with gymnasium.spaces.Box observation sub-spaces where keys should be again strings
identifying the respective observation. The datatype of the gymnasium.spaces.Box sub-spaces should be np.float32 already from
the beginning. Also make sure to set low and high bounds properly.

```
observation_dict: dict = dict()
observation_dict["observation_0"] = gymnasium.spaces.Box(low=0, high=10, shape=(100,), dtype=np.float32)
observation_dict["observation_1"] = gymnasium.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

observation_space = gymnasium.spaces.Dict(spaces=action_dict)
```

This has several reasons:
 - When training on GPUs the standard data type is anyways float32 and a cast is required
 - When applying observation normalization (e.g. mean zero, std one) we also need to cast to float32 to get a valid result

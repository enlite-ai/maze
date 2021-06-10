# Contributing to <img align="center" src="https://github.com/enlite-ai/maze/raw/main/docs/source/logos/main_logo.png" alt="Maze" width=250 height="80">

We welcome and encourage all community contributions to Maze. If you're reading this, you might be interested in participating - thank you! 

This document outlines what to consider when contributing to Maze. While we recommend adhering to these guidelines, don't fret over them - especially if you are new to Maze. 

**How can I contribute to Maze?** You can help to improve Maze and the community around it in several ways:
* [Report issues](#bugs-and-feature-requests). If you encounter a bug, an inaccuracy in or lack of documentation or just want to requst a feature, create an issue to let us know. We will get back to you as soon as possible.
* Participate in the community: [Ask and/or answer questions, initiate and/or join discussions](#questions-discussions-etc). A supportive community is invaluable to an open-source project.
* [Improve the documentation](#improve-the-documentation). A good an extensive documentation is crucial. Whether it is a fixed typo, rephrasing for clarity's sake or entirely new documentation, we highly appreciate it.
* [Improve the code](#improve-the-code). Want to get your hands dirty and make life easier for yourself and your fellow developers? We'll support you in finding appropriate issues and will guide you through the process of setting up, developing and finally creating a pull request.

In the following you'll find them explained in more detail. If you feel like something is missing from this document, feel free to open an issue or shoot us a question on [Github discussions](https://github.com/enlite-ai/maze/discussions).

## General Guidelines

Please have a look at our [code of conduct](https://github.com/enlite-ai/maze/blob/main/CODE_OF_CONDUCT.md) if you consider contributing.

## Reporting Issues

### Bugs and Feature Requests

Report these issues via the [Github issue tracker](https://github.com/enlite-ai/maze/issues). Please adhere to the 
provided issue template.   
If you think you found a security issue, please have a look at our [security policy](https://github.com/enlite-ai/maze/blob/main/SECURITY.md) before opening an issue.  

### Questions, Discussions, etc.

If you have questions on how to use Maze or seek clarification on something insufficiently explained in the [documentation](https://maze-rl.readthedocs.io/en/latest/), post a question on [StackOverflow (tagged `maze-rl`)](https://stackoverflow.com/questions/tagged/maze-rl) or start a conversation on [GitHub discussions](https://github.com/enlite-ai/maze/discussions).

Open-ended questions, discussions etc. are best posted to [Github discussions](https://github.com/enlite-ai/maze/discussions).

## Contributing Changes to the Code Base

### Setting up your Development Environment

If you want to contribute changes, you'll have to set up a local development environment first. If you want to use [pip](https://pip.pypa.io/en/stable/), follow the installation instructions [here](https://maze-rl.readthedocs.io/en/latest/getting_started/installation.html#installation).   
If you prefer Conda, you can install all dependencies - after cloning the repository - with `conda env create --name maze -f maze/maze-core-environment.yml`, followed by `pip install -e maze` (this configures the included CLI entrypoints).

### Improving the Documentation

Maze uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate its documentation. You can find the hosted documentation on [ReadTheDocs](https://maze-rl.readthedocs.io/en/latest/). The source code (as `.rst` files) is available [here](https://github.com/enlite-ai/maze/tree/main/docs/source). 

To build and check the documentation locally (make sure you activated your environment with installed Sphinx beforehand) run `sphinx-build -M html source build -W --keep-going -j auto`.

Documentation changes are subject to the same [pull request process](#submitting-a-pull-request) as code changes. 

### Improving the Code

If you are interested in improving Maze' code base, but don't have a specific feature or problem in mind, check out the [open issues](https://github.com/enlite-ai/maze/issues). Select one that suits you and comment on it. We'll respond to you as soon as possible. We'll gladly provide more information and context if needed.

If you want to implement a specific improvement not covered by any of the open issues, write one following the provided issue template. Expect that there will be a discussion and consolidation phase before we can sign off on your proposal, as we have to evaluate how it fits into the existing framework and paradigms.  

Please create an issue _before_ you start implementing - the subsequent discussion might influence your implementation signficantly. This goes for both features and bugfixes.

### Submitting a Pull Request

If you are not familiar with pull requests: Many helpful resources are available, e.g. on [StackOverflow](https://stackoverflow.com/a/14681796/10681301) and the [GitHub documentation](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). 

Generally, an issue should be created before a pull request is opened. This helps to make your work visibile, avoid duplicates and helps us to support you in the process of implementing the feature or bugfix. A pull request should reference the issue it is resolving. The issue will be close once the corresponding pull request is closed.

Once you create a pull request, we will assign a reviewer to you. This reviewer will actively consult with you on your PR and whether and which changes are necessary before merging it. 

We recommend to break proposed changes into small, single-purpose pull requests. This simplifies handling and reviewing. 

All changes must ensure that (a) the existing test suite does not regress, i.e. that the existing functionality is not broken, and (b) add at least one test checking that the included changes work as intended.  

We utilize [`pytest` for testing](https://docs.pytest.org/en/latest/). The entire test can be run with `cd maze && pytest test`. Note that some IDEs (e.g. PyCharm) have built-in support for `pytest`. 

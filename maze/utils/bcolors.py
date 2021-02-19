""" This file contains general utility functionality. """


class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def format_colored(string: str, color: str) -> str:
        """ Format color of string.

        :param string: the text to print
        :param color: the bash color to use
        :return: the color enriched text
        """
        return color + string + BColors.ENDC

    @staticmethod
    def print_colored(string: str, color: str) -> None:
        """ Print color formatted string.

        :param string: the text to print
        :param color: the bash color to use
        """
        print(color + string + BColors.ENDC)

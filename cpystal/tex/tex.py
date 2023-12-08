"""`tex`: for writing tex files.
"""
from __future__ import annotations
from collections import defaultdict
import os
import re

from TexSoup import TexSoup, TexNode

class TexFile(object):
    """Class of TeX file.
    """
    def __init__(self, filename: str) -> None:
        self.filename: str = filename
        self.dirname: str = os.path.dirname(filename) + "/"
        self.basename: str = os.path.basename(filename)
        with open(filename, mode="r") as f:
            self.contents: list[str] = f.readlines()
        self.properties: dict[str, list[str]] = defaultdict(list)
        self.arguments: dict[str, list[str]] = defaultdict(list)
        self.children_files: list[TexFile] = []
        self._analyze()
        text: str = "".join(self.contents)
        text = re.sub(r"(\\right[\)\}\]])([^\s])", r"\1 \2", text)
        # TexSoup has bugs that "\right)" (or any closing parenthesis) must be followed by a space.
        self.soup: TexNode = TexSoup(text)
    
    def __str__(self) -> str:
        return self.basename
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.filename}')"
    
    def _analyze(self) -> None:
        def _find_property(line: str, property_name: str) -> bool:
            if line.startswith(fr"\{property_name}"):
                self.properties[property_name].append(re.sub(fr"\\{property_name}(\[(.*)\])?\{{(.*)\}}.*", r"\3", line))
                self.arguments[property_name].append(re.sub(fr"\\{property_name}(\[(.*)\])?\{{(.*)\}}.*", r"\2", line))
                return True
            else:
                return False

        def _find_statement(line: str, property_name: str) -> bool:
            if line.startswith(fr"\{property_name}"):
                self.properties[property_name].append(re.sub(fr"\\{property_name}\{{(.*)\}}.*", r"\1", line))
                return True
            else:
                return False

        for line in self.contents:
            line = re.sub(r"%.+", "", line) # remove comments
            line = line.lstrip().rstrip()

            # you should edit this part depending on target journal.
            _find_property(line, "title")
            _find_property(line, "author")
            _find_property(line, "date")
            _find_property(line, "usepackage")
            _find_property(line, "input")
            
        for basename in self.properties["input"]:
            filename: str = self.dirname + basename
            self.children_files.append(TexFile(filename))

    @staticmethod
    def _extract_main_text(
            soup: TexNode,
            number_of_properties: defaultdict[str, int] = None,
            is_effective: bool = True
        ) -> tuple[list[str], defaultdict[str, int]]:
        counting_properties: set[str] = set(
            [
                "cite",
                "align",
                "$",
                "figure",
            ]
        )
        target_properties: set[str] = set(
            [
                "document",
                "caption",
                "section",
                "subsection",
                "subsubsection",
                "tabular",
            ]
        )
        if number_of_properties is None:
            number_of_properties: defaultdict[str, int] = defaultdict(int)
        res: list[str] = []
        for text in soup.contents:
            if isinstance(text, str) and is_effective: # raw text
                text = re.sub(r"%.+", "", text)
                if len(text) == 0:
                    continue
                res.append(text)
            elif isinstance(text, TexNode):
                # print(text.name)
                if text.name in target_properties: 
                    texts, n_properties = TexFile._extract_main_text(text)
                    res.extend(texts)
                    for key, value in n_properties.items():
                        number_of_properties[key] += value
                elif text.name == "$" and is_effective: # $a+b$ is counted as 1 word.
                    res.append("$")
                elif text.name == "SI" and is_effective: # \SI{10}{meV} is counted as 1 word.
                    res.append("SI")
                else:
                    texts, n_properties = TexFile._extract_main_text(text, is_effective=False)
                    res.extend(texts)
                    for key, value in n_properties.items():
                        number_of_properties[key] += value
                if text.name in counting_properties:
                    number_of_properties[text.name] += 1
            else:
                pass
        return res, number_of_properties

    def word_count(self) -> int:
        main_text, number_of_properties = self._extract_main_text(self.soup)
        divided_text: str = " ".join([re.sub(r"\n|,|\(|\)|\\|&", " ", s) for s in main_text]).split()
        # print(divided_text)
        return len(divided_text)

    def word_count_all(self) -> int:
        """Count the number of word in text.

        Note:
            Excludes abstract, acknowledgments and supplementary.
            Physical Review Journals: https://journals.aps.org/authors/length-guide
        
        Returns:
            (int): Number of words.
        """
        res: int = 0
        res += self.word_count()
        for child in self.children_files:
            if "abstract" in child.basename or "acknowledgments" in child.basename or "supplementary" in child.basename:
                continue
            res += child.word_count_all()
        return res


def print_all_citation(filename: str) -> None:
    """Print all citation in a tex file.

    Args:
        filename (str): Filename of tex file.
    """
    with open(filename, mode="r") as f:
        text: str = f.read()

    print(*re.findall(r"\\cite(?:\[.+\])?\{[^\}]+\}", text), sep="\n")
    return 

def get_all_citation(filenames: str | list[str]) -> list[str]:
    """Get all citations.

    Args:
        filenames (str | list[str]): Filename or list of filenames.

    Returns:
        list[str]: List of cited documents in order of appearance.
    """
    now_number: int = 1
    reference_names: dict[str, int] = {}
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        with open(filename, mode="r") as f:
            text: str = f.read()
        for s in re.findall(r"\\cite\{[^\}]+\}", text):
            for ref in re.sub(r"\\cite\{(.+)\}", r"\1", s).replace(r" ","").split(","):
                if not ref in reference_names:
                    reference_names[ref] = now_number
                    now_number += 1
    res: list[str] = sorted(list(reference_names.keys()), key=lambda x:reference_names[x])
    # print(*list(enumerate(res,start=1)), sep="\n")
    return res

def contract_numbers(nums: list[int]) -> str:
    """Contract citation numbers.

    Args:
        nums (list[int]): Integers. In general, not sorted.

    Returns:
        str: Contracted numbers.
    
    Examples:
        `contract_numbers([4,7,1,9,8,2])`
        >>> "1-2, 4, 7-9"
    """
    nums = sorted(nums)
    if len(nums) == 0:
        return ""
    res: list[str] = []
    start: int = nums[0]
    now: int = nums[0]
    for i in range(1, len(nums)):
        if nums[i] - now == 1:
            pass
        else:
            if start == now:
                res.append(str(now))
            else:
                res.append(f"{start}-{now}")
            start = nums[i]
        now = nums[i]
    if start == now:
        res.append(str(now))
    else:
        res.append(f"{start}-{now}")
    return ", ".join(res)
    
def reference_numbers_used_in_supplementary(main_filename: str, supple_filename: str) -> str:
    """Return a string that represents list of reference numbers used in the supplementary file.

    Args:
        main_filename (str): Name of the main tex file.
        supple_filename (str): Name of the supplementary tex file.

    Returns:
        str: String of list of cited documents in order of appearance.
    """
    references: list[int] = get_all_citation([main_filename, supple_filename])
    reference_names: dict[str] = {v:k for k, v in enumerate(references, start=1)}

    nums: set[int] = set()
    with open(supple_filename, mode="r") as f:
        supple_text: str = f.read()
    for s in re.findall(r"\\cite\{[^\}]+\}", supple_text):
        for ref in re.sub(r"\\cite\{(.+)\}", r"\1", s).split(","):
            nums.add(reference_names[ref])
    return contract_numbers(list(nums))

def escape_regex(text: str) -> str:
    """Escape the special characters in the plain text (for using pattern of regular expression).

    Args:
        text (str): Text.

    Returns:
        str: Escaped text.
    """
    special_chars: str = r"[\.\^\$\*\+\?\{\}\[\]\\\|\(\)]"
    escaped_text: str = re.sub(special_chars, r"\\\g<0>", text)
    return escaped_text
    
def replace_supplemental_cite_with_plain_number(main_filename: str, supple_filename: str) -> None:
    """Replace the '\cite' command in the supplementary file with the plain citation number in the main file.

    Note:
        All citations that exist only in the supplementary file must be cited in main file with '\nocite' command.
        You can use the function `print_all_citation` to confirm citations in the supplementary file.

    Args:
        main_filename (str): Name of the main tex file.
        supple_filename (str): Name of the supplementary tex file.
    """
    references: list[int] = get_all_citation([main_filename, supple_filename])
    reference_names: dict[str] = {v:k for k, v in enumerate(references, start=1)}

    with open(supple_filename, mode="r") as f:
        supple_text: str = f.read()
    new_text: str = supple_text
    for s in re.findall(r"\\cite\{[^\}]+\}", supple_text):
        nums: list[int] = []
        for ref in re.sub(r"\\cite\{(.+)\}", r"\1", s).split(","):
            nums.append(reference_names[ref])
        pattern: str = escape_regex(s)
        new_text = re.sub(pattern, f"[{contract_numbers(nums)}]", new_text)

    new_supple_filename: str = os.path.dirname(supple_filename) + "/new_" + os.path.basename(supple_filename)
    with open(new_supple_filename, mode="w") as f:
        f.write(new_text)
    return 

def eliminate_nouse_references_from_bib(bib_filename: str, filenames: list[str]) -> None:
    """Eliminate references which is not used in the files from the bib file.

    Note:
        The bib file must not contain the character '@' except that of field statements.

    Args:
        filenames (list[str]): List of filenames of tex files.
        bib_filename (str): Filename of bib file.
    """
    references: list[int] = get_all_citation(filenames)
    reference_names: dict[str] = {v:k for k, v in enumerate(references, start=1)}
    with open(bib_filename, mode="r") as f:
        bib_text: str = f.read()

    new_text: str = bib_text
    depth: int = 0
    field_start: int = 0
    bracket_start: int = 0
    for i, c in enumerate(bib_text):
        if c == "@": # unnecessary '@' causes an error. note that mail address.
            field_start = i
        elif c == "{":
            depth += 1
            if depth == 1:
                bracket_start = i+1
        elif c == "}":
            depth -= 1
            if depth == 0:
                ref_name: str = bib_text[bracket_start:i].split(",")[0]
                if not ref_name in reference_names:
                    new_text = new_text.replace(bib_text[field_start:i+1], "")
    new_text = re.sub(r"\n\n\n+", r"\n\n", new_text)
    new_bib_filename: str = os.path.dirname(bib_filename) + "/new_" + os.path.basename(bib_filename)
    with open(new_bib_filename, mode="w") as f:
        f.write(new_text)
    return 


def estimate_words_displayed_math_PRL(lines: int, occupy_two_column: bool = False) -> int:
    """Estimate equivalent word size of a displayed math in PRL.

    Note:
        Assume that texts are written in two-column mode.

    Args:
        lines (int): Number of lines.
        occupy_two_column (bool): Whether the figure occupies two columns in two-column mode. Defaults to False.

    Returns:
        int: Equivalent word size.
    """
    factor: int = 2 if occupy_two_column else 1
    return 16 * lines * factor

def estimate_words_figure_PRL(aspect: float, occupy_two_column: bool = False) -> int:
    """Estimate equivalent word size of a figure in PRL.

    Note:
        Assume that texts are written in two-column mode.

    Args:
        aspect (float): Aspect ratio (width / height).
        occupy_two_column (bool): Whether the figure occupies two columns in two-column mode. Defaults to False.

    Returns:
        int: Equivalent word size.
    """
    factor: int = 2 if occupy_two_column else 1
    return int(150/aspect + 20) * factor

def estimate_words_table_PRL(lines: int, occupy_two_column: bool = False) -> int:
    """Estimate equivalent word size of a table in PRL.

    Note:
        Assume that texts are written in two-column mode.

    Args:
        lines (int): Number of lines.
        occupy_two_column (bool): Whether the figure occupies two columns in two-column mode. Defaults to False.

    Returns:
        int: Equivalent word size.
    """
    if lines == 0:
        return 0
    factor: int = 2 if occupy_two_column else 1
    return 13 + int(6.5 * lines) * factor

def count_word_size_PRL(
        filename: str,
        lines_of_displayed_maths: list[int] | None = None,
        aspect_ratios_of_figures: list[float] | None = None,
        lines_of_tables: list[int] | None = None,
    ) -> int:
    """Count the total word size of tex file for Physical Review Letters.

    Args:
        filename (str): Filename of tex file.
        lines_of_displayed_maths (list[int] | None, optional): List of the number of line of displayed maths. Defaults to None.
        aspect_ratios_of_figures (list[float] | None, optional): List of the aspect ratio of figures. Defaults to None.
        lines_of_tables (list[int] | None, optional): List of the number of line of tables. Defaults to None.

    Returns:
        int: Total word size.
    """
    displayed_maths: int = 0
    figures: int = 0
    tables: int = 0
    text: int = TexFile(filename).word_count_all()
    if lines_of_displayed_maths is not None:
        displayed_maths = sum([estimate_words_displayed_math_PRL(l) for l in lines_of_displayed_maths])
    if aspect_ratios_of_figures is not None:
        figures: int = sum([estimate_words_figure_PRL(r) for r in aspect_ratios_of_figures])
    if lines_of_tables is not None:
        tables: int = sum([estimate_words_table_PRL(l) for l in lines_of_tables])
    return text + displayed_maths + figures + tables


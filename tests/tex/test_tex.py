from __future__ import annotations

from cpystal.tex.tex import (
    TexFile,
    print_all_citation,
    get_all_citation,
    contract_numbers,
    reference_numbers_used_in_supplementary,
    escape_regex,
    replace_supplemental_cite_with_plain_number,
    eliminate_nouse_references_from_bib,
    count_word_size_PRL,
)

def test_tex_001():
    main_filename: str = "./main.tex"
    supple_filename: str = "./supplementary.tex"
    bib_filename: str = "./thesis.bib"

    print(count_word_size_PRL(main_filename, [1,1], [1.0, 1.0, 1.0, 1.5], []))

    print_all_citation(supple_filename)
    print(reference_numbers_used_in_supplementary(main_filename,supple_filename))

    replace_supplemental_cite_with_plain_number([main_filename,supple_filename])
    eliminate_nouse_references_from_bib(bib_filename,[main_filename,supple_filename])


    
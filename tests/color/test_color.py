from __future__ import annotations

from cpystal.color.color import (Color, RGB, RED_RGB, GREEN_RGB, BLUE_RGB,
                                 CYAN_RGB, MAGENTA_RGB, YELLOW_RGB,
                                 WHITE_RGB, BLACK_RGB, GRAY_RGB)

def test_color_001():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)

    assert r.color == (1.0, 0.0, 0.0)
    assert g.color == (0.0, 1.0, 0.0)
    assert b.color == (0.0, 0.0, 1.0)
    assert r.color_code() == "#ff0000"
    assert r.color_system == "RGB"
    assert r.get_properties() == ((1.0, 0.0, 0.0), "RGB", "D65")

def test_operation_001():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)

    assert -r == c
    assert -g == m
    assert -b == y
    assert r == -c
    assert g == -m
    assert b == -y
    assert -black == black
    assert -white == white

def test_operation_002():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)

    assert r + g == g + r == y
    assert b + r == r + b == m
    assert g + b == b + g == c
    assert c + m == m + c == white
    assert m + y == y + m == white
    assert y + c == c + y == white
    assert r + g + b == white
    assert c + m + y == white
    assert black + white == white
    assert r + black == r
    assert g + black == g
    assert b + black == b
    assert c + black == c
    assert m + black == m
    assert y + black == y
    assert r + white == g + white == b + white == white
    assert c + white == m + white == y + white == white

    assert r - g != r + -g
    assert r + g - g != r - g + g
    assert r - r == black

def test_operation_003():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)
    gray: Color = Color(GRAY_RGB, RGB)
    maroon: Color = Color((0.5, 0.0, 0.0), RGB)
    green: Color = Color((0.0, 0.5, 0.0), RGB)
    navy: Color = Color((0.0, 0.0, 0.5), RGB)
    olive: Color = Color((0.5, 0.5, 0.0), RGB)
    purple: Color = Color((0.5, 0.0, 0.5), RGB)
    teal: Color = Color((0.0, 0.5, 0.5), RGB)

    assert r * g == g * r == olive
    assert g * b == b * g == teal
    assert b * r == r * b == purple
    assert c * y == Color((0.5, 1.0, 0.5), RGB)
    assert y * m == Color((1.0, 0.5, 0.5), RGB)
    assert m * c == Color((0.5, 0.5, 1.0), RGB)
    assert black * white == gray
    assert (r * g) * b == r * (g * b) == Color((1/3, 1/3, 1/3), RGB)
    assert c * m * y == Color((2/3, 2/3, 2/3), RGB)

    assert r * g / g == r


def test_operation_004():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)
    gray: Color = Color(GRAY_RGB, RGB)
    maroon: Color = Color((0.5, 0.0, 0.0), RGB)
    green: Color = Color((0.0, 0.5, 0.0), RGB)
    navy: Color = Color((0.0, 0.0, 0.5), RGB)
    olive: Color = Color((0.5, 0.5, 0.0), RGB)
    purple: Color = Color((0.5, 0.0, 0.5), RGB)
    teal: Color = Color((0.0, 0.5, 0.5), RGB)

    assert r @ g == r @ b == maroon != g @ r
    assert g @ b == g @ r == green != b @ g
    assert b @ r == b @ g == navy != r @ b
    assert black @ white == black


def test_bit_operation_001():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)

    assert ~r == c
    assert ~g == m
    assert ~b == y
    assert r == ~c
    assert g == ~m
    assert b == ~y
    assert ~black == white
    assert ~white == black

def test_bit_operation_002():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)
    
    assert r | g == g | r == y
    assert b | r == r | b == m
    assert g | b == b | g == c
    assert c | m == m | c == white
    assert m | y == y | m == white
    assert y | c == c | y == white
    assert r | g | b == white
    assert c | m | y == white
    assert black | white == white
    assert r | black == r
    assert g | black == g
    assert b | black == b
    assert c | black == c
    assert m | black == m
    assert y | black == y
    assert r | white == g | white == b | white == white
    assert c | white == m | white == y | white == white

def test_bit_operation_003():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)

    assert r & g == g & r == black
    assert b & r == r & b == black
    assert g & b == b & g == black
    assert c & m == m & c == b
    assert m & y == y & m == r
    assert y & c == c & y == g
    assert r & g & b == black
    assert c & m & y == black
    assert black & white == black
    assert r & white == r
    assert g & white == g
    assert b & white == b
    assert c & white == c
    assert m & white == m
    assert y & white == y
    assert r & black == g & black == b & black == black
    assert c & black == m & black == y & black == black

def test_bit_operation_004():
    r: Color = Color(RED_RGB, RGB)
    g: Color = Color(GREEN_RGB, RGB)
    b: Color = Color(BLUE_RGB, RGB)
    c: Color = Color(CYAN_RGB, RGB)
    m: Color = Color(MAGENTA_RGB, RGB)
    y: Color = Color(YELLOW_RGB, RGB)
    white: Color = Color(WHITE_RGB, RGB)
    black: Color = Color(BLACK_RGB, RGB)

    assert r ^ g == g ^ r == y
    assert b ^ r == r ^ b == m
    assert g ^ b == b ^ g == c
    assert c ^ m == m ^ c == y
    assert m ^ y == y ^ m == c
    assert y ^ c == c ^ y == m
    assert r ^ g ^ b == white
    assert c ^ m ^ y == black
    assert black ^ white == white
    assert r ^ white == c
    assert g ^ white == m
    assert b ^ white == y
    assert c ^ white == r
    assert m ^ white == g
    assert y ^ white == b
    assert r ^ black == r
    assert g ^ black == g
    assert b ^ black == b
    assert c ^ black == c
    assert m ^ black == m
    assert y ^ black == y
    assert r ^ r == b ^ b == g ^ g == black
    assert c ^ c == m ^ m == y ^ y == black




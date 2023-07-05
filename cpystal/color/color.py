"""`color`: for dealing with color space.
"""
from __future__ import annotations

import math
from typing import Any, Iterator, Tuple, Union

import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors # type: ignore
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt

Color_type = Union[Tuple[int,int,int], Tuple[float,float,float]]
class Color:
    """Express color spaces. A part of this class is based on MATLAB and the Python standard module 'colorsys'.

    Note:
        Thereafter, we use the following type alias without notice:
            Color_type := tuple[int,int,int] | tuple[float,float,float].

        RGB = Red, Green, Blue
            R: [0.0, 1.0], transparent -> red.
            G: [0.0, 1.0], transparent -> green.
            B: [0.0, 1.0], transparent -> blue.
            This RGB is supposed to be linearized.
        HSV = Hue(color phase), Saturation(colorfulness), Value(brightness)
            H: [0.0, 1.0], red -> yellow -> green -> blue -> magenta -> red.
            S: [0.0, 1.0], medium -> maximum.
            V: [0.0, 1.0], black-strong -> black-weak.
        HLS = Hue(color phase), Luminance(lightness), Saturation(colorfulness)
            H: [0.0, 1.0], red -> yellow -> green -> blue -> magenta -> red.
            L: [0.0, 1.0], black -> white.
            S: [0.0, 1.0], medium -> maximum.
        YIQ = Y(perceived grey level), I(same-phase), Q(anti-phase)
            Y: [0.0, 1.0], black -> white.
            I: [-0.5959, 0.5959], blue -> orange.
            Q: [-0.5229, 0.5229], green -> violet.
        XYZ
            X: [0.0, 0.951]
            Y: [0.0, 1.0]
            Z: [0.0, 1.090]
        L*a*b*
            L*: [0, 100]
            a*: [-128, 127]
            b*: [-128, 127]

        Note that `color_system` of the result color of binary operation is always 'RGB'.

    Attributes:
        color (Color_type): Color value.
        color_system (str): Color system (RGB, HSV, HLS, YIQ).


    """
    def __init__(self,
                color: Color_type,
                color_system: str,
                white_point: str = "D65",
                ) -> None:
        self._color: Color_type = color
        self._color_system: str = color_system
        self._white_point: str = white_point
        self._value_range: dict[str, list[tuple[float, float]]] = {
            "RGB": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "HSV": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "HLS": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "sRGB": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "Adobe RGB": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "YIQ": [(0.0, 1.0), (-0.5959, 0.5959), (-0.5229, 0.5229)],
            "XYZ": [(0.0, 0.951), (0.0, 1.0), (0.0, 1.090)],
            "L*a*b*": [(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)],
        }
        self._weight: int = 1

    def __str__(self) -> str:
        return f"{self.color_system}{tuple(round(c, 2) for c in self.color)}"
    
    def __repr__(self) -> str:
        return f"Color({repr(self._color)}, {repr(self.color_system)}, white_point={repr(self.white_point)})"

    def __neg__(self) -> Color:
        new_color: Color = self.__deepcopy__()
        if new_color.color_system == "RGB":
            r,g,b = new_color._color
            m: float = max(r,g,b) + min(r,g,b)
            new_color._color = (m-r, m-g, m-b)
            return new_color
        else:
            new_color = (-new_color.to_rgb()).to_other_system(new_color.color_system)
            return new_color
    
    def __invert__(self) -> Color:
        new_color: Color = self.__deepcopy__()
        if new_color.color_system == "RGB":
            r,g,b = new_color._color
            new_color._color = (1-r, 1-g, 1-b)
            return new_color
        else:
            new_color = (~new_color.to_rgb()).to_other_system(new_color.color_system)
            return new_color

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Color):
            return self.get_properties() == other.get_properties()
        else:
            return False
        
    def __neq__(self, other: Any) -> bool:
        return not self.__eq__(other)
    
    def __add__(self, other: Color) -> Color:
        """Simple addition operationn between `Color` instances.

        Args:
            other (Color): `Color` instance.

        Returns:
            Color: Result of the operation.
        
        Examples:
            >>> r: Color = Color(RED_RGB, RGB)
            >>> g: Color = Color(GREEN_RGB, RGB)
            >>> b: Color = Color(BLUE_RGB, RGB)
            >>> c: Color = Color(CYAN_RGB, RGB)
            >>> m: Color = Color(MAGENTA_RGB, RGB)
            >>> y: Color = Color(YELLOW_RGB, RGB)
            >>> white: Color = Color(WHITE_RGB, RGB)
            >>> black: Color = Color(BLACK_RGB, RGB)

            >>> assert r + g == g + r == y
            >>> assert b + r == r + b == m
            >>> assert g + b == b + g == c
            >>> assert c + m == m + c == white
            >>> assert m + y == y + m == white
            >>> assert y + c == c + y == white
            >>> assert r + g + b == white
            >>> assert c + m + y == white
            >>> assert black + white == white
            >>> assert r + black == r
            >>> assert g + black == g
            >>> assert b + black == b
            >>> assert c + black == c
            >>> assert m + black == m
            >>> assert y + black == y
            >>> assert r + white == g + white == b + white == white
            >>> assert c + white == m + white == y + white == white
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        (_, maxr),( _, maxg), (_, maxb) = self._value_range["RGB"]
        return self.__class__(
            color=(min(a+x, maxr), min(b+y, maxg), min(c+z, maxb)),
            color_system="RGB",
        )

    def __sub__(self, other: Color) -> Color:
        """Inverse simple addition operationn between `Color` instances.

        Args:
            other (Color): `Color` instance.

        Returns:
            Color: Result of the operation.
        
        Examples:
            >>> r: Color = Color(RED_RGB, RGB)
            >>> g: Color = Color(GREEN_RGB, RGB)
            >>> b: Color = Color(BLUE_RGB, RGB)
            >>> c: Color = Color(CYAN_RGB, RGB)
            >>> m: Color = Color(MAGENTA_RGB, RGB)
            >>> y: Color = Color(YELLOW_RGB, RGB)
            >>> white: Color = Color(WHITE_RGB, RGB)
            >>> black: Color = Color(BLACK_RGB, RGB)

            >>> assert r - g != r + -g
            >>> assert r + g - g != r - g + g
            >>> assert r - r == black
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        (minr, _), (ming, _), (minb, _) = self._value_range["RGB"]
        return self.__class__(
            color=(max(a-x, minr), max(b-y, ming), max(c-z, minb)),
            color_system="RGB",
        )
    
    @staticmethod
    def _weighted_average(x1: float, x2: float, w1: float, w2: float) -> float:
        return (w1*x1 + w2*x2) / (w1 + w2)
    
    def __mul__(self, other: Color) -> Color:
        """Weighted average operationn between `Color` instances.

        Args:
            other (Color): `Color` instance.

        Returns:
            Color: Result of the operation.
        
        Examples:
            >>> r: Color = Color(RED_RGB, RGB)
            >>> g: Color = Color(GREEN_RGB, RGB)
            >>> b: Color = Color(BLUE_RGB, RGB)
            >>> c: Color = Color(CYAN_RGB, RGB)
            >>> m: Color = Color(MAGENTA_RGB, RGB)
            >>> y: Color = Color(YELLOW_RGB, RGB)
            >>> white: Color = Color(WHITE_RGB, RGB)
            >>> black: Color = Color(BLACK_RGB, RGB)
            >>> gray: Color = Color(GRAY_RGB, RGB)
            >>> olive: Color = Color((0.5, 0.5, 0.0), RGB)
            >>> purple: Color = Color((0.5, 0.0, 0.5), RGB)
            >>> teal: Color = Color((0.0, 0.5, 0.5), RGB)

            >>> assert r * g == g * r == olive
            >>> assert g * b == b * g == teal
            >>> assert b * r == r * b == purple
            >>> assert c * y == Color((0.5, 1.0, 0.5), RGB)
            >>> assert y * m == Color((1.0, 0.5, 0.5), RGB)
            >>> assert m * c == Color((0.5, 0.5, 1.0), RGB)
            >>> assert black * white == gray
            >>> assert (r * g) * b == r * (g * b) == Color((1/3, 1/3, 1/3), RGB)
            >>> assert c * m * y == Color((2/3, 2/3, 2/3), RGB)
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        n, m = self._weight, other._weight
        p: float = self._weighted_average(a,x,n,m)
        q: float = self._weighted_average(b,y,n,m)
        r: float = self._weighted_average(c,z,n,m)
        new_color: Color = self.__class__(
            color=(p, q, r),
            color_system="RGB",
        )
        new_color._weight = n + m
        return new_color
    
    def __truediv__(self, other: Color) -> Color:
        """Anti-weighted average.

        Note:
            `ZeroDivisionError` is called if self._weight == other._weight.
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        n, m = self._weight, other._weight
        p: float = self._weighted_average(a,x,n,-m)
        q: float = self._weighted_average(b,y,n,-m)
        r: float = self._weighted_average(c,z,n,-m)
        new_color: Color = self.__class__(
            color=(p, q, r),
            color_system="RGB",
        )
        new_color._weight = n - m
        return new_color
    
    def __matmul__(self, other: Color) -> Color:
        """Multiplicative color change.

        ref: A. Kitaoka, Journal of Color Science Association of Japan, 35(3), 234-236, (2011).
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        t: float = 0.5
        return self.__class__(
            color=((t+(1-t)*x)*a, (t+(1-t)*y)*b, (t+(1-t)*z)*c),
            color_system="RGB",
        )
    
    def __xor__(self, other: Color) -> Color:
        """XOR-like operationn between `Color` instances.

        Args:
            other (Color): `Color` instance.

        Returns:
            Color: Result of the operation.
        
        Examples:
            >>> r: Color = Color(RED_RGB, RGB)
            >>> g: Color = Color(GREEN_RGB, RGB)
            >>> b: Color = Color(BLUE_RGB, RGB)
            >>> c: Color = Color(CYAN_RGB, RGB)
            >>> m: Color = Color(MAGENTA_RGB, RGB)
            >>> y: Color = Color(YELLOW_RGB, RGB)
            >>> white: Color = Color(WHITE_RGB, RGB)
            >>> black: Color = Color(BLACK_RGB, RGB)

            >>> assert r ^ g == g ^ r == y
            >>> assert b ^ r == r ^ b == m
            >>> assert g ^ b == b ^ g == c
            >>> assert c ^ m == m ^ c == y
            >>> assert m ^ y == y ^ m == c
            >>> assert y ^ c == c ^ y == m
            >>> assert r ^ g ^ b == white
            >>> assert c ^ m ^ y == black
            >>> assert black ^ white == white
            >>> assert r ^ white == c
            >>> assert g ^ white == m
            >>> assert b ^ white == y
            >>> assert r ^ black == r
            >>> assert g ^ black == g
            >>> assert b ^ black == b
            >>> assert r ^ r == b ^ b == g ^ g == black
            >>> assert c ^ c == m ^ m == y ^ y == black
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        p: float = np.sin(np.arcsin(a) + np.arcsin(x))
        q: float = np.sin(np.arcsin(b) + np.arcsin(y))
        r: float = np.sin(np.arcsin(c) + np.arcsin(z))
        return self.__class__(
            color=(p, q, r),
            color_system="RGB",
        )
    
    def __or__(self, other: Color) -> Color:
        """OR-like operationn between `Color` instances.

        Args:
            other (Color): `Color` instance.

        Returns:
            Color: Result of the operation.
        
        Examples:
            >>> r: Color = Color(RED_RGB, RGB)
            >>> g: Color = Color(GREEN_RGB, RGB)
            >>> b: Color = Color(BLUE_RGB, RGB)
            >>> c: Color = Color(CYAN_RGB, RGB)
            >>> m: Color = Color(MAGENTA_RGB, RGB)
            >>> y: Color = Color(YELLOW_RGB, RGB)
            >>> white: Color = Color(WHITE_RGB, RGB)
            >>> black: Color = Color(BLACK_RGB, RGB)

            >>> assert r | g == g | r == y
            >>> assert b | r == r | b == m
            >>> assert g | b == b | g == c
            >>> assert c | m == m | c == white
            >>> assert m | y == y | m == white
            >>> assert y | c == c | y == white
            >>> assert r | g | b == white
            >>> assert c | m | y == white
            >>> assert black | white == white
            >>> assert r | black == r
            >>> assert g | black == g
            >>> assert b | black == b
            >>> assert c | black == c
            >>> assert m | black == m
            >>> assert y | black == y
            >>> assert r | white == g | white == b | white == white
            >>> assert c | white == m | white == y | white == white
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        p: float = 1 - (1-a) * (1-x)
        q: float = 1 - (1-b) * (1-y)
        r: float = 1 - (1-c) * (1-z)
        return self.__class__(
            color=(p, q, r),
            color_system="RGB",
        )

    def __and__(self, other: Color) -> Color:
        """AND-like operationn between `Color` instances.

        Args:
            other (Color): `Color` instance.

        Returns:
            Color: Result of the operation.
        
        Examples:
            >>> r: Color = Color(RED_RGB, RGB)
            >>> g: Color = Color(GREEN_RGB, RGB)
            >>> b: Color = Color(BLUE_RGB, RGB)
            >>> c: Color = Color(CYAN_RGB, RGB)
            >>> m: Color = Color(MAGENTA_RGB, RGB)
            >>> y: Color = Color(YELLOW_RGB, RGB)
            >>> white: Color = Color(WHITE_RGB, RGB)
            >>> black: Color = Color(BLACK_RGB, RGB)

            >>> assert r & g == g & r == black
            >>> assert b & r == r & b == black
            >>> assert g & b == b & g == black
            >>> assert c & m == m & c == b
            >>> assert m & y == y & m == r
            >>> assert y & c == c & y == g
            >>> assert r & g & b == black
            >>> assert c & m & y == black
            >>> assert black & white == black
            >>> assert r & white == r
            >>> assert g & white == g
            >>> assert b & white == b
            >>> assert c & white == c
            >>> assert m & white == m
            >>> assert y & white == y
            >>> assert r & black == g & black == b & black == black
            >>> assert c & black == m & black == y & black == black
        """
        a,b,c = self.to_rgb()._color
        x,y,z = other.to_rgb()._color
        p: float = 0 if a == x == 0 else a*x / (a+x-a*x)
        q: float = 0 if b == y == 0 else b*y / (b+y-b*y)
        r: float = 0 if c == z == 0 else c*z / (c+z-c*z)
        return self.__class__(
            color=(p, q, r),
            color_system="RGB",
        )
    
    def __len__(self) -> int:
        return len(self._color)

    def __iter__(self) -> Iterator[float | int]:
        yield from self._color
    
    def __getitem__(self, key: Any) -> float:
        return self._color[key]
    
    def __deepcopy__(self) -> Color:
        return self.__class__(
            color=self._color,
            color_system=self.color_system,
            white_point=self.white_point
        )
    
    def _check_color_value(self) -> None:
        res: list[float | int] = list(self._color)
        for i, c in enumerate(self._color):
            minc, maxc = self._value_range[self.color_system][i]
            if c < minc:
                res[i] = minc
            if c > maxc:
                res[i] = maxc
        self._color = (res[0], res[1], res[2])

    @staticmethod
    def _hls_calc(m1: float, m2: float, hue: float) -> float:
        hue = hue % 1.0
        if hue < 1.0/6.0:
            return m1 + (m2-m1)*hue*6.0
        if hue < 0.5:
            return m2
        if hue < 2.0/3.0:
            return m1 + (m2-m1)*(2.0/3.0-hue)*6.0
        return m1
    
    @staticmethod
    def _round_color(color: Color_type) -> Color_type:
        return tuple(round(c, 4) for c in color)

    @property
    def color(self) -> Color_type:
        return self._round_color(self._color)

    @property
    def color_system(self) -> str:
        return self._color_system

    @property
    def white_point(self) -> str:
        return self._white_point
    
    @white_point.setter
    def white_point(self, value: str) -> None:
        """あとで実装する．これを実装するには，各white point間での colorの変換メソッドを実装する必要がある
        途中でwhite pointを変更されるとself.colorが指す色が変わってしまうので管理する．
        """
        raise NotImplementedError("future works")
        if self._white_point != value:
            if self.color_system == "XYZ":
                pass
            if self.color_system == "L*a*b*":
                pass
    
    def deepcopy(self) -> Color:
        return self.__deepcopy__()

    def get_properties(self) -> tuple[Color_type, str, str]:
        """Properties of `Color` instance.

        Returns:
            tuple[Color_type, str, str]: (color, color system, white point)
        """
        return (self.color, self.color_system, self.white_point)
    
    def get_base_info(self) -> tuple[str, str]:
        """Basic information of `Color` instance.

        Returns:
            tuple[str, str]: (color system, white point)
        """
        return (self.color_system, self.white_point)

    def rgb_to_hsv(self) -> Color:
        """RGB -> HSV

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

            The value range of (h,s,v) is:
                h: [0.0, 1.0] (or [0, 360])
                s: [0.0, 1.0] (or [0, MAX_SV])
                v: [0.0, 1.0] (or [0, MAX_SV])

        Returns:
            (Color): Color expressed in HSV.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")
        r,g,b = self._color
        maxc: float = max(r, g, b)
        minc: float = min(r, g, b)
        v: float = maxc
        if maxc == minc:
            return self.__class__((0., 0., v), "HSV")
        s: float = (maxc-minc) / maxc
        rc: float = (maxc-r) / (maxc-minc)
        gc: float = (maxc-g) / (maxc-minc)
        bc: float = (maxc-b) / (maxc-minc)
        h: float
        if r == maxc:
            h = bc - gc
        elif g == maxc:
            h = 2.0 + rc - bc
        else:
            h = 4.0 + gc - rc
        h = (h/6.0) % 1.0
        color: Color_type = (h, s, v)
        new_color: Color = self.__class__(
            color=color,
            color_system="HSV",
        )
        new_color._check_color_value()
        return new_color

    def hsv_to_rgb(self) -> Color:
        """HSV -> RGB

        Note:
            The value range of (h,s,v) is:
                h: [0.0, 1.0] (or [0, 360])
                s: [0.0, 1.0] (or [0, MAX_SV])
                v: [0.0, 1.0] (or [0, MAX_SV])
                
            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "HSV":
            raise ValueError("'color_system' must be 'HSV'")
        h,s,v = self._color
        if s == 0.0:
            return self.__class__((v, v, v), "RGB")
        i: int = int(h*6.0)
        f: float = (h*6.0) - i
        p: float = v * (1.0-s)
        q: float = v * (1.0-s*f)
        t: float = v * (1.0-s*(1.0-f))
        i = i%6
        if i == 0:
            r, g, b = v, t, p
        if i == 1:
            r, g, b = q, v, p
        if i == 2:
            r, g, b = p, v, t
        if i == 3:
            r, g, b = p, q, v
        if i == 4:
            r, g, b = t, p, v
        if i == 5:
            r, g, b = v, p, q
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color._check_color_value()
        return new_color
        
    def rgb_to_hls(self) -> Color:
        """RGB -> HLS

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]
            
            The value range of (h,l,s) is:
                h: [0.0, 1.0] (or [0, 360])
                l: [0.0, 1.0] (or [0, MAX_LS])
                s: [0.0, 1.0] (or [0, MAX_LS])

        Returns:
            (Color): Color expressed in HLS.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")
        r,g,b = self._color
        maxc: float = max(r, g, b)
        minc: float = min(r, g, b)
        l: float = (minc+maxc) / 2.0
        if minc == maxc:
            return self.__class__((0.0, l, 0.0), "HLS")
        s: float
        if l <= 0.5:
            s = (maxc-minc) / (maxc+minc)
        else:
            s = (maxc-minc) / (2.0-maxc-minc)
        rc: float = (maxc-r) / (maxc-minc)
        gc: float = (maxc-g) / (maxc-minc)
        bc: float = (maxc-b) / (maxc-minc)
        h: float
        if r == maxc:
            h = bc - gc
        elif g == maxc:
            h = 2.0 + rc - bc
        else:
            h = 4.0 + gc - rc
        h = (h/6.0) % 1.0
        color: Color_type = (h, l, s)
        new_color: Color = self.__class__(
            color=color,
            color_system="HLS",
        )
        new_color._check_color_value()
        return new_color

    def hls_to_rgb(self) -> Color:
        """HLS -> RGB

        Note:
            The value range of (h,l,s) is:
                h: [0.0, 1.0] (or [0, 360])
                l: [0.0, 1.0] (or [0, MAX_LS])
                s: [0.0, 1.0] (or [0, MAX_LS])
            
            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "HLS":
            raise ValueError("'color_system' must be 'HLS'")
        h,l,s = self._color
        if s == 0.0:
            r = g = b = l
        else:
            m2: float
            if l <= 0.5:
                m2 = l * (1.0+s)
            else:
                m2 = l + s - l*s
            m1: float = 2.0*l - m2
            r, g, b = self._hls_calc(m1, m2, h+1.0/3.0), self._hls_calc(m1, m2, h), self._hls_calc(m1, m2, h-1.0/3.0)
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color._check_color_value()
        return new_color
    
    def rgb_to_yiq(self) -> Color:
        """RGB -> YIQ

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]
            
            The value range of (y,i,q) is:
                y: [0.0, 1.0]
                i: [-0.5959, 0.5959]
                q: [-0.5229, 0.5229].

        Returns:
            (Color): Color expressed in YIQ.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")
        r,g,b = self._color
        y: float = 0.299*r + 0.587*g + 0.114*b
        i: float = 0.596*r - 0.274*g - 0.322*b
        q: float = 0.211*r - 0.523*g + 0.312*b
        color: Color_type = (y, i, q)
        new_color: Color = self.__class__(
            color=color,
            color_system="YIQ",
        )
        new_color._check_color_value()
        return new_color

    def yiq_to_rgb(self) -> Color:
        """YIQ -> RGB

        Note:
            The value range of (y,i,q) is:
                y: [0.0, 1.0]
                i: [-0.5959, 0.5959]
                q: [-0.5229, 0.5229].

            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "YIQ":
            raise ValueError("'color_system' must be 'YIQ'")
        y,i,q = self._color
        r: float = y + 0.956*i + 0.621*q
        g: float = y - 0.273*i - 0.647*q
        b: float = y - 1.104*i + 1.701*q
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color._check_color_value()
        return new_color

    def rgb_to_srgb(self) -> Color:
        """RGB -> sRGB

        Note:
            The value range of (r,g,b) in RGB and sRGB is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

        Returns:
            (Color): Color expressed in sRGB.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")

        def _f(u: float) -> float:
            if u <= 0.0031308:
                return 12.92 * u
            else:
                return 1.055 * u**(1/2.4) - 0.055

        r,g,b = self._color
        color: Color_type = (_f(r), _f(g), _f(b))
        new_color: Color = self.__class__(
            color=color,
            color_system="sRGB",
        )
        new_color._check_color_value()
        return new_color

    def srgb_to_rgb(self) -> Color:
        """sRGB -> RGB

        Note:
            The value range of (r,g,b) in RGB and sRGB is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "sRGB":
            raise ValueError("'color_system' must be 'sRGB'")

        def _f(u: float) -> float:
            if u <= 0.040450:
                return u / 12.92
            else:
                return ((u + 0.055) / 1.055) ** 2.4

        sr,sg,sb = self._color
        color: Color_type = (_f(sr), _f(sg), _f(sb))
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color._check_color_value()
        return new_color

    def rgb_to_adobergb(self) -> Color:
        """RGB -> Adobe RGB

        Note:
            The value range of (r,g,b) in RGB and Adobe RGB is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

        Returns:
            (Color): Color expressed in Adobe RGB.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")

        def _f(u: float) -> float:
            if u <= 0.00174:
                return 32.0 * u
            else:
                return u ** (1/2.2)

        r,g,b = self._color
        color: Color_type = (_f(r), _f(g), _f(b))
        new_color: Color = self.__class__(
            color=color,
            color_system="Adobe RGB",
        )
        new_color._check_color_value()
        return new_color

    def adobergb_to_rgb(self) -> Color:
        """Adobe RGB -> RGB

        Note:
            The value range of (r,g,b) in RGB and Adobe RGB is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]

        Returns:
            (Color): Color expressed in Adobe RGB.
        """
        if self.color_system != "Adobe RGB":
            raise ValueError("'color_system' must be 'Adobe RGB'")

        def _f(u: float) -> float:
            if u <= 0.0556:
                return u / 32.0
            else:
                return u ** 2.2

        ar,ag,ab = self._color
        color: Color_type = (_f(ar), _f(ag), _f(ab))
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color._check_color_value()
        return new_color

    def rgb_to_xyz(self, white_point: str = "D65") -> Color:
        """RGB -> XYZ

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]
                
        Args:
            white point (str):
                'D65': (x,y,z) = (0.95046, 1.0, 1.08906)
                'D50': (x,y,z) = (0.9642, 1.0, 0.8249)

        Returns:
            (Color): Color expressed in XYZ.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")
        r,g,b = self._color
        x: float = 0.412391*r + 0.357584*g + 0.180481*b
        y: float = 0.212639*r + 0.715169*g + 0.072192*b
        z: float = 0.019331*r + 0.119195*g + 0.950532*b
        # x: float = 2.76883*r + 1.75171*g + 1.13014*b
        # y: float = 1.00000*r + 4.59061*g + 0.06007*b
        # z: float = 0.00000*r + 0.05651*g + 5.59417*b
        color: Color_type = (x, y, z)
        new_color: Color = self.__class__(
            color=color,
            color_system="XYZ",
        )
        # new_color._check_color_value()
        return new_color

    def xyz_to_rgb(self) -> Color:
        """XYZ -> RGB

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0]
                g: [0.0, 1.0]
                b: [0.0, 1.0]
            
            D65 white point: (x,y,z) = (0.95046, 1.0, 1.08906)
            D50 white point: (x,y,z) = (0.9642, 1.0, 0.8249)

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "XYZ":
            raise ValueError("'color_system' must be 'XYZ'")
        x,y,z = self._color
        r: float = 3.240970*x - 1.537383*y - 0.498611*z
        g: float = -0.969244*x + 1.875968*y + 0.041555*z
        b: float = 0.055630*x - 0.203977*y + 1.056972*z
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color._check_color_value()
        return new_color

    def _d65_to_d50(self, xyz: Color_type) -> Color_type:
        """D65 -> D50

        Note:
            In CIE1931,
            D65 white point: (x,y,z) = (0.95046, 1.0, 1.08906)
            D50 white point: (x,y,z) = (0.9642, 1.0, 0.8249).
            D65 and D50 represent Planck radiation 
            with correlated color temperatures of ~6500 K and ~5000 K, respectively.
        
        Args:
            xyz (Color_type): Color value in xyz(D65).
        
        Returns:
            (Color_type): Color value in xyz(D50).
        """
        x,y,z = xyz
        xx: float = 1.047886*x + 0.022919*y - 0.050216*z
        yy: float = 0.029582*x + 0.990484*y - 0.017079*z
        zz: float = -0.009252*x + 0.015073*y + 0.751678*z
        return (xx, yy, zz)

    def _d50_to_d65(self, xyz: Color_type) -> Color_type:
        """D65 -> D50

        Note:
            In CIE1931,
            D65 white point: (x,y,z) = (0.95046, 1.0, 1.08906)
            D50 white point: (x,y,z) = (0.9642, 1.0, 0.8249).
            D65 and D50 represent Planck radiation 
            with correlated color temperatures of ~6500 K and ~5000 K, respectively.
        
        Args:
            xyz (Color_type): Color value in xyz(D50).
        
        Returns:
            (Color_type): Color value in xyz(D65).
        """
        x,y,z = xyz
        xx: float = 0.955512*x - 0.023073*y - 0.063309*z
        yy: float = -0.028325*x + 1.009942*y + 0.021055*z
        zz: float = 0.012329*x - 0.020536*y + 1.330714*z
        return (xx, yy, zz)

    def xyz_to_lab(self) -> Color:
        """XYZ -> L*a*b*

         Note:
            The XYZ and RGB color systems basically use the D65 standard light source as the white point,
            but the L*a*b* color system uses the D50.
            In order to adjust for this difference, Bradford transformation is used.

            The value range of (l,a,b) is:
                l: [0.0, 100.0]
                a: may take an absolute value greater than 100
                b: may take an absolute value greater than 100. 

            D65 white point: (x,y,z) = (0.95046, 1.0, 1.08906)
            D50 white point: (x,y,z) = (0.9642, 1.0, 0.8249).
            
        Returns:
            (Color): Color expressed in L*a*b*.
        """
        if self.color_system != "XYZ":
            raise ValueError("'color_system' must be 'XYZ'")

        def _f(u: float) -> float:
            if u > (6/29)**3:
                return u ** (1/3)
            else:
                return ((29/3)**3 * u + 16) / 116
        xw,yw,zw = 0.9642, 1.0, 0.8249 # D50 white point
        x,y,z = self._d65_to_d50(self._color) # Bradford transformation
        fx,fy,fz = _f(x/xw), _f(y/yw), _f(z/zw)
        l: float = 116*fy - 16
        a: float = 500 * (fx-fy)
        b: float = 200 * (fy-fz)
        color: Color_type = (l, a, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="L*a*b*",
        )
        new_color._check_color_value()
        return new_color

    def lab_to_xyz(self) -> Color:
        """L*a*b* -> XYZ

         Note:
            The XYZ and RGB color systems basically use the D65 standard light source as the white point,
            but the L*a*b* color system uses the D50.
            In order to adjust for this difference, Bradford transformation is used.

            The value range of (l,a,b) is:
                l: [0.0, 100.0]
                a: may take an absolute value greater than 100
                b: may take an absolute value greater than 100. 
            
            D65 white point: (x,y,z) = (0.95046, 1.0, 1.08906)
            D50 white point: (x,y,z) = (0.9642, 1.0, 0.8249).
            
        Returns:
            (Color): Color expressed in L*a*b*.
        """

        if self.color_system != "L*a*b*":
            raise ValueError("'color_system' must be 'L*a*b*'")

        def _invf(u: float) -> float:
            if u > 6/29:
                return u ** 3
            else:
                return (3/29)**3 * (116*u - 16)
        xw,yw,zw = 0.9642, 1.0, 0.8249 # D50 white point
        l,a,b = self._color
        fy: float = (l+16) / 116
        fx: float = fy + a/500
        fz: float = fy - b/200
        x: float = _invf(fx) * xw
        y: float = _invf(fy) * yw
        z: float = _invf(fz) * zw
        color: Color_type = self._d50_to_d65((x, y, z)) # inverse Bradford transformation
        new_color: Color = self.__class__(
            color=color,
            color_system="XYZ",
        )
        # new_color._check_color_value()
        return new_color

    def to_rgb(self) -> Color:
        if self.color_system == "RGB":
            return self.__deepcopy__()
        elif self.color_system == "HSV":
            return self.hsv_to_rgb()
        elif self.color_system == "HLS":
            return self.hls_to_rgb()
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb()
        elif self.color_system == "sRGB":
            return self.srgb_to_rgb()
        elif self.color_system == "Adobe RGB":
            return self.adobergb_to_rgb()
        elif self.color_system == "XYZ":
            return self.xyz_to_rgb()
        elif self.color_system == "L*a*b*":
            return self.lab_to_xyz().xyz_to_rgb()
        else:
            raise ValueError
            return
    
    def to_hsv(self) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_hsv()
        elif self.color_system == "HSV":
            return self.__deepcopy__()
        elif self.color_system == "HLS":
            return self.hls_to_rgb().rgb_to_hsv()
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb().rgb_to_hsv()
        elif self.color_system == "sRGB":
            return self.srgb_to_rgb().rgb_to_hsv()
        elif self.color_system == "Adobe RGB":
            return self.adobergb_to_rgb().rgb_to_hsv()
        elif self.color_system == "XYZ":
            return self.xyz_to_rgb().rgb_to_hsv()
        elif self.color_system == "L*a*b*":
            return self.lab_to_xyz().xyz_to_rgb().rgb_to_hsv()
        else:
            raise ValueError
            return
    
    def to_hls(self) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_hls()
        elif self.color_system == "HSV":
            return self.hsv_to_rgb().rgb_to_hls()
        elif self.color_system == "HLS":
            return self.__deepcopy__()
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb().rgb_to_hls()
        elif self.color_system == "sRGB":
            return self.srgb_to_rgb().rgb_to_hls()
        elif self.color_system == "Adobe RGB":
            return self.adobergb_to_rgb().rgb_to_hls()
        elif self.color_system == "XYZ":
            return self.xyz_to_rgb().rgb_to_hls()
        elif self.color_system == "L*a*b*":
            return self.lab_to_xyz().xyz_to_rgb().rgb_to_hls()
        else:
            raise ValueError
            return

    def to_yiq(self) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_yiq()
        elif self.color_system == "HSV":
            return self.hsv_to_rgb().rgb_to_yiq()
        elif self.color_system == "HLS":
            return self.hls_to_rgb().rgb_to_yiq()
        elif self.color_system == "YIQ":
            return self.__deepcopy__()
        elif self.color_system == "sRGB":
            return self.srgb_to_rgb().rgb_to_yiq()
        elif self.color_system == "Adobe RGB":
            return self.adobergb_to_rgb().rgb_to_yiq()
        elif self.color_system == "XYZ":
            return self.xyz_to_rgb().rgb_to_yiq()
        elif self.color_system == "L*a*b*":
            return self.lab_to_xyz().xyz_to_rgb().rgb_to_yiq()
        else:
            raise ValueError
            return
    
    def to_xyz(self) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_xyz()
        elif self.color_system == "HSV":
            return self.hsv_to_rgb().rgb_to_xyz()
        elif self.color_system == "HLS":
            return self.hls_to_rgb().rgb_to_xyz()
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb().rgb_to_xyz()
        elif self.color_system == "sRGB":
            return self.srgb_to_rgb().rgb_to_xyz()
        elif self.color_system == "Adobe RGB":
            return self.adobergb_to_rgb().rgb_to_xyz()
        elif self.color_system == "XYZ":
            return self.__deepcopy__()
        elif self.color_system == "L*a*b*":
            return self.lab_to_xyz().xyz_to_rgb().rgb_to_xyz()
        else:
            raise ValueError
            return
    
    def to_lab(self) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_xyz().xyz_to_lab()
        elif self.color_system == "HSV":
            return self.hsv_to_rgb().rgb_to_xyz().xyz_to_lab()
        elif self.color_system == "HLS":
            return self.hls_to_rgb().rgb_to_xyz().xyz_to_lab()
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb().rgb_to_xyz().xyz_to_lab()
        elif self.color_system == "sRGB":
            return self.srgb_to_rgb().rgb_to_xyz().xyz_to_lab()
        elif self.color_system == "Adobe RGB":
            return self.adobergb_to_rgb().rgb_to_xyz().xyz_to_lab()
        elif self.color_system == "XYZ":
            return self.xyz_to_lab()
        elif self.color_system == "L*a*b*":
            return self.__deepcopy__()
        else:
            raise ValueError
            return
    
    def to_srgb(self) -> Color:
        return self.to_rgb().rgb_to_srgb()
    
    def to_adobergb(self) -> Color:
        return self.to_rgb().rgb_to_adobergb()

    def to_other_system(self, color_system: str) -> Color:
        """Convert the color system.

        Args:
            color_system (str): Color system.

        Returns:
            Color: `Color` instance.
        """
        if color_system == "RGB":
            return self.to_rgb()
        elif color_system == "HSV":
            return self.to_hsv()
        elif color_system == "HLS":
            return self.to_hls()
        elif color_system == "YIQ":
            return self.to_yiq()
        elif color_system == "sRGB":
            return self.to_srgb()
        elif color_system == "Adobe RGB":
            return self.to_adobergb()
        elif color_system == "XYZ":
            return self.to_xyz()
        elif color_system == "L*a*b*":
            return self.to_lab()
        else:
            raise ValueError
            return

    def color_code(self) -> str:
        """Hexadecimal color code

        Returns:
            (str): Color code in hexadecimal notation.
        """
        r,g,b = self.to_rgb()
        return f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"

    def to_rgba(self, alpha: float | None = None) -> tuple[float, float, float, float]:
        """Hexadecimal color code

        Returns:
            (tuple[float, float, float, float]): (r,g,b,a)
        """
        r,g,b = self.to_rgb()
        a: float
        if alpha is None:
            a = 1.0
        else:
            a = alpha
        return (r, g, b, a)

    def projection(self, other_color: Color) -> Color:
        """Project the color onto the other color like a mathematical 3D vector.

        Args:
            other_color (Color): Target of color projection.

        Returns:
            (Color): Projected color. Note that `color_system` of the returned color is always 'RGB'.
        """
        u: npt.NDArray[np.float32] = np.array(self.to_rgb()._color)
        v: npt.NDArray[np.float32] = np.array(other_color.to_rgb()._color)
        if np.linalg.norm(v) == 0.0:
            a, b, c = 0.0, 0.0, 0.0
        else:
            a, b, c = (u @ v) * v / (np.linalg.norm(v) ** 2) # orthographic projection
        return self.__class__(
            color=(float(a), float(b), float(c)),
            color_system="RGB",
        )
    
    def to_grayscale(self) -> str:
        """Get the grayscaled color. 
            The return type is `str` to conform to the specifications of `matplotlib`.

        Note:
            The 'Y' value in 'CIE XYZ' color system is used to convert the color to grayscale.

        Returns:
            (str): String of float value. The value will be in [0.0, 1.0].
        """
        return str(self.to_xyz()._color[1])
    
    def rgb_256(self) -> Color_type:
        """RGB color in 0-255.

        Returns:
            Color_type: `Color` instance.
        """
        r,g,b = self.to_rgb()._color
        return (int(r*255), int(g*255), int(b*255))
    
    def srgb_256(self) -> Color_type:
        """sRGB color in 0-255.

        Returns:
            Color_type: `Color` instance.
        """
        r,g,b = self.to_rgb().rgb_to_srgb()._color
        return (int(r*255), int(g*255), int(b*255))
    
    @classmethod
    def color_temperature(cls, T: float) -> Color:
        """Color temperature.

        Args:
            T (float): Temperature (K).

        Returns:
            Color: Color temperature.

        References:
            https://en.wikipedia.org/wiki/Planckian_locus
            http://yamatyuu.net/other/color/cie1931xyzrgb/xyz.html
        """
        color_temperature_list: list[tuple[int, float, float]] = [
            (1000, 0.6527, 0.3445),
            (1100, 0.6387, 0.3565),
            (1200, 0.6250, 0.3675),
            (1300, 0.6116, 0.3772),
            (1400, 0.5985, 0.3858),
            (1500, 0.5857, 0.3931),
            (1600, 0.5732, 0.3993),
            (1700, 0.5611, 0.4043),
            (1800, 0.5492, 0.4082),
            (1900, 0.5378, 0.4112),
            (2000, 0.5267, 0.4133),
            (3000, 0.4369, 0.4041),
            (4000, 0.3804, 0.3768),
            (5000, 0.3451, 0.3516),
            (6000, 0.3221, 0.3318),
            (7000, 0.3064, 0.3166),
            (8000, 0.2952, 0.3048),
            (9000, 0.2869, 0.2956),
            (10000, 0.2807, 0.2884),
            (20000, 0.2565, 0.2577),
            (30000, 0.2501, 0.2489),
            (40000, 0.2472, 0.2448),
            (50000, 0.2456, 0.2425),
            (60000, 0.2446, 0.2410),
            (70000, 0.2439, 0.2400),
            (80000, 0.2433, 0.2392),
            (90000, 0.2429, 0.2386),
            (100000, 0.2426, 0.2381),
        ]
        def _interp(target: float) -> tuple[float, float]:
            for i in range(len(color_temperature_list)-1):
                t1, x1, y1 = color_temperature_list[i]
                t2, x2, y2 = color_temperature_list[i+1]
                if t1 <= target <= t2:
                    x_target: float = x1 + (target-t1) / (t2-t1) * (x2-x1)
                    y_target: float = y1 + (target-t1) / (t2-t1) * (y2-y1)
                    return (x_target, y_target)
            raise ValueError 
        
        x: float
        y: float

        if 1667 <= T <= 4000:
            x = -0.2661239e9/T**3 - 0.2343589e6/T**2 + 0.8776956e3/T + 0.179910
        elif 4000 < T <= 25000:
            x = -3.0258469e9/T**3 + 2.1070379e6/T**2 + 0.2226347e3/T + 0.240390
        else:
            x, _ = _interp(T)

        if 1667 <= T <= 2222:
            y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
        elif 2222 < T <= 4000:
            y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
        elif 4000 < T <= 25000:
            y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
        else:
            _, y = _interp(T)

        z: float = 1 - x - y
        return Color(
            color=(x, y, z),
            color_system="XYZ"
        )
    
    @property
    def D65(self) -> Color:
        """D65 white point.

        TODO:
            This should be a class property in Python 3.9.
        """
        return self.color_temperature(6500 * 1.4388 / 1.438)
    
    @property
    def D50(self) -> Color:
        """D50 white point.

        TODO:
            This should be a class property in Python 3.9.
        """
        return self.color_temperature(5000 * 1.4388 / 1.438)

    @classmethod
    def from_color_code(cls, code: str) -> Color:
        """Construct a `Color` instance from color code.

        Args:
            name (str): Color code written in '#xxxxxx' format.
                For example, '#FFFFFF' is white.

        Returns:
            Color: `Color` instance.
        """
        r: float = int(f"0x{code[1:3]}", base=16)/255
        g: float = int(f"0x{code[3:5]}", base=16)/255
        b: float = int(f"0x{code[5:7]}", base=16)/255
        return Color((r,g,b), "RGB")

    @classmethod
    def starlight(cls) -> list[Color]:
        """Starlight colors.

        Returns:
            list[Color]: List of `Color`.
        """
        return [
            cls.from_color_code("#fb5458"),
            cls.from_color_code("#6292e9"),
            cls.from_color_code("#62bd93"),
            cls.from_color_code("#fe9952"),
            cls.from_color_code("#cbc6cc"),
            cls.from_color_code("#95caee"),
            cls.from_color_code("#fdd162"),
            cls.from_color_code("#8c67aa"),
            cls.from_color_code("#e08696"),
        ]
    
    @classmethod
    def from_name(cls, name: str) -> Color:
        """Construct a `Color` instance from color name.

        Args:
            name (str): Name of color. See `matplotlib.colors`.

        Returns:
            Color: `Color` instance.
        """
        return cls.from_color_code(mcolors.CSS4_COLORS[name])

    def change_hue_hsv(self, hue: float) -> Color:
        new: Color = self.to_hsv()
        new._color = (hue, new._color[1], new._color[2])
        return new.to_other_system(self.color_system)
    
    def change_saturation_hsv(self, saturation: float) -> Color:
        new: Color = self.to_hsv()
        new._color = (new._color[0], saturation, new._color[2])
        return new.to_other_system(self.color_system)
    
    def change_value_hsv(self, value: float) -> Color:
        new: Color = self.to_hsv()
        new._color = (new._color[0], new._color[1], value)
        return new.to_other_system(self.color_system)
    
    def rotate_hsv(self, degree: int) -> Color:
        h, _, _ = self.to_hsv()
        return self.change_hue_hsv((h + degree/360) % 1.0)

class Gradation:
    """Color gradation

    Attributes:
        start (Color): Start color of the gradation.
        end (Color): End color of the gradation.
        middle (Color | None): Middle color of the gradation. Defaults to None.

    TODO:
        The constructor of Gradation generates Gradation.color_list.
        The methods gradation_*_list is converted to classmethod.
        Make Gradation object iterable (Implement __iter__).

    Examples:
        >>> ### you can use this class as 'matplotlib.cm' objects.
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib.cm as cm
        >>> import numpy as np

        >>> R = Color(RED_RGB, RGB)
        >>> B = Color(BLUE_RGB, RGB)
        >>> Gr = Gradation(R,B)

        >>> x = np.linspace(0,2*np.pi,100)
        >>> for i in range(30):
        >>>     # plt.plot(x,i*np.sin(x), color=cm.hsv(i/30.0))
        >>>     plt.plot(x,i*np.sin(x), color=Gr.gradation_linear(i/30).to_rgba())
        >>> plt.xlim(0,2*np.pi)
        >>> plt.show()
    """
    def __init__(self, start: Color,
                end: Color,
                middle: Color | None = None,
                num: int = 50,
                next_color_system: str | None = None
                ) -> None:
        if middle is None:
            if start.get_base_info() != end.get_base_info():
                raise ValueError("'start' and 'end' must have the same properties")
        else:
            if not (start.get_base_info() == middle.get_base_info() == end.get_base_info()):
                raise ValueError("'start' and 'middle' and 'end' must have the same properties")
        self.start: Color = start
        self.end: Color = end
        self.middle: Color | None = middle
        self.color_list: list[Color] = [
            self._get_linear(i/(num-1),
                             start,
                             end,
                             middle=middle,
                             next_color_system=next_color_system)
            for i in range(num)
        ]
        self.next_color_system: str | None = next_color_system

    def __str__(self) -> str:
        return str(self.color_list)
    
    def __repr__(self) -> str:
        return f"Gradation({repr(self.start)}, {repr(self.end)}, middle={repr(self.middle)}, "\
                f"num={repr(len(self.color_list))}, next_color_system={repr(self.next_color_system)})"

    def __neg__(self) -> Gradation:
        return self.from_color_list([-c for c in self.color_list])
        
    def __invert__(self) -> Gradation:
        return self.from_color_list([~c for c in self.color_list])

    def __len__(self) -> int:
        return len(self.color_list)
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Gradation):
            return len(self) == len(other) and all([c1 == c2 for c1, c2 in zip(self, other)])
        else:
            return False
    
    def __neq__(self, other: Any) -> bool:
        return not self.__eq__(other)
    
    def __add__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 + c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c + other for c in self.color_list])
        else:
            raise ValueError
        
    def __sub__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 - c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c - other for c in self.color_list])
        else:
            raise ValueError
    
    def __mul__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 * c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c * other for c in self.color_list])
        else:
            raise ValueError
        
    def __truediv__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 / c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c / other for c in self.color_list])
        else:
            raise ValueError
        
    def __matmul__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 @ c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c @ other for c in self.color_list])
        else:
            raise ValueError
    
    def __and__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 & c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c & other for c in self.color_list])
        else:
            raise ValueError

    def __or__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 | c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c | other for c in self.color_list])
        else:
            raise ValueError
    
    def __xor__(self, other: Any) -> Gradation:
        if isinstance(other, Gradation):
            if len(self) == len(other):
                return self.from_color_list([c1 ^ c2 for c1, c2 in zip(self, other)])
            else:
                raise ValueError
        elif isinstance(other, Color):
            return self.from_color_list([c ^ other for c in self.color_list])
        else:
            raise ValueError
        
    def __iter__(self) -> Iterator[Color]:
        yield from self.color_list

    def __call__(self, proportion: float) -> Color:
        return self.color_list[int((len(self.color_list)-1) * proportion)]

    @classmethod
    def from_color_list(cls, color_list: list[Color]) -> Gradation:
        new: Gradation = cls(color_list[0].deepcopy(), color_list[-1].deepcopy(), num=len(color_list),)
        new.color_list = [c.deepcopy() for c in color_list]
        return new

    def rgba_color_list(self) -> list[tuple[float,float,float,float]]:
        res: list[tuple[float,float,float,float]] = []
        for color in self.color_list:
            r,g,b = color.to_rgb()
            res.append((r, g, b, 1.0))
        return res

    @staticmethod
    def _internal_division(
        start: Color_type,
        end: Color_type,
        proportion: float,
    ) -> Color_type:
        a,b,c = start
        x,y,z = end
        u: float = a + (x-a) * proportion
        v: float = b + (y-b) * proportion
        w: float = c + (z-c) * proportion
        return (u, v, w)

    @classmethod
    def _get_linear(
            cls,
            proportion: float,
            start: Color,
            end: Color,
            middle: Color | None = None,
            next_color_system: str | None = None,
        ) -> Color:
        """Make a color gradation linearly in a color space.

        Note:
            A gradation can be represented as a curve in a color space.
            Straight geodesic lines are used as the gradation curves in this method,
            assuming that the color space is considered as a real 3D Euclidean space.

        Args:
            proportion (float): Floating point number in [0.0, 1.0].
                proportion = 0.0 -> start color, proportion = 1.0 -> end color.

        Returns:
            (Color): Color corresponding the number 'proportion' in the gradation.
        """
        if not (0.0 <= proportion <= 1.0):
            raise ValueError("'proportion' must be in [0.0, 1.0]")
        if middle is None:
            if start.get_base_info() != end.get_base_info():
                raise ValueError("'start' and 'end' must have the same properties")
        else:
            if not (start.get_base_info() == middle.get_base_info() == end.get_base_info()):
                raise ValueError("'start' and 'middle' and 'end' must have the same properties")

        if next_color_system is None:
            next_color_system = start.color_system
        color_system: str = start.color_system
        if middle is None:
            u,v,w = cls._internal_division(start, end, proportion)
        else:
            if proportion < 0.5:
                u,v,w = cls._internal_division(start, middle, proportion)
            else:
                u,v,w = cls._internal_division(middle, end, proportion-0.5)
        return Color(color=(u,v,w),
            color_system=color_system,
        ).to_other_system(next_color_system)

    @staticmethod
    def _chart_goal(
        start: Color_type,
        end: Color_type,
        proportion: float,
        order: tuple[int, int, int],
        full_dist: float,
    ) -> Color_type:
        i,j,k = order
        res: Color_type = [0., 0., 0.]
        di: float = end[i] - start[i]
        dj: float = end[j] - start[j]
        dk: float = end[k] - start[k]
        if proportion < abs(di) / full_dist:
            res[i] = start[i] + di * proportion
            res[j] = start[j]
            res[k] = start[k]
        elif proportion < (abs(di) + abs(dj)) / full_dist:
            res[i] = end[i]
            res[j] = start[j] + dj * (proportion - abs(di)/full_dist)
            res[k] = start[k]
        else:
            res[i] = end[i]
            res[j] = start[j]
            res[k] = start[k] + dk * (proportion - (abs(di)+abs(dj))/full_dist)
        return tuple(res)
    
    @classmethod
    def _get_chart(
            cls,
            proportion: float,
            start: Color,
            end: Color,
            middle: Color | None = None,
            order: tuple[int, int, int] = (0,1,2),
            next_color_system: str | None = None,
        ) -> Color:
        """Make a color gradation like a chart line in a color space.

        Note:
            A gradation can be represented as a curve in a color space.
            Chart lines along with each color axis are used as the gradation curves in this method,
            assuming that the color space is considered as a real 3D Euclidean space.

        Args:
            proportion (float): Floating point number in [0.0, 1.0].
                proportion = 0.0 -> start color, proportion = 1.0 -> end color.
            order (tuple[int, int, int]): Priority of color axes in selecting the direction of chart.
                Defaults to (0,1,2). In RGB color space, the chart heads "R" direction at first, 
                "G" direction at second and "B" direction at last if `order = (0,1,2)`.

        Returns:
            (Color): Color corresponding the number 'proportion' in the gradation.
        """
        if not (0.0 <= proportion <= 1.0):
            raise ValueError("'proportion' must be in [0.0, 1.0]")
        if middle is None:
            if start.get_base_info() != end.get_base_info():
                raise ValueError("'start' and 'end' must have the same properties")
        else:
            if not (start.get_base_info() == middle.get_base_info() == end.get_base_info()):
                raise ValueError("'start' and 'middle' and 'end' must have the same properties")
        
        if next_color_system is None:
            next_color_system = start.color_system
        color_system: str = start.color_system
        res: list[float]
        full_dist: float
        if middle is None:
            a,b,c = start
            x,y,z = end
            full_dist = abs(a-x) + abs(b-y) + abs(c-z)
            res = cls._chart_goal(start, end, proportion, order, full_dist)
        else:
            a,b,c = start
            p,q,r = middle
            x,y,z = end
            full_dist = abs(a-p) + abs(b-q) + abs(c-r) + abs(p-x) + abs(q-y) + abs(r-z)
            mid_dist: float = abs(a-p) + abs(b-q) + abs(c-r)
            if proportion < mid_dist / full_dist:
                res = cls._chart_goal(start, middle, proportion, order, full_dist)
            else:
                res = cls._chart_goal(middle, end, proportion-mid_dist / full_dist, order, full_dist)
            
        return Color(color=(res[0], res[1], res[2]),
            color_system=color_system,
        ).to_other_system(next_color_system)

    @classmethod
    def _get_helical(
            cls,
            proportion: float,
            start: Color,
            end: Color,
            middle: Color | None = None,
            clockwise: bool = True,
            next_color_system: str | None = None,
        ) -> Color:
        """Make a color gradation helically in a color space.
        This method is mainly used for HSV or HLS.

        Note:
            A gradation can be represented by a curve in a color space.
            Helical curves are used as the gradation curves in this method,
            assuming that a cylindrical coordinates system is defined in the color space.
            
            proportion = 0.0 -> start color, proportion = 1.0 -> end color.

        Args:
            proportion (float): Floating point number in [0.0, 1.0].
            start (Color): Start color.
            end (Color): End color.
            middle (Color, optional): Middle color. Defaults to None.
            clockwise (bool, optional): If True, direction of spiral winding is clockwise. Defaults to True.
            next_color_system (str, optional): Color system of return value. Defaults to None.

        Returns:
            (Color): Color corresponding the number 'proportion' in the gradation.
        """
        if not (0.0 <= proportion <= 1.0):
            raise ValueError("'proportion' must be in [0.0, 1.0]")
        if middle is None:
            if start.get_base_info() != end.get_base_info():
                raise ValueError("'start' and 'end' must have the same properties")
        else:
            if not (start.get_base_info() == middle.get_base_info() == end.get_base_info()):
                raise ValueError("'start' and 'middle' and 'end' must have the same properties")
        
        if next_color_system is None:
            next_color_system = start.color_system
        color_system: str = start.color_system
        if middle is None:
            a,b,c = start
            x,y,z = end
            if clockwise and a > x:
                x += 1.0
            if (not clockwise) and a < x:
                x -= 1.0
            u,v,w = cls._internal_division((a,b,c), (x,y,z), proportion)
        else:
            a,b,c = start
            p,q,r = middle
            x,y,z = end
            if proportion < 0.5:
                if clockwise and a > p:
                    p += 1
                if (not clockwise) and a < p:
                    p -= 1
                u,v,w = cls._internal_division((a,b,c), (p,q,r), proportion)
            else:
                if clockwise and p > x:
                    x += 1
                if (not clockwise) and p < x:
                    x -= 1
                u,v,w = cls._internal_division((p,q,r), (x,y,z), proportion-0.5)
        return Color(color=(u%1.0, v, w),
            color_system=color_system,
        ).to_other_system(next_color_system)
    
    def convert_to_helical(
            self,
            num: int = 50,
            clockwise: bool = True,
            next_color_system: str | None = None
        ) -> None:
        """Make a list of color gradation helically in a color space.
        This method is mainly used for HSV or HLS.

        Note:
            A gradation can be represented by a curve in a color space.
            Helical curves are used as the gradation curves in this method,
            assuming that a cylindrical coordinates system is defined in the color space.

        Args:
            num (int): Length of the return list.
            clockwise (bool): If True, direction of spiral winding is clockwise. Defaults to True.
        """
        self.color_list = [
            self._get_helical(i/(num-1),
                             self.start,
                             self.end,
                             middle=self.middle,
                             clockwise=clockwise,
                             next_color_system=next_color_system)
            for i in range(num)
        ]

    def convert_to_chart(
            self,
            num: int = 50,
            order: tuple[int, int, int] = (0,1,2),
            next_color_system: str | None = None,
        ) -> Gradation:
        """Make a list of color gradation like a chart in a color space.

        Note:
            A gradation can be represented as a curve in a color space.
            Chart lines along with each color axis are used as the gradation curves in this method,
            assuming that the color space is considered as a real 3D Euclidean space.

        Args:
            num (int): Length of the return list.

        Returns:
            (Gradation): Gradation object.
        """
        self.color_list = [
            self._get_chart(i/(num-1),
                             self.start,
                             self.end,
                             middle=self.middle,
                             order=order,
                             next_color_system=next_color_system)
            for i in range(num)
        ]

    @classmethod
    def gradation_linear_list(
            cls,
            start: Color,
            end: Color,
            middle: Color | None = None,
            num: int = 50,
            next_color_system: str | None = None
        ) -> Gradation:
        """Make a list of color gradation linearly in a color space.

        Note:
            A gradation can be represented as a curve in a color space.
            Straight geodesic lines are used as the gradation curves in this method,
            assuming that the color space is considered as a real 3D Euclidean space.

        Args:
            num (int): Length of the return list.

        Returns:
            (Gradation): Gradation object.
        """
        color_list: list[Color] = [
            cls._get_linear(i/(num-1),
                             start,
                             end,
                             middle=middle,
                             next_color_system=next_color_system)
            for i in range(num)
        ]
        return cls.from_color_list(color_list)

    @classmethod
    def gradation_chart_list(
            cls,
            start: Color,
            end: Color,
            middle: Color | None = None,
            num: int = 50,
            order: tuple[int, int, int] = (0,1,2),
            next_color_system: str | None = None,
        ) -> Gradation:
        """Make a list of color gradation like a chart in a color space.

        Note:
            A gradation can be represented as a curve in a color space.
            Chart lines along with each color axis are used as the gradation curves in this method,
            assuming that the color space is considered as a real 3D Euclidean space.

        Args:
            num (int): Length of the return list.

        Returns:
            (Gradation): Gradation object.
        """
        color_list: list[Color] = [
            cls._get_chart(i/(num-1),
                             start,
                             end,
                             middle=middle,
                             order=order,
                             next_color_system=next_color_system)
            for i in range(num)
        ]
        return cls.from_color_list(color_list)
    
    @classmethod
    def gradation_helical_list(
            cls,
            start: Color,
            end: Color,
            middle: Color | None = None,
            num: int = 50,
            clockwise: bool = True,
            next_color_system: str | None = None
        ) -> Gradation:
        """Make a list of color gradation helically in a color space.
        This method is mainly used for HSV or HLS.

        Note:
            A gradation can be represented by a curve in a color space.
            Helical curves are used as the gradation curves in this method,
            assuming that a cylindrical coordinates system is defined in the color space.

        Args:
            num (int): Length of the return list.
            clockwise (bool): If True, direction of spiral winding is clockwise. Defaults to True.

        Returns:
            (Gradation): Gradation object.
        """
        color_list: list[Color] = [
            cls._get_helical(i/(num-1),
                             start,
                             end,
                             middle=middle,
                             clockwise=clockwise,
                             next_color_system=next_color_system)
            for i in range(num)
        ]
        return cls.from_color_list(color_list)
    
    def visualize_gradation(self, rgb_type: str = "RGB") -> None:
        x: npt.NDArray = np.linspace(0, 1, 256).reshape(1, 256)
        fig: plt.figure = plt.figure(figsize=(5, 2))
        ax: plt.subplot = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        cdict: dict[str, list[tuple[float, float, float]]] = {"red":[], "green": [], "blue": []}
        N: int = len(self.color_list)
        for i, c in enumerate(self.color_list):
            r,g,b = c.to_other_system(rgb_type)
            cdict["red"].append((i/(N-1), r, r))
            cdict["green"].append((i/(N-1), g, g))
            cdict["blue"].append((i/(N-1), b, b))
        cmap = matplotlib.colors.LinearSegmentedColormap("custom", cdict, N=N)
        ax.imshow(x, cmap=cmap, aspect="auto")
        plt.show()
    
    def compare_gradations(self, gradation_list: list[Gradation], rgb_type: str = "RGB") -> None:
        x: npt.NDArray = np.linspace(0, 1, 256).reshape(1, 256)
        fig: plt.figure = plt.figure(figsize=(5, 2))
        gradation_list = [self] + gradation_list
        for j, gradation in enumerate(gradation_list, 1):
            ax: plt.subplot = fig.add_subplot(len(gradation_list), 1, j)
            ax.set_axis_off()
            cdict: dict[str, list[tuple[float, float, float]]] = {"red":[], "green": [], "blue": []}
            N: int = len(gradation.color_list)
            for i, c in enumerate(gradation.color_list):
                r,g,b = c.to_other_system(rgb_type)
                cdict["red"].append((i/(N-1), r, r))
                cdict["green"].append((i/(N-1), g, g))
                cdict["blue"].append((i/(N-1), b, b))
            cmap = matplotlib.colors.LinearSegmentedColormap("custom", cdict, N=N)
            ax.imshow(x, cmap=cmap, aspect="auto")
        plt.show()

    @classmethod
    def planckian_curve(cls, lowT: float, highT: float, num: int = 256) -> Gradation:
        return cls.from_color_list([Color.color_temperature(T).to_srgb() for T in np.linspace(lowT,highT,num)])

    @classmethod
    def equidistant(cls, num: int) -> list[Color]:
        if num == 10:
            return cls.ten_colors()
        elif num == 4:
            return cls.four_colors()
        elif num == 7:
            return cls.seven_colors()
        elif num == 8:
            return cls.eight_colors()
        else:
            return cls.gradation_helical_list(Color(BLUE_HSV,"HSV"), Color(RED_HSV,"HSV"), num=num, clockwise=False, next_color_system="sRGB").color_list

    @staticmethod
    def ten_colors() -> list[Color]:
        return [
            Color.from_color_code("#000000"),
            Color.from_color_code("#8c67aa").change_saturation_hsv(0.8),
            Color.from_color_code("#0000ff"),
            Color.from_color_code("#6292e9").change_saturation_hsv(0.8),
            Color.from_color_code("#95caee").change_saturation_hsv(0.5),
            Color.from_color_code("#62bd93").change_saturation_hsv(0.8),
            Color.from_color_code("#adff2f").change_saturation_hsv(0.8),
            Color.from_color_code("#fdd162").change_saturation_hsv(0.8),
            Color.from_color_code("#fe9952").change_saturation_hsv(0.8),
            Color.from_color_code("#fb5458").change_saturation_hsv(0.8),
        ]
    
    @staticmethod
    def four_colors() -> list[Color]:
        return [
            Color.from_color_code("#8c67aa"),
            Color.from_color_code("#62bd93"),
            Color.from_color_code("#fdd162"),
            Color.from_color_code("#fb5458"),
        ]
    
    @staticmethod
    def seven_colors() -> list[Color]:
        return [
            Color.from_color_code("#8c67aa"),
            Color.from_color_code("#6292e9"),
            Color.from_color_code("#62bd93"),
            Color.from_color_code("#95caee"),
            Color.from_color_code("#fdd162"),
            Color.from_color_code("#fe9952"),
            Color.from_color_code("#fb5458"),
            Color.from_color_code("#e08696"),
        ]
    
    @staticmethod
    def eight_colors() -> list[Color]:
        return [
            Color.from_color_code("#8c67aa").change_saturation_hsv(0.8),
            Color.from_color_code("#0000ff"),
            # Color.from_color_code("#6292e9").change_saturation_hsv(0.8),
            Color.from_color_code("#95caee").change_saturation_hsv(0.5),
            Color.from_color_code("#62bd93").change_saturation_hsv(0.8),
            Color.from_color_code("#adff2f").change_saturation_hsv(0.8),
            Color.from_color_code("#fdd162").change_saturation_hsv(0.8),
            Color.from_color_code("#fe9952").change_saturation_hsv(0.8),
            Color.from_color_code("#fb5458").change_saturation_hsv(0.8),
        ]

    @classmethod
    def multi_middle(cls, base_colors: list[Color], approach_mode: list[str]) -> Gradation:
        n: int = 255
        m: int = len(base_colors)
        color_list: list[Color] = []
        for i in range(m-1):
            start: Color = base_colors[i]
            end: Color = base_colors[i+1]
            if i == m-2:
                l = n - i*(n//(m-1))
            else:
                l = n//(m-1)
            for j in range(l):
                if approach_mode[i] == "helical":
                    color_list.append(cls._get_helical(j/l, start, end))
                elif approach_mode[i] == "helical-anticlockwise":
                    color_list.append(cls._get_helical(j/l, start, end, colockwise=False))
                elif approach_mode[i] == "linear":
                    color_list.append(cls._get_linear(j/l, start, end))
                elif approach_mode[i] == "chart":
                    color_list.append(cls._get_chart(j/l, start, end))
        return cls.from_color_list(color_list)

    @classmethod
    def planetearth(cls) -> Gradation:
        base_colors: list[Color] = [
            Color((0,0,0), "HSV"),
            Color((26/360,1,1), "HSV"),
            Color((60/360,1,1), "HSV"),
            Color((120/360,1,1), "HSV"),
            Color((167/360,1,0.71), "HSV"),
            Color((240/360,0,0.4), "HSV"),
            Color((240/360,0,1), "HSV"),
            Color((240/360,1,1), "HSV"),
        ]
        return cls.multi_middle(base_colors, ["helical"]*(len(base_colors)-1))

    def to_matplotlib_colormap(self) -> mcolors.ListedColormap:
        colors: npt.NDArray = np.array(
            [c.to_rgb().color for c in self.color_list]
        )
        return mcolors.ListedColormap(colors)


    
def plot_colortable(colors: list[Color], *, ncols: int = 4, sort_colors: bool = True) -> plt.figure:
    """Plot a table of colors.

    Args:
        colors (list[Color]): List of colors.
        ncols (int, optional): Number of columns. Defaults to 4.
        sort_colors (bool, optional): If True, the output will be sorted by the hue. Defaults to True.

    Returns:
        plt.figure: Figure object.
    """
    cell_width: int = 212
    cell_height: int = 22
    swatch_width: int = 48
    margin: int = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        colors = sorted(
            colors, key=lambda c: c.to_hsv().color)
    else:
        colors = list(colors)

    n: int = len(colors)
    nrows: int = math.ceil(n / ncols)
    width: int = cell_width * 4 + 2 * margin
    height: int = cell_height * nrows + 2 * margin
    dpi: int = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, color in enumerate(colors):
        row: int = i % nrows
        col: int = i // nrows
        y: int = row * cell_height
        swatch_start_x: int = cell_width * col
        text_pos_x: int = cell_width * col + swatch_width + 7
        ax.text(text_pos_x, y, color.color_code(), fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')
        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=color.to_rgb(), edgecolor='0.7')
        )
    return fig

def rgb_to_hsv(rgb: Color_type, digitization: bool = False, MAX_SV: int = 100) -> Color_type:
    """RGB -> HSV

    Note:
        The value range of (r,g,b) is:
            r: [0.0, 1.0]
            g: [0.0, 1.0]
            b: [0.0, 1.0]
            
        
        The value range of (h,s,v) is:
            h: [0.0, 1.0] (or [0, 360])
            s: [0.0, 1.0] (or [0, MAX_SV])
            v: [0.0, 1.0] (or [0, MAX_SV])
            
        
    Args:
        rgb (Color_type): (r,g,b).
    """
    r,g,b = rgb
    if digitization:
        r /= 255
        g /= 255
        b /= 255
    maxc: float = max(r, g, b)
    minc: float = min(r, g, b)
    v: float = maxc
    if maxc == minc:
        if digitization:
            return (0, 0, int(v*MAX_SV))
        else:
            return (0., 0., v)
    s: float = (maxc-minc) / maxc
    rc: float = (maxc-r) / (maxc-minc)
    gc: float = (maxc-g) / (maxc-minc)
    bc: float = (maxc-b) / (maxc-minc)
    h: float
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h/6.0) % 1.0
    if digitization:
        return (int(h*360), int(s*MAX_SV), int(v*MAX_SV))
    else:
        return (h, s, v)

def hsv_to_rgb(hsv: Color_type, digitization: bool = False, MAX_SV: int = 100) -> Color_type:
    """HSV -> RGB

    Note:
        The value range of (h,s,v) is:
            h: [0.0, 1.0] (or [0, 360])
            s: [0.0, 1.0] (or [0, MAX_SV])
            v: [0.0, 1.0] (or [0, MAX_SV])
            
        
        The value range of (r,g,b) is:
            r: [0.0, 1.0]
            g: [0.0, 1.0]
            b: [0.0, 1.0]
            
        
    Args:
        hsv (Color_type): (h,s,v).
    """
    h,s,v = hsv
    if digitization:
        h /= 360
        s /= MAX_SV
        v /= MAX_SV
    if s == 0.0:
        return (v, v, v)
    i: int = int(h*6.0)
    f: float = (h*6.0) - i
    p: float = v * (1.0-s)
    q: float = v * (1.0-s*f)
    t: float = v * (1.0-s*(1.0-f))
    i = i%6
    if i == 0:
        r, g, b = v, t, p
    if i == 1:
        r, g, b = q, v, p
    if i == 2:
        r, g, b = p, v, t
    if i == 3:
        r, g, b = p, q, v
    if i == 4:
        r, g, b = t, p, v
    if i == 5:
        r, g, b = v, p, q
    if digitization:
        return (r*255, g*255, b*255)
    else:
        return (r, g, b)

def rgb_to_hls(rgb: Color_type, digitization: bool = False, MAX_LS: int = 100) -> Color_type:
    """RGB -> HLS

    Note:
        The value range of (r,g,b) is:
            r: [0.0, 1.0]
            g: [0.0, 1.0]
            b: [0.0, 1.0]
            
        
        The value range of (h,l,s) is:
            h: [0.0, 1.0] (or [0, 360])
            l: [0.0, 1.0] (or [0, MAX_LS])
            s: [0.0, 1.0] (or [0, MAX_LS])
            
        
    Args:
        rgb (Color_type): (r,g,b).
    """
    r,g,b = rgb
    if digitization:
        r /= 255
        g /= 255
        b /= 255
    maxc: float = max(r, g, b)
    minc: float = min(r, g, b)
    l: float = (minc+maxc) / 2.0
    if minc == maxc:
        return (0.0, l, 0.0)
    s: float
    if l <= 0.5:
        s = (maxc-minc) / (maxc+minc)
    else:
        s = (maxc-minc) / (2.0-maxc-minc)
    rc: float = (maxc-r) / (maxc-minc)
    gc: float = (maxc-g) / (maxc-minc)
    bc: float = (maxc-b) / (maxc-minc)
    h: float
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h/6.0) % 1.0
    if digitization:
        return (int(h*360), int(l*MAX_LS), int(s*MAX_LS))
    else:
        return (h, l, s)

def hls_to_rgb(hls: Color_type, digitization: bool = False, MAX_LS: int = 100) -> Color_type:
    """HLS -> RGB

    Note:
        The value range of (h,l,s) is:
            h: [0.0, 1.0] (or [0, 360])
            l: [0.0, 1.0] (or [0, MAX_LS])
            s: [0.0, 1.0] (or [0, MAX_LS])
            
        
        The value range of (r,g,b) is:
            r: [0.0, 1.0]
            g: [0.0, 1.0]
            b: [0.0, 1.0]
            
        
    Args:
        hls (Color_type): (h,l,s).
    """
    h,l,s = hls
    if digitization:
        h /= 360
        l /= MAX_LS
        s /= MAX_LS
    r: float; g: float; b: float
    if s == 0.0:
        r = g = b = l
    else:
        m2: float
        if l <= 0.5:
            m2 = l * (1.0+s)
        else:
            m2 = l + s - l*s
        m1: float = 2.0*l - m2
        r, g, b = _hls_calc(m1, m2, h+1.0/3.0), _hls_calc(m1, m2, h), _hls_calc(m1, m2, h-1.0/3.0)
    if digitization:
        return (int(r*255), int(g*255), int(b*255))
    else:
        return (r, g, b)

def _hls_calc(m1: float, m2: float, hue: float) -> float:
    hue = hue % 1.0
    if hue < 1.0/6.0:
        return m1 + (m2-m1)*hue*6.0
    if hue < 0.5:
        return m2
    if hue < 2.0/3.0:
        return m1 + (m2-m1)*(2.0/3.0-hue)*6.0
    return m1

def rgb_to_yiq(rgb: Color_type, digitization: bool = False) -> Color_type:
    """RGB -> YIQ

    Note:
        The value range of (r,g,b) is:
            r: [0.0, 1.0]
            g: [0.0, 1.0]
            b: [0.0, 1.0]
            
        
        The value range of (y,i,q) is:
            y: [0.0, 1.0]
            i: [-0.5959, 0.5959]
            q: [-0.5229, 0.5229].
        
    Args:
        rgb (Color_type): (r,g,b).
    """
    r,g,b = rgb
    if digitization:
        r /= 255
        g /= 255
        b /= 255
    y: float = 0.299*r + 0.587*g + 0.114*b
    i: float = 0.596*r - 0.274*g - 0.322*b
    q: float = 0.211*r - 0.523*g + 0.312*b
    return (y, i, q)

def yiq_to_rgb(yiq: Color_type, digitization: bool = False) -> Color_type:
    """YIQ -> RGB

    Note:
        The value range of (y,i,q) is:
            y: [0.0, 1.0]
            i: [-0.5959, 0.5959]
            q: [-0.5229, 0.5229].

        The value range of (r,g,b) is:
            r: [0.0, 1.0]
            g: [0.0, 1.0]
            b: [0.0, 1.0]
            

    Args:
        yiq (Color_type): (y,i,q).
    """
    y,i,q = yiq
    r: float = y + 0.956*i + 0.621*q
    g: float = y - 0.273*i - 0.647*q
    b: float = y - 1.104*i + 1.701*q
    if digitization:
        return (int(r*255), int(g*255), int(b*255))
    else:
        return (r, g, b)

def view_gradation(color_list: list[Color]) -> None:
    """Plot the color gradation by matplotlib.

    Args:
        color_list (list[Color]): list of `Color` instances.

    """
    x: npt.NDArray[np.float32] = np.linspace(-np.pi, np.pi)
    for i, c in enumerate(color_list):
        y: npt.NDArray[np.float32] = i/len(color_list) * np.sin(x)
        plt.plot(x, y, color=c.to_rgb()._color)
    plt.show()
    return


# constants
RGB: str = "RGB"
HSV: str = "HSV"
HLS: str = "HLS"
YIQ: str = "YIQ"

RED_RGB: Color_type = (1.0, 0.0, 0.0)
GREEN_RGB: Color_type = (0.0, 1.0, 0.0)
BLUE_RGB: Color_type = (0.0, 0.0, 1.0)
YELLOW_RGB: Color_type = (1.0, 1.0, 0.0)
MAGENTA_RGB: Color_type = (1.0, 0.0, 1.0)
CYAN_RGB: Color_type = (0.0, 1.0, 1.0)
WHITE_RGB: Color_type = (1.0, 1.0, 1.0)
BLACK_RGB: Color_type = (0.0, 0.0, 0.0)
GRAY_RGB: Color_type = (0.5, 0.5, 0.5)

RED_HSV: Color_type = (0.0, 1.0, 1.0)
YELLOW_HSV: Color_type = (1/6, 1.0, 1.0)
GREEN_HSV: Color_type = (2/6, 1.0, 1.0)
CYAN_HSV: Color_type = (3/6, 1.0, 1.0)
BLUE_HSV: Color_type = (4/6, 1.0, 1.0)
MAGENTA_HSV: Color_type = (5/6, 1.0, 1.0)
WHITE_HSV: Color_type = (0.0, 0.0, 1.0)
BLACK_HSV: Color_type = (0.0, 0.0, 0.0)

RED_HLS: Color_type = (0.0, 0.5, 1.0)
YELLOW_HLS: Color_type = (1/6, 0.5, 1.0)
GREEN_HLS: Color_type = (2/6, 0.5, 1.0)
CYAN_HLS: Color_type = (3/6, 0.5, 1.0)
BLUE_HLS: Color_type = (4/6, 0.5, 1.0)
MAGENTA_HLS: Color_type = (5/6, 0.5, 1.0)
WHITE_HLS: Color_type = (0.0, 1.0, 0.0)
BLACK_HLS: Color_type = (0.0, 0.0, 0.0)

RED_YIQ: Color_type = (0.299, 0.596, 0.211)
YELLOW_YIQ: Color_type = (0.886, 0.322, -0.312)
GREEN_YIQ: Color_type = (0.587, -0.274, -0.523)
CYAN_YIQ: Color_type = (0.701, -0.596, -0.211)
BLUE_YIQ: Color_type = (0.114, -0.322, 0.312)
MAGENTA_YIQ: Color_type = (0.413, 0.274, 0.523)
WHITE_YIQ: Color_type = (1.0, 0.0, 0.0)
BLACK_YIQ: Color_type = (0.0, 0.0, 0.0)



def main() -> None:
    pass
    def check_color_temp():
        for T in [1000,2000,3000,4000,5000,6500,7000,10000,50000]:
            print(T)
            print(Color.color_temperature(T))
            print(Color.color_temperature(T).srgb_255())
            print()
        color_list = [Color.color_temperature(T).to_srgb() for T in range(1000,10000+1,1000)]
        # Gradation.visualize_gradation(Gradation.planckian_curve(1000,10000))

        color_list = Gradation(Color(BLUE_HSV,"HSV"), Color(RED_HSV,"HSV")).gradation_helical_list(num=256,clockwise=False, next_color_system="RGB")
        color_list2 = Gradation(Color(BLUE_HSV,"HSV"), Color(RED_HSV,"HSV")).gradation_helical_list(num=256,clockwise=False, next_color_system="sRGB")
        color_list3 = Gradation(Color(BLUE_HSV,"HSV"), Color(RED_HSV,"HSV")).gradation_helical_list(num=10,clockwise=False, next_color_system="RGB")
        color_list4 = Gradation.equidistant(10)
        Gradation.compare_gradations([color_list,color_list2, color_list3, color_list4])

    # check_color_temp()
    # plot_colortable(mcolors.BASE_COLORS, ncols=3, sort_colors=False)

    # names = sorted(mcolors.CSS4_COLORS, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    # names = sorted([(*tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))),c) for c in mcolors.CSS4_COLORS])
    # print(names)


    # colors = [Color(Color.from_color_code(c),RGB) for _, c in mcolors.CSS4_COLORS.items()]

    # colors = [
    #     Color((240/360, 1.0, 0.5), HSV),
    #     Color((240/360, 1.0, 1.0), HSV),
    #     Color((210/360, 1.0, 1.0), HSV),
    #     Color((190/360, 1.0, 1.0), HSV),
    #     Color((160/360, 1.0, 1.0), HSV),
    #     Color((110/360, 1.0, 1.0), HSV),
    #     Color((70/360, 1.0, 1.0), HSV),
    #     Color((50/360, 1.0, 1.0), HSV),
    #     Color((30/360, 1.0, 1.0), HSV),
    #     Color((0/360, 1.0, 0.95), HSV),
    # ]
    # import matplotlib.cm as cm

    # colors = [
    #     Color.from_color_code("#000000"),
    #     # Color.from_color_code("#6600cc"),
    #     Color.from_color_code("#8c67aa").change_saturation_hsv(0.8),
    #     Color.from_color_code("#0000ff"),
    #     Color.from_color_code("#6292e9").change_saturation_hsv(0.8),
    #     Color.from_color_code("#95caee").change_saturation_hsv(0.5),
    #     Color.from_color_code("#62bd93").change_saturation_hsv(0.8),
    #     Color.from_color_code("#adff2f").change_saturation_hsv(0.8),
    #     Color.from_color_code("#fdd162").change_saturation_hsv(0.8),
    #     Color.from_color_code("#fe9952").change_saturation_hsv(0.8),
    #     # Color.from_color_code("#e08696"),
    #     Color.from_color_code("#fb5458").change_saturation_hsv(0.8),
    # ]
    # plot_colortable(colors, ncols=2, sort_colors=False)
    # # plot_colortable(Gradation.ten_colors(), ncols=1, sort_colors=False)
    # plt.show()

    # print(*[c.to_hsv() for c in colors],sep="\n")

    # print(Color((270/360,1.0,0.8), HSV).color_code())

    # print(colors)



    # r: Color = Color(RED_HSV, HSV)
    # g: Color = Color(GREEN_HSV, HSV)
    # b: Color = Color(BLUE_HSV, HSV)
    # g = Gradation.multi_middle([b,r,g], ["helical"]*3)
    # r: Color = Color(RED_RGB, RGB)
    # g: Color = Color(GREEN_RGB, RGB)
    # b: Color = Color(BLUE_RGB, RGB)
    # g = Gradation.multi_middle([b,r,g], ["linear"]*3)
    # g.visualize_gradation()







    # C1: Color = Color(RED_RGB, RGB)
    # C2: Color = Color(RED_HSV, HSV)
    # C3: Color = Color(GREEN_HSV, HSV)
    # C4: Color = Color(BLUE_YIQ, YIQ)
    # print(C1.color_code())
    # print(-(-C1.to_rgb()))

    # import numpy as np
    # A: npt.NDArray[np.float32] = np.array([
    #     [1.047886,  0.022919, -0.050216],
    #     [0.029582,  0.990484, -0.017079],
    #     [-0.009252,  0.015073,  0.751678]
    # ])
    # print(np.linalg.inv(A))
    # pass
    # # G = Gradation(Color(RED_RGB, RGB), Color(BLUE_RGB, RGB), middle=Color(GREEN_RGB, RGB))

    # G = Gradation(Color(RED_HSV, HSV), Color(BLUE_HSV, HSV), middle=Color(GREEN_HSV, HSV))
    # # G = Gradation(Color(RED_HLS, HLS), Color(BLUE_HLS, HLS))
    # # G = Gradation(Color(RED_YIQ, YIQ), Color(BLUE_YIQ, YIQ))

    # # view_gradation(G.gradation_linear_list())
    # # view_gradation(G.gradation_helical_list(clockwise=False))
    # print(G.gradation_chart_list(order=(2,1,0)))
    # view_gradation(G.gradation_chart_list(num=100,order=(0,2,1)))



if __name__ == "__main__":
    main()


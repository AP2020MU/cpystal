"""`color`: for dealing with color space.
"""
from __future__ import annotations

from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

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
        self.__color: Color_type = color
        self.__color_system: str = color_system
        self.__white_point: str = white_point
        self.__value_range: dict[str, list[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
            "RGB": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "HSV": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "HLS": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "sRGB": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "Adobe RGB": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "YIQ": [(0.0, 1.0), (-0.5959, 0.5959), (-0.5229, 0.5229)],
            "XYZ": [(0.0, 0.951), (0.0, 1.0), (0.0, 1.090)],
            "L*a*b*": [(0, 100), (-128, 127), (-128, 127)],
        }

    def __str__(self) -> str:
        return f"{self.color_system}{self.color}"
    
    def __repr__(self) -> str:
        return str(self)

    def __neg__(self) -> Color:
        new_color: Color = self.__deepcopy__()
        if new_color.color_system == "RGB":
            r,g,b = new_color.color
            m: int = max(r,g,b) + min(r,g,b)
            new_color.__color = (m-r, m-g, m-b)
            return new_color
        elif new_color.color_system == "HSV":
            h,s,v = new_color.color
            new_color.__color = ((h+0.5) % 1.0, s, v)
            return new_color
        elif new_color.color_system == "HLS":
            h,l,s = new_color.color
            new_color.__color = ((h+0.5) % 1.0, l, s)
            return new_color
        elif new_color.color_system == "YIQ":
            new_color = (-new_color.to_rgb()).rgb_to_yiq()
            return new_color
        elif new_color.color_system == "sRGB":
            new_color = (-new_color.to_rgb()).rgb_to_srgb()
            return new_color
        elif new_color.color_system == "Adobe RGB":
            new_color = (-new_color.to_rgb()).rgb_to_srgb()
            return new_color
        elif new_color.color_system == "XYZ":
            new_color = (-new_color.to_rgb()).rgb_to_xyz()
            return new_color
        elif new_color.color_system == "L*a*b*":
            new_color = (-new_color.to_rgb()).rgb_to_lab()
            return new_color
        else:
            pass

    def __add__(self, other: Color) -> Color:
        a,b,c = self.to_rgb().color
        x,y,z = other.to_rgb().color
        _, maxr, _, maxg, _, maxb = self.__value_range["RGB"]
        return self.__class__(
            color=(min(a+x,maxr), min(b+y, maxg), min(c+z, maxb)),
            color_system="RGB",
        )

    def __sub__(self, other: Color) -> Color:
        a,b,c = self.to_rgb().color
        x,y,z = other.to_rgb().color
        minr, _, ming, _, minb, _ = self.__value_range["RGB"]
        return self.__class__(
            color=(max(a-x,minr), max(b-y, ming), max(c-z, minb)),
            color_system="RGB",
        )
    
    def __mul__(self, other: Color) -> Color:
        """Additive color change.

        ref: A. Kitaoka, Journal of Color Science Association of Japan, 35(3), 234-236, (2011).
        """
        a,b,c = self.to_rgb().color
        x,y,z = other.to_rgb().color
        t: float = 0.5
        return self.__class__(
            color=(t*a+(1-t)*x, t*b+(1-t)*y, t*c+(1-t)*z),
            color_system="RGB",
        )
    
    def __div__(self, other: Color) -> Color:
        a,b,c = self.to_rgb().color
        x,y,z = other.to_rgb().color
        t: float = 0.5
        return self.__class__(
            color=(a/t-(1-t)/t*x, b/t+(1-t)/t*y, c/t+(1-t)/t*z),
            color_system="RGB",
        )
    
    def __mutmul__(self, other: Color) -> Color:
        """Multiplicative color change.

        ref: A. Kitaoka, Journal of Color Science Association of Japan, 35(3), 234-236, (2011).
        """
        a,b,c = self.to_rgb().color
        x,y,z = other.to_rgb().color
        t: float = 0.5
        return self.__class__(
            color=((t+(1-t)*x)*a, (t+(1-t)*y)*b, (t+(1-t)*z)*c),
            color_system="RGB",
        )
    
    def __len__(self) -> int:
        return len(self.color)

    def __iter__(self) -> float:
        yield from self.color
    
    def __getitem__(self, key: Any) -> float:
        return self.color[key]
    
    def __deepcopy__(self) -> Color:
        return self.__class__(
            color=self.color,
            color_system=self.color_system,
        )
    
    def __check_color_value(self) -> None:
        res: list = list(self.color)
        for i, c in enumerate(self.color):
            minc, maxc = self.__value_range[self.color_system][i]
            if c < minc:
                res[i] = minc
            if c > maxc:
                res[i] = maxc
        self.__color = tuple(res)

    def __hls_calc(self, m1: float, m2: float, hue: float) -> float:
        hue = hue % 1.0
        if hue < 1.0/6.0:
            return m1 + (m2-m1)*hue*6.0
        if hue < 0.5:
            return m2
        if hue < 2.0/3.0:
            return m1 + (m2-m1)*(2.0/3.0-hue)*6.0
        return m1

    @property
    def color(self) -> Color_type:
        return self.__color

    @property
    def color_system(self) -> Color_type:
        return self.__color_system

    @property
    def white_point(self) -> str:
        return self.__white_point
    
    @white_point.setter
    def white_point(self, value: str) -> None:
        """あとで実装する．これを実装するには，各white point間での colorの変換メソッドを実装する必要がある
        途中でwhite pointを変更されるとself.colorが指す色が変わってしまうので管理する．
        """
        raise NotImplementedError("future works")
        if self.__white_point != value:
            if self.color_system == "XYZ":
                pass
            if self.color_system == "L*a*b*":
                pass
    
    def deepcopy(self) -> Color:
        return self.__deepcopy__()

    def get_properties(self) -> tuple[str]:
        return (self.color_system, )

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

        r,g,b = self.color
        maxc: float = max(r, g, b)
        minc: float = min(r, g, b)
        v: float = maxc
        if maxc == minc:
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
        color: Color_type = (h, s, v)
        new_color: Color = self.__class__(
            color=color,
            color_system="HSV",
        )
        new_color.__check_color_value()
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

        h,s,v = self.color
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
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color.__check_color_value()
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

        r,g,b = self.color
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
        color: Color_type = (h, l, s)
        new_color: Color = self.__class__(
            color=color,
            color_system="HLS",
        )
        new_color.__check_color_value()
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

        h,l,s = self.color
        r: float
        g: float
        b: float
        if s == 0.0:
            r = g = b = l
        else:
            m2: float
            if l <= 0.5:
                m2 = l * (1.0+s)
            else:
                m2 = l + s - l*s
            m1: float = 2.0*l - m2
            r, g, b = self.__hls_calc(m1, m2, h+1.0/3.0), self.__hls_calc(m1, m2, h), self.__hls_calc(m1, m2, h-1.0/3.0)
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color.__check_color_value()
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
    
        r,g,b = self.color
        y: float = 0.299*r + 0.587*g + 0.114*b
        i: float = 0.596*r - 0.274*g - 0.322*b
        q: float = 0.211*r - 0.523*g + 0.312*b
        color: Color_type = (y, i, q)
        new_color: Color = self.__class__(
            color=color,
            color_system="YIQ",
        )
        new_color.__check_color_value()
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

        y,i,q = self.color
        r: float = y + 0.956*i + 0.621*q
        g: float = y - 0.273*i - 0.647*q
        b: float = y - 1.104*i + 1.701*q
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color.__check_color_value()
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

        r,g,b = self.color
        color: Color_type = (_f(r), _f(g), _f(b))
        new_color: Color = self.__class__(
            color=color,
            color_system="sRGB",
        )
        new_color.__check_color_value()
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

        sr,sg,sb = self.color
        color: Color_type = (_f(sr), _f(sg), _f(sb))
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color.__check_color_value()
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

        r,g,b = self.color
        color: Color_type = (_f(r), _f(g), _f(b))
        new_color: Color = self.__class__(
            color=color,
            color_system="Adobe RGB",
        )
        new_color.__check_color_value()
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

        ar,ag,ab = self.color
        color: Color_type = (_f(ar), _f(ag), _f(ab))
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color.__check_color_value()
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

        r,g,b = self.color
        x: float = 0.412391*r + 0.357584*g + 0.180481*b
        y: float = 0.212639*r + 0.715169*g + 0.072192*b
        z: float = 0.019331*r + 0.119195*g + 0.950532*b
        color: Color_type = (x, y, z)
        new_color: Color = self.__class__(
            color=color,
            color_system="XYZ",
        )
        # new_color.__check_color_value()
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
            
        x,y,z = self.color
        r: float = 3.240970*x - 1.537383*y - 0.498611*z
        g: float = -0.969244*x + 1.875968*y + 0.041555*z
        b: float = 0.055630*x - 0.203977*y + 1.056972*z
        color: Color_type = (r, g, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="RGB",
        )
        new_color.__check_color_value()
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
        x,y,z = self._d65_to_d50(self.color) # Bradford transformation
        fx,fy,fz = _f(x/xw), _f(y/yw), _f(z/zw)
        l: float = 116*fy - 16
        a: float = 500 * (fx-fy)
        b: float = 200 * (fy-fz)
        color: Color_type = (l, a, b)
        new_color: Color = self.__class__(
            color=color,
            color_system="L*a*b*",
        )
        new_color.__check_color_value()
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
        l,a,b = self.color
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
        # new_color.__check_color_value()
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
            return

    
    def color_code(self) -> str:
        """Hexadecimal color code

        Returns:
            (str): Color code in hexadecimal notation.
        """
        r,g,b = self.to_rgb()
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    
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
        u = np.array(self.to_rgb().color)
        v = np.array(other_color.to_rgb().color)
        lv = np.linalg.norm(v)
        if lv == 0.0:
            a, b, c = 0.0, 0.0, 0.0
        else:
            a, b, c = (u @ v) * v / (lv ** 2) # orthographic projection
        return self.__class__(
            color=(a,b,c),
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
        return str(self.to_xyz().color[1])


class Gradation:
    """Color gradation

    Attributes:
        start (Color): Start color of the gradation.
        end (Color): End color of the gradation.
        middle (Color | None): Middle color of the gradation. Defaults to None.

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
                ) -> None:
        if middle is None:
            if start.get_properties() != end.get_properties():
                raise ValueError("'start' and 'end' must have same properties")
        else:
            if not (start.get_properties() == middle.get_properties() == end.get_properties()):
                raise ValueError("'start' and 'middle' and 'end' must have same properties")
        self.start: Color = start
        self.end: Color = end
        self.middle: Color_type | None = middle

    def rgb_to_rgba(self, color_list: list[Color]) -> list[tuple[float,float,float,float]]:
        res: list[tuple[float,float,float,float]] = []
        for color in color_list:
            r,g,b = color.to_rgb()
            res.append((r, g, b, 1.0))
        return res

    def gradation_linear(self, proportion: float) -> Color:
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
        if self.middle is None:
            if self.start.get_properties() != self.end.get_properties():
                raise ValueError("'start' and 'end' must have same properties")
        else:
            if not (self.start.get_properties() == self.middle.get_properties() == self.end.get_properties()):
                raise ValueError("'start' and 'middle' and 'end' must have same properties")

        color_system: str = self.start.color_system
        u: float
        v: float
        w: float
        if self.middle is None:
            a,b,c = self.start
            x,y,z = self.end
            u = a + (x-a) * proportion
            v = b + (y-b) * proportion
            w = c + (z-c) * proportion
        else:
            a,b,c = self.start
            p,q,r = self.middle
            x,y,z = self.end
            if proportion < 0.5:
                u = a + (p-a) * proportion
                v = b + (q-b) * proportion
                w = c + (r-c) * proportion
            else:
                u = p + (x-p) * (proportion-0.5)
                v = q + (y-q) * (proportion-0.5)
                w = r + (z-r) * (proportion-0.5)
        return Color(color=(u,v,w),
            color_system=color_system,
        )

    def gradation_chart(self, proportion: float, order: tuple[int, int, int] = (0,1,2)) -> Color:
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
        if self.middle is None:
            if self.start.get_properties() != self.end.get_properties():
                raise ValueError("'start' and 'end' must have same properties")
        else:
            if not (self.start.get_properties() == self.middle.get_properties() == self.end.get_properties()):
                raise ValueError("'start' and 'middle' and 'end' must have same properties")

        color_system: str = self.start.color_system

        if self.middle is None:
            a,b,c = self.start
            x,y,z = self.end
            res: list[float, float, float] = [0., 0., 0.]
            dist: float = (abs(a-x) + abs(b-y) + abs(c-z)) * proportion
            if dist < abs(self.start[order[0]]-self.end[order[0]]):
                res[order[0]] = self.start[order[0]] + (self.end[order[0]]-self.start[order[0]]) * proportion
                res[order[1]] = self.start[order[1]]
                res[order[2]] = self.start[order[2]]
            elif dist < abs(self.start[order[0]]-self.end[order[0]]) + abs(self.start[order[1]]-self.end[order[1]]):
                res[order[0]] = self.end[order[0]]
                res[order[1]] = self.start[order[1]] + (self.end[order[1]]-self.start[order[1]]) * proportion
                res[order[2]] = self.start[order[2]]
            else:
                res[order[0]] = self.end[order[0]]
                res[order[1]] = self.start[order[1]]
                res[order[2]] = self.start[order[2]] + (self.end[order[2]]-self.start[order[2]]) * proportion
        else:
            a,b,c = self.start
            p,q,r = self.middle
            x,y,z = self.end
            res: list[float, float, float] = [0., 0., 0.]
            dist: float = (abs(a-p) + abs(b-q) + abs(c-r) + abs(p-x) + abs(q-y) + abs(r-z)) * proportion
            if dist < abs(a-p) + abs(b-q) + abs(c-r):
                if dist < abs(self.start[order[0]]-self.middle[order[0]]):
                    res[order[0]] = self.start[order[0]] + (self.middle[order[0]]-self.start[order[0]]) * proportion
                    res[order[1]] = self.start[order[1]]
                    res[order[2]] = self.start[order[2]]
                elif dist < abs(self.start[order[0]]-self.middle[order[0]]) + abs(self.start[order[1]]-self.middle[order[1]]):
                    res[order[0]] = self.middle[order[0]]
                    res[order[1]] = self.start[order[1]] + (self.middle[order[1]]-self.start[order[1]]) * proportion
                    res[order[2]] = self.start[order[2]]
                else:
                    res[order[0]] = self.middle[order[0]]
                    res[order[1]] = self.start[order[1]]
                    res[order[2]] = self.start[order[2]] + (self.middle[order[2]]-self.start[order[2]]) * proportion
            else:
                dist -= abs(a-p) + abs(b-q) + abs(c-r)
                if dist < abs(self.middle[order[0]]-self.end[order[0]]):
                    res[order[0]] = self.middle[order[0]] + (self.end[order[0]]-self.middle[order[0]]) * proportion
                    res[order[1]] = self.middle[order[1]]
                    res[order[2]] = self.middle[order[2]]
                elif dist < abs(self.middle[order[0]]-self.end[order[0]]) + abs(self.middle[order[1]]-self.end[order[1]]):
                    res[order[0]] = self.end[order[0]]
                    res[order[1]] = self.middle[order[1]] + (self.end[order[1]]-self.middle[order[1]]) * proportion
                    res[order[2]] = self.middle[order[2]]
                else:
                    res[order[0]] = self.end[order[0]]
                    res[order[1]] = self.middle[order[1]]
                    res[order[2]] = self.middle[order[2]] + (self.end[order[2]]-self.middle[order[2]]) * proportion
        return Color(color=tuple(res),
            color_system=color_system,
        )

    def gradation_helical(self,
                        proportion: float,
                        clockwise: bool = True,
                        ) -> Color:
        """Make a color gradation helically in a color space.
        This method is mainly used for HSV or HLS.

        Note:
            A gradation can be represented by a curve in a color space.
            Helical curves are used as the gradation curves in this method,
            assuming that a cylindrical coordinates system is defined in the color space.

        Args:
            proportion (float): Floating point number in [0.0, 1.0].
                proportion = 0.0 -> start color, proportion = 1.0 -> end color.
            clockwise (bool): If True, direction of spiral winding is clockwise. Defaults to True.

        Returns:
            (Color): Color corresponding the number 'proportion' in the gradation.
        """
        if not (0.0 <= proportion <= 1.0):
            raise ValueError("'proportion' must be in [0.0, 1.0]")
        if self.middle is None:
            if self.start.get_properties() != self.end.get_properties():
                raise ValueError("'start' and 'end' must have same properties")
        else:
            if not (self.start.get_properties() == self.middle.get_properties() == self.end.get_properties()):
                raise ValueError("'start' and 'middle' and 'end' must have same properties")
        
        start: Color = self.start.deepcopy()
        end: Color = self.end.deepcopy()
        middle: Color | None = self.middle
        if middle is not None:
            middle = self.middle.deepcopy()
        color_system: str = start.color_system
        u: float
        v: float
        w: float
        if middle is None:
            a,b,c = start
            x,y,z = end
            if clockwise and a > x:
                x += 1.0
            if (not clockwise) and a < x:
                x -= 1.0
            u = a + (x-a) * proportion
            v = b + (y-b) * proportion
            w = c + (z-c) * proportion
        else:
            a,b,c = start
            p,q,r = middle
            x,y,z = end
            if proportion < 0.5:
                if clockwise and a > p:
                    p += 1
                if (not clockwise) and a < p:
                    p -= 1
                u = a + (p-a) * proportion
                v = b + (q-b) * proportion
                w = c + (r-c) * proportion
            else:
                if clockwise and p > x:
                    x += 1
                if (not clockwise) and p < x:
                    x -= 1
                u = p + (x-p) * (proportion-0.5)
                v = q + (y-q) * (proportion-0.5)
                w = r + (z-r) * (proportion-0.5)
        return Color(color=(u%1.0, v, w),
            color_system=color_system,
        )

    def gradation_linear_list(self, num: int = 50) -> Color:
        """Make a list of color gradation linearly in a color space.

        Note:
            A gradation can be represented as a curve in a color space.
            Straight geodesic lines are used as the gradation curves in this method,
            assuming that the color space is considered as a real 3D Euclidean space.

        Args:
            num (int): Length of the return list.

        Returns:
            (list[Color]): Gradation color list.
        """
        if self.middle is None:
            if self.start.get_properties() != self.end.get_properties():
                raise ValueError("'start' and 'end' must have same properties")
        else:
            if not (self.start.get_properties() == self.middle.get_properties() == self.end.get_properties()):
                raise ValueError("'start' and 'middle' and 'end' must have same properties")

        color_system: str = self.start.color_system
        res: list[Color] = []
        if num == 1:
            return [self.start.deepcopy()]
        u: float
        v: float
        w: float
        if self.middle is None:
            a,b,c = self.start
            x,y,z = self.end
            for i in range(num):
                u = a + (x-a)*i/(num-1)
                v = b + (y-b)*i/(num-1)
                w = c + (z-c)*i/(num-1)
                res.append(
                    Color(color=(u,v,w),
                    color_system=color_system,
                    )
                )
            return res
        else:
            a,b,c = self.start
            p,q,r = self.middle
            x,y,z = self.end
            if num % 2 == 1:
                for i in range(num//2+1):
                    u = a + (p-a)*i/(num//2)
                    v = b + (q-b)*i/(num//2)
                    w = c + (r-c)*i/(num//2)
                    res.append(
                        Color(color=(u,v,w),
                        color_system=color_system,
                        )
                    )
                for i in range(1,num-num//2):
                    u = p + (x-p)*i/(num-num//2-1)
                    v = q + (y-q)*i/(num-num//2-1)
                    w = r + (z-r)*i/(num-num//2-1)
                    res.append(
                        Color(color=(u,v,w),
                        color_system=color_system,
                        )
                    )
            else:
                for i in range(num//2):
                    u = a + (p-a)*i*2/(num-1)
                    v = b + (q-b)*i*2/(num-1)
                    w = c + (r-c)*i*2/(num-1)
                    res.append(
                        Color(color=(u,v,w),
                        color_system=color_system,
                        )
                    )
                for i in range(num//2-1,-1,-1):
                    u = x - (x-p)*i*2/(num-1)
                    v = y - (y-q)*i*2/(num-1)
                    w = z - (z-r)*i*2/(num-1)
                    res.append(
                        Color(color=(u,v,w),
                        color_system=color_system,
                        )
                    )
            return res

    def gradation_chart_list(self, num: int = 50, order: tuple[float, float, float] = (0,1,2)) -> Color:
        """Make a list of color gradation like a chart in a color space.

        Note:
            A gradation can be represented as a curve in a color space.
            Chart lines along with each color axis are used as the gradation curves in this method,
            assuming that the color space is considered as a real 3D Euclidean space.

        Args:
            num (int): Length of the return list.

        Returns:
            (list[Color]): Gradation color list.
        """
        if self.middle is None:
            if self.start.get_properties() != self.end.get_properties():
                raise ValueError("'start' and 'end' must have same properties")
        else:
            if not (self.start.get_properties() == self.middle.get_properties() == self.end.get_properties()):
                raise ValueError("'start' and 'middle' and 'end' must have same properties")

        if num == 1:
            return [self.start.deepcopy()]
        
        return [self.gradation_chart(proportion=i/(num-1), order=order) for i in range(num)]
        
    def gradation_helical_list(self,
                        num: int = 50,
                        clockwise: bool = True,
                        ) -> list[Color]:
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
            (list[Color]): Gradation color list.
        """
        if self.middle is None:
            if self.start.get_properties() != self.end.get_properties():
                raise ValueError("'start' and 'end' must have same properties")
        else:
            if not (self.start.get_properties() == self.middle.get_properties() == self.end.get_properties()):
                raise ValueError("'start' and 'middle' and 'end' must have same properties")
        
        start: Color = self.start.deepcopy()
        end: Color = self.end.deepcopy()
        middle: Color | None = self.middle
        if middle is not None:
            middle = self.middle.deepcopy()

        color_system: str = start.color_system
        res: list[Color] = []
        if num == 1:
            return [start.deepcopy()]
        u: float
        v: float
        w: float
        if middle is None:
            a,b,c = start
            x,y,z = end
            if clockwise and a > x:
                x += 1
            if (not clockwise) and a < x:
                x -= 1
            for i in range(num):
                u = a + (x-a)*i/(num-1)
                v = b + (y-b)*i/(num-1)
                w = c + (z-c)*i/(num-1)
                res.append(
                    Color(color=(u%1.0, v, w),
                    color_system=color_system,
                    )
                )
            return res
        else:
            a,b,c = start
            p,q,r = middle
            x,y,z = end
            if num % 2 == 1:
                if clockwise and a > p:
                    p += 1
                if (not clockwise) and a < p:
                    p -= 1
                for i in range(num//2+1):
                    u = a + (p-a)*i/(num//2)
                    v = b + (q-b)*i/(num//2)
                    w = c + (r-c)*i/(num//2)
                    res.append(
                        Color(color=(u%1.0, v, w),
                        color_system=color_system,
                        )
                    )
                p %= 1.0
                if clockwise and p > x:
                    x += 1
                if (not clockwise) and p < x:
                    x -= 1
                for i in range(1,num-num//2):
                    u = p + (x-p)*i/(num-num//2-1)
                    v = q + (y-q)*i/(num-num//2-1)
                    w = r + (z-r)*i/(num-num//2-1)
                    res.append(
                        Color(color=(u%1.0, v, w),
                        color_system=color_system,
                        )
                    )
            else:
                if clockwise and a > p:
                    p += 1
                if (not clockwise) and a < p:
                    p -= 1
                for i in range(num//2):
                    u = a + (p-a)*i*2/(num-1)
                    v = b + (q-b)*i*2/(num-1)
                    w = c + (r-c)*i*2/(num-1)
                    res.append(
                        Color(color=(u%1.0, v, w),
                        color_system=color_system,
                        )
                    )
                p %= 1.0
                if clockwise and p > x:
                    x += 1
                if (not clockwise) and p < x:
                    x -= 1
                for i in range(num//2-1,-1,-1):
                    u = x - (x-p)*i*2/(num-1)
                    v = y - (y-q)*i*2/(num-1)
                    w = z - (z-r)*i*2/(num-1)
                    res.append(
                        Color(color=(u%1.0, v, w),
                        color_system=color_system,
                        )
                    )
            return res



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
    r: float
    g: float
    b: float
    if s == 0.0:
        r = g = b = l
    else:
        m2: float
        if l <= 0.5:
            m2 = l * (1.0+s)
        else:
            m2 = l + s - l*s
        m1: float = 2.0*l - m2
        r, g, b = __hls_calc(m1, m2, h+1.0/3.0), __hls_calc(m1, m2, h), __hls_calc(m1, m2, h-1.0/3.0)
    if digitization:
        return (int(r*255), int(g*255), int(b*255))
    else:
        return (r, g, b)

def __hls_calc(m1: float, m2: float, hue: float) -> float:
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
    x: np.ndarray = np.linspace(-np.pi, np.pi)
    for i, c in enumerate(color_list):
        print(i,c.to_rgb().color)
        y: np.ndarray = i/len(color_list) * np.sin(x)
        plt.plot(x, y, color=c.to_rgb().color)
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
    C: Color = Color(RED_RGB, RGB)
    C: Color = Color(RED_HSV, HSV)
    C: Color = Color(GREEN_HSV, HSV)
    C: Color = Color(BLUE_YIQ, YIQ)
    print(C.color_code())
    print(-(-C.to_rgb()))

    import numpy as np
    A = np.array([
        [1.047886,  0.022919, -0.050216],
        [0.029582,  0.990484, -0.017079],
        [-0.009252,  0.015073,  0.751678]
    ])
    print(np.linalg.inv(A))
    pass
    # G = Gradation(Color(RED_RGB, RGB), Color(BLUE_RGB, RGB), middle=Color(GREEN_RGB, RGB))

    G = Gradation(Color(RED_HSV, HSV), Color(BLUE_HSV, HSV), middle=Color(GREEN_HSV, HSV))
    # G = Gradation(Color(RED_HLS, HLS), Color(BLUE_HLS, HLS))
    # G = Gradation(Color(RED_YIQ, YIQ), Color(BLUE_YIQ, YIQ))

    # view_gradation(G.gradation_linear_list())
    # view_gradation(G.gradation_helical_list(clockwise=False))
    print(G.gradation_chart_list(order=(2,1,0)))
    view_gradation(G.gradation_chart_list(num=100,order=(0,2,1)))



if __name__ == "__main__":
    main()


"""`color`: for dealing with color space.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

Color_type = Union[Tuple[int,int,int], Tuple[float,float,float]]
class Color:
    """Express color spaces. Most of this class is based on the standard module 'colorsys'.

    Note:
        Thereafter, we use the following type alias without notice:
            Color_type := Union[Tuple[int,int,int], Tuple[float,float,float]].

        RGB = Red, Green, Blue
            R: [0.0, 1.0] (or [0, 255]), transparent -> red.
            G: [0.0, 1.0] (or [0, 255]), transparent -> green.
            B: [0.0, 1.0] (or [0, 255]), transparent -> blue.
        HSV = Hue(color phase), Saturation(colorfulness), Value(brightness)
            H: [0.0, 1.0] (or [0, 360]), red -> yellow -> green -> blue -> magenta -> red.
            S: [0.0, 1.0] (or [0, MAX_SV]), medium -> maximum.
            V: [0.0, 1.0] (or [0, MAX_SV]), black-weak -> black-strong.
        HLS = Hue(color phase), Luminance(lightness), Saturation(colorfulness)
            H: [0.0, 1.0] (or [0, 360]), red -> yellow -> green -> blue -> magenta -> red.
            V: [0.0, 1.0] (or [0, MAX_SV]), black -> white.
            S: [0.0, 1.0] (or [0, MAX_SV]), medium -> maximum.
        YIQ = Y(perceived grey level), I(same-phase), Q(anti-phase)
            Y: [0.0, 1.0], black -> white.
            I: [-0.5959, 0.5959], blue -> orange.
            Q: [-0.5229, 0.5229], green -> violet.
        
    Attributes:
        color (Color_type): Color value.
        color_system (str): Color system (RGB, HSV, HLS, YIQ).
        digitization (bool): If True, elements of 'color' are integers. 
            Otherwise floating point number in [0.0, 1.0].
        MAX_SV (int): Max value of S,V in HSV system.
        MAX_LS (int): Max value of L,S in HLS system.
    
    """
    def __init__(self,
                color: Color_type,
                color_system: str,
                digitization: bool = False, 
                MAX_SV: int = 100, 
                MAX_LS: int = 100,
                ) -> None:
        self.color: Color_type = color
        self.color_system: str = color_system
        self.digitization: bool = digitization
        self.MAX_SV: int = MAX_SV
        self.MAX_LS: int = MAX_LS
    
    def __str__(self) -> str:
        return f"{self.color} ({self.color_system})"

    def __iter__(self) -> float:
        yield from self.color
    
    def __deepcopy__(self) -> Color:
        return self.__class__(
            color=self.color,
            color_system=self.color_system,
            digitization=self.digitization,
            MAX_SV=self.MAX_SV,
            MAX_LS=self.MAX_LS,
        )

    def set_color(self, color: Color_type) -> None:
        self.color = color

    def set_color_system(self, color_system: str) -> None:
        self.color_system = color_system

    def change_digitization(self, digitization: bool) -> None:
        if self.digitization != digitization:
            if digitization:
                if self.color_system == "RGB":
                    r,g,b = self.color
                    self.color = (int(r*255), int(g*255), int(b*255))
                    self.digitization = digitization
                elif self.color_system == "HSV":
                    h,s,v = self.color
                    self.color = (int(h*360), int(s*self.MAX_SV), int(v*self.MAX_SV))
                    self.digitization = digitization
                elif self.color_system == "HLS":
                    h,l,s = self.color
                    self.color = (int(h*360), int(l*self.MAX_LS), int(s*self.MAX_LS))
                    self.digitization = digitization
                else:
                    pass
            else:
                if self.color_system == "RGB":
                    r,g,b = self.color
                    self.color = (r/255, g/255, b/255)
                    self.digitization = digitization
                elif self.color_system == "HSV":
                    h,s,v = self.color
                    self.color = (h/360, s/self.MAX_SV, v/self.MAX_SV)
                    self.digitization = digitization
                elif self.color_system == "HLS":
                    h,l,s = self.color
                    self.color = (h/360, l/self.MAX_LS, s/self.MAX_LS)
                    self.digitization = digitization
                else:
                    pass

    def rgb_to_hsv(self, digitization: Optional[bool] = None, MAX_SV: Optional[int] = None) -> Color:
        """RGB -> HSV

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0] (or [0, 255])
                g: [0.0, 1.0] (or [0, 255])
                b: [0.0, 1.0] (or [0, 255])
                (if 'digitization' is True, the right side is used).

            The value range of (h,s,v) is:
                h: [0.0, 1.0] (or [0, 360])
                s: [0.0, 1.0] (or [0, MAX_SV])
                v: [0.0, 1.0] (or [0, MAX_SV])
                (if 'digitization' is True, the right side is used).

        Args:
            digitization (Optional[bool]): If True, elements of 'color' will be integers. Defaults to None. 
            MAX_SV (Optional[int]): Max value of S,V. Defaults to None. 

        Returns:
            (Color): Color expressed in HSV.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")

        if digitization is None:
            digitization = self.digitization
        if MAX_SV is None:
            MAX_SV = self.MAX_SV

        r,g,b = self.color
        if self.digitization:
            r /= 255
            g /= 255
            b /= 255
        maxc: float = max(r, g, b)
        minc: float = min(r, g, b)
        v: float = maxc
        if maxc == minc:
            if self.digitization:
                return (0, 0, int(v*self.MAX_SV))
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
            color = (int(h*360), int(s*MAX_SV), int(v*MAX_SV))
        else:
            color = (h, s, v)
        new_color: Color = self.__class__(
                color=color,
                color_system="HSV",
                digitization=digitization,
                MAX_SV=MAX_SV,
                MAX_LS=self.MAX_LS,
                )
        return new_color

    def hsv_to_rgb(self, digitization: Optional[bool] = None) -> Color:
        """HSV -> RGB

        Note:
            The value range of (h,s,v) is:
                h: [0.0, 1.0] (or [0, 360])
                s: [0.0, 1.0] (or [0, MAX_SV])
                v: [0.0, 1.0] (or [0, MAX_SV])
                (if 'digitization' is True, the right side is used).
            
            The value range of (r,g,b) is:
                r: [0.0, 1.0] (or [0, 255])
                g: [0.0, 1.0] (or [0, 255])
                b: [0.0, 1.0] (or [0, 255])
                (if 'digitization' is True, the right side is used).
            
        Args:
            digitization (Optional[bool]): If True, elements of 'color' will be integers. Defaults to None. 

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "HSV":
            raise ValueError("'color_system' must be 'HSV'")

        if digitization is None:
            digitization = self.digitization

        h,s,v = self.color
        if self.digitization:
            h /= 360
            s /= self.MAX_SV
            v /= self.MAX_SV
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
            color = (int(r*255), int(g*255), int(b*255))
        else:
            color = (r, g, b)
        new_color: Color = self.__class__(
                color=color,
                color_system="RGB",
                digitization=digitization,
                MAX_SV=self.MAX_SV,
                MAX_LS=self.MAX_LS,
                )
        return new_color
        

    def rgb_to_hls(self, digitization: Optional[bool] = None, MAX_LS: Optional[int] = None) -> Color:
        """RGB -> HLS

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0] (or [0, 255])
                g: [0.0, 1.0] (or [0, 255])
                b: [0.0, 1.0] (or [0, 255])
                (if 'digitization' is True, the right side is used).
            
            The value range of (h,l,s) is:
                h: [0.0, 1.0] (or [0, 360])
                l: [0.0, 1.0] (or [0, MAX_LS])
                s: [0.0, 1.0] (or [0, MAX_LS])
                (if 'digitization' is True, the right side is used).
            
        Args:
            digitization (Optional[bool]): If True, elements of 'color' will be integers. Defaults to None. 

        Returns:
            (Color): Color expressed in HLS.
        """
        if self.color_system != "RGB":
            raise ValueError("'color_system' must be 'RGB'")

        if digitization is None:
            digitization = self.digitization
        if MAX_LS is None:
            MAX_LS = self.MAX_LS

        r,g,b = self.color
        if self.digitization:
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
            color = (int(h*360), int(l*MAX_LS), int(s*MAX_LS))
        else:
            color = (h, l, s)
        new_color: Color = self.__class__(
                color=color,
                color_system="HLS",
                digitization=digitization,
                MAX_SV=self.MAX_SV,
                MAX_LS=MAX_LS,
                )
        return new_color

    
    def hls_to_rgb(self, digitization: Optional[bool] = None) -> Color:
        """HLS -> RGB

        Note:
            The value range of (h,l,s) is:
                h: [0.0, 1.0] (or [0, 360])
                l: [0.0, 1.0] (or [0, MAX_LS])
                s: [0.0, 1.0] (or [0, MAX_LS])
                (if 'digitization' is True, the right side is used).
            
            The value range of (r,g,b) is:
                r: [0.0, 1.0] (or [0, 255])
                g: [0.0, 1.0] (or [0, 255])
                b: [0.0, 1.0] (or [0, 255])
                (if 'digitization' is True, the right side is used).
            
        Args:
            digitization (Optional[bool]): If True, elements of 'color' will be integers. Defaults to None. 

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "HLS":
            raise ValueError("'color_system' must be 'HLS'")

        if digitization is None:
            digitization = self.digitization

        h,l,s = self.color
        if self.digitization:
            h /= 360
            l /= self.MAX_LS
            s /= self.MAX_LS
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
        if digitization:
            color = (int(r*255), int(g*255), int(b*255))
        else:
            color = (r, g, b)
        new_color: Color = self.__class__(
                color=color,
                color_system="RGB",
                digitization=digitization,
                MAX_SV=self.MAX_SV,
                MAX_LS=self.MAX_LS,
                )
        return new_color
    
    def __hls_calc(self, m1: float, m2: float, hue: float) -> float:
        hue = hue % 1.0
        if hue < 1.0/6.0:
            return m1 + (m2-m1)*hue*6.0
        if hue < 0.5:
            return m2
        if hue < 2.0/3.0:
            return m1 + (m2-m1)*(2.0/3.0-hue)*6.0
        return m1
    
    def rgb_to_yiq(self) -> Color:
        """RGB -> YIQ

        Note:
            The value range of (r,g,b) is:
                r: [0.0, 1.0] (or [0, 255])
                g: [0.0, 1.0] (or [0, 255])
                b: [0.0, 1.0] (or [0, 255])
                (if 'digitization' is True, the right side is used).
            
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
        if self.digitization:
            r /= 255
            g /= 255
            b /= 255
        y: float = 0.299*r + 0.587*g + 0.114*b
        i: float = 0.596*r - 0.274*g - 0.322*b
        q: float = 0.211*r - 0.523*g + 0.312*b
        color: Color_type = (y, i, q)
        new_color: Color = self.__class__(
                color=color,
                color_system="YIQ",
                digitization=False,
                MAX_SV=self.MAX_SV,
                MAX_LS=self.MAX_LS,
                )
        return new_color

    def yiq_to_rgb(self, digitization: Optional[bool] = None) -> Color:
        """YIQ -> RGB

        Note:
            The value range of (y,i,q) is:
                y: [0.0, 1.0]
                i: [-0.5959, 0.5959]
                q: [-0.5229, 0.5229].

            The value range of (r,g,b) is:
                r: [0.0, 1.0] (or [0, 255])
                g: [0.0, 1.0] (or [0, 255])
                b: [0.0, 1.0] (or [0, 255])
                (if 'digitization' is True, the right side is used).

        Args:
            digitization (Optional[bool]): If True, elements of 'color' will be integers. Defaults to None. 

        Returns:
            (Color): Color expressed in RGB.
        """
        if self.color_system != "YIQ":
            raise ValueError("'color_system' must be 'YIQ'")

        if digitization is None:
            digitization = self.digitization

        y,i,q = self.color
        r: float = y + 0.956*i + 0.621*q
        g: float = y - 0.273*i - 0.647*q
        b: float = y - 1.104*i + 1.701*q
        if digitization:
            color = (int(r*255), int(g*255), int(b*255))
        else:
            color = (r, g, b)
        new_color: Color = self.__class__(
                color=color,
                color_system="RGB",
                digitization=digitization,
                MAX_SV=self.MAX_SV,
                MAX_LS=self.MAX_LS,
                )
        return new_color
    
    def to_rgb(self, digitization: Optional[bool] = None) -> Color:
        if self.color_system == "RGB":
            new_color: Color = self.__deepcopy__()
            new_color.change_digitization(digitization=digitization)
            return new_color
        elif self.color_system == "HSV":
            return self.hsv_to_rgb(digitization=digitization)
        elif self.color_system == "HLS":
            return self.hls_to_rgb(digitization=digitization)
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb(digitization=digitization)
        else:
            return
    
    def to_hsv(self, digitization: Optional[bool] = None, MAX_SV: Optional[int] = None) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_hsv(digitization=digitization, MAX_SV=MAX_SV)
        elif self.color_system == "HSV":
            new_color: Color = self.__deepcopy__()
            new_color.change_digitization(digitization=digitization)
            return new_color
        elif self.color_system == "HLS":
            return self.hls_to_rgb().rgb_to_hsv(digitization=digitization, MAX_SV=MAX_SV)
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb().rgb_to_hsv(digitization=digitization, MAX_SV=MAX_SV)
        else:
            return
    
    def to_hls(self, digitization: Optional[bool] = None, MAX_LS: Optional[int] = None) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_hls(digitization=digitization, MAX_LS=MAX_LS)
        elif self.color_system == "HSV":
            return self.hsv_to_rgb().rgb_to_hls(digitization=digitization, MAX_LS=MAX_LS)
        elif self.color_system == "HLS":
            new_color: Color = self.__deepcopy__()
            new_color.change_digitization(digitization=digitization)
            return new_color
        elif self.color_system == "YIQ":
            return self.yiq_to_rgb().rgb_to_hls(digitization=digitization, MAX_LS=MAX_LS)
        else:
            return

    def to_yiq(self, digitization: Optional[bool] = None) -> Color:
        if self.color_system == "RGB":
            return self.rgb_to_yiq()
        elif self.color_system == "HSV":
            return self.hsv_to_rgb(digitization=digitization).rgb_to_yiq()
        elif self.color_system == "HLS":
            return self.hls_to_rgb(digitization=digitization).rgb_to_yiq()
        elif self.color_system == "YIQ":
            new_color: Color = self.__deepcopy__()
            new_color.change_digitization(digitization=digitization)
            return new_color
        else:
            return
    
    def color_code(self) -> str:
        """Hexadecimal color code

        Returns:
            (str): Color code in hexadecimal notation.
        """
        r,g,b = self.to_rgb(digitization=True)
        return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_hsv(rgb: Color_type, digitization: bool = False, MAX_SV: int = 100) -> Color_type:
    """RGB -> HSV

    Note:
        The value range of (r,g,b) is:
            r: [0.0, 1.0] (or [0, 255])
            g: [0.0, 1.0] (or [0, 255])
            b: [0.0, 1.0] (or [0, 255])
            (if 'digitization' is True, the right side is used).
        
        The value range of (h,s,v) is:
            h: [0.0, 1.0] (or [0, 360])
            s: [0.0, 1.0] (or [0, MAX_SV])
            v: [0.0, 1.0] (or [0, MAX_SV])
            (if 'digitization' is True, the right side is used).
        
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
            (if 'digitization' is True, the right side is used).
        
        The value range of (r,g,b) is:
            r: [0.0, 1.0] (or [0, 255])
            g: [0.0, 1.0] (or [0, 255])
            b: [0.0, 1.0] (or [0, 255])
            (if 'digitization' is True, the right side is used).
        
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
            r: [0.0, 1.0] (or [0, 255])
            g: [0.0, 1.0] (or [0, 255])
            b: [0.0, 1.0] (or [0, 255])
            (if 'digitization' is True, the right side is used).
        
        The value range of (h,l,s) is:
            h: [0.0, 1.0] (or [0, 360])
            l: [0.0, 1.0] (or [0, MAX_LS])
            s: [0.0, 1.0] (or [0, MAX_LS])
            (if 'digitization' is True, the right side is used).
        
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
            (if 'digitization' is True, the right side is used).
        
        The value range of (r,g,b) is:
            r: [0.0, 1.0] (or [0, 255])
            g: [0.0, 1.0] (or [0, 255])
            b: [0.0, 1.0] (or [0, 255])
            (if 'digitization' is True, the right side is used).
        
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
            r: [0.0, 1.0] (or [0, 255])
            g: [0.0, 1.0] (or [0, 255])
            b: [0.0, 1.0] (or [0, 255])
            (if 'digitization' is True, the right side is used).
        
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
            r: [0.0, 1.0] (or [0, 255])
            g: [0.0, 1.0] (or [0, 255])
            b: [0.0, 1.0] (or [0, 255])
            (if 'digitization' is True, the right side is used).

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




def gradation_linear(start: Color_type,
                    end: Color_type,
                    middle: Optional[Color_type] = None,
                    num: int = 50,
                    ) -> List[Color_type]:
    """Make a list of color gradation linearly in a color space.

    Note:
        A gradation can be represented as a curve in a color space.
        Straight geodesic lines are used as the gradation curves in this method,
        assuming that the color space is considered as a real 3D Euclidean space.

    Args:
        start (Color_type): Start color of the gradation.
        end (Color_type): End color of the gradation.
        middle (Optional[Color_type]): Middle color of the gradation. Defaults to None.
        num (int): Length of the return list.

    Returns:
        (List[Color_type]): Gradation color list.
    """
    res: List[Color_type] = []
    if num == 1:
        return [start]
    u: float
    v: float
    w: float
    if middle is None:
        a,b,c = start
        x,y,z = end
        for i in range(num):
            u = a + (x-a)*i/(num-1)
            v = b + (y-b)*i/(num-1)
            w = c + (z-c)*i/(num-1)
            res.append((u,v,w))
        return res
    else:
        a,b,c = start
        p,q,r = middle
        x,y,z = end
        if num % 2 == 1:
            for i in range(num//2+1):
                u = a + (p-a)*i/(num//2)
                v = b + (q-b)*i/(num//2)
                w = c + (r-c)*i/(num//2)
                res.append((u,v,w))
            for i in range(1,num-num//2):
                u = p + (x-p)*i/(num-num//2-1)
                v = q + (y-q)*i/(num-num//2-1)
                w = r + (z-r)*i/(num-num//2-1)
                res.append((u,v,w))
        else:
            for i in range(num//2):
                u = a + (p-a)*i*2/(num-1)
                v = b + (q-b)*i*2/(num-1)
                w = c + (r-c)*i*2/(num-1)
                res.append((u,v,w))
            for i in range(num//2-1,-1,-1):
                u = x - (x-p)*i*2/(num-1)
                v = y - (y-q)*i*2/(num-1)
                w = z - (z-r)*i*2/(num-1)
                res.append((u,v,w))
        return res
    
def gradation_helical(start: Color_type,
                    end: Color_type,
                    middle: Optional[Color_type] = None,
                    num: int = 50,
                    clockwise: bool = True,
                    ) -> List[Color_type]:
    """Make a list of color gradation helically in a color space.
    This method is mainly used for HSV or HLS.

    Note:
        A gradation can be represented by a curve in a color space.
        Helical curves are used as the gradation curves in this method,
        assuming that a cylindrical coordinates system is defined in the color space.

    Args:
        start (Color_type): Start color of the gradation.
        end (Color_type): End color of the gradation.
        middle (Optional[Color_type]): Middle color of the gradation. Defaults to None.
        num (int): Length of the return list.
        clockwise (bool): If True, direction of spiral winding is clockwise. Defaults to True.

    Returns:
        (List[Color_type]): Gradation color list.
    """
    res: List[Color_type] = []
    if num == 1:
        return [start]
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
            res.append((u%1.0, v, w))
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
                res.append((u%1.0, v, w))
            p %= 1.0
            if clockwise and p > x:
                x += 1
            if (not clockwise) and p < x:
                x -= 1
            for i in range(1,num-num//2):
                u = p + (x-p)*i/(num-num//2-1)
                v = q + (y-q)*i/(num-num//2-1)
                w = r + (z-r)*i/(num-num//2-1)
                res.append((u%1.0, v, w))
        else:
            if clockwise and a > p:
                p += 1
            if (not clockwise) and a < p:
                p -= 1
            for i in range(num//2):
                u = a + (p-a)*i*2/(num-1)
                v = b + (q-b)*i*2/(num-1)
                w = c + (r-c)*i*2/(num-1)
                res.append((u%1.0, v, w))
            p %= 1.0
            if clockwise and p > x:
                x += 1
            if (not clockwise) and p < x:
                x -= 1
            for i in range(num//2-1,-1,-1):
                u = x - (x-p)*i*2/(num-1)
                v = y - (y-q)*i*2/(num-1)
                w = z - (z-r)*i*2/(num-1)
                res.append((u%1.0, v, w))
        return res



RED_RGB: Color_type = (1.0, 0.0, 0.0)
GREEN_RGB: Color_type = (0.0, 1.0, 0.0)
BLUE_RGB: Color_type = (0.0, 0.0, 1.0)
RGB: str = "RGB"
HSV: str = "HSV"
HLS: str = "HLS"
YIQ: str = "YIQ"


def main() -> None:

    C = Color(RED_RGB, RGB)
    red = (1,0,0)
    blue = (0,0,1)
    green = (0,1,0)
    # print(C.gradation_linear(red, blue, middle=green, num=100))
    p = (0.2,0,0)
    q = (0.6,0,0)
    r = (0.8,0,0)
    print(gradation_helical(p, r, middle=q, num=5, clockwise=False))
    print(C.color_code())

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    x = np.linspace(0,2*np.pi,100)
    for i in range(30):   
        plt.plot(x,i*np.sin(x),color=cm.hsv(i/30.0))
    plt.xlim(0,2*np.pi)
    plt.show()



    pass
    return

if __name__ == "__main__":
    main()


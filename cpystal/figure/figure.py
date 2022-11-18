from __future__ import annotations

import glob
import os

from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageFont, ImageChops # type: ignore

class ImageProcessing:
    """画像処理

    Attributes:
        filepath (str): 対象の画像ファイルパス
        dirname (str): ファイルパスのうち上位ディレクトリを表す部分
        name_original (str): 元画像の(拡張子を除いた)ファイル名
        extension (str): 拡張子
        name_current (str): 現在の(拡張子を除いた)ファイル名
        original (str): 元画像
        current (str): 処理後の画像

    Methods:
        all_transaction: カレントディレクトリ内の全画像について、画像形式をselfの形式に一括変更したものを新たに生成
        split: 画像分割
        save: 画像保存
        change_name: (拡張子を除いた)新しいファイル名を設定
        grayscaling: グレースケール化
        transparent: 透過化
        eliminate_color: 特定の色を透明化
        invert: 画素値反転
        flip: 上下鏡映反転
        mirror: 左右鏡映反転
        rotate: 任意角度回転
        add_margin: 余白追加
        erode_margin: 余白侵食
        delete_margin: 余白削除
        resize1: サイズ変更(倍率指定)
        resize2: サイズ変更(参照画像指定)
        to_A4: A4サイズに変更(dpi=300)
        crop: 切り取り
        crop_center: 画像中央を中心に切り取り
        crop_max_square: 画像から最大正方形を切り取り
        blur: ぼかし
        text: テキスト追加
        combine: 画像連結
        paste: 画像貼り付け
        paint_rectangle: 指定領域を単色塗りつぶし
        extract_line: 線画抽出
        blend: 一様に画像合成
        composite: マスク画像に沿って画像合成
    """
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.dirname: str = os.path.dirname(filepath)
        self.name_original: str
        self.extension: str
        self.name_original, self.extension = os.path.splitext(os.path.basename(filepath))
        self.name_current: str = self.name_original
        img: Image = Image.open(filepath)
        self.original: Image = img
        self.current: Image = img

    def all_transaction(self) -> None:
        """カレントディレクトリ内の全画像について、画像形式をselfの形式に一括変更したものを新たに生成
        """
        current_directory_path: str = os.getcwd()
        files: list[str] = glob.glob(current_directory_path + '/*')
        l_extension: int = len(self.extension)
        for f in files:
            if f[-l_extension] == '.' and f[-l_extension:] != self.extension:
                input_im: Image = Image.open(f)
                rgb_im: Image = input_im.convert('RGB')
                ftitle, fextension = os.path.splitext(f)
                if self.extension == '.jpg':
                    rgb_im.save(ftitle + self.extension, 'JPEG', quality=95)
                else:
                    rgb_im.save(ftitle + self.extension, self.extension[-l_extension+1:].upper(), quality=95)
        print("transaction finished")
    
    def split(self, vertical_split: int, horizontal_split: int, flag: bool = False) -> list:
        """画像分割

        Args:
            vertical_split (int): 縦の分割数
            horizontal_split (int): 横の分割数
            flag (bool): Trueなら保存

        Returns:
            (list[list[Image]]): 画像行列
        """
        height_splitted: int = self.current.height // vertical_split - 1
        width_splitted: int = self.current.width // horizontal_split - 1
        im_matrix: list[list] = []
        for v in range(vertical_split):
            im_list: list = []
            for h in range(horizontal_split):
                im_splitted_vh: Image = self.current.crop((h*width_splitted, v*height_splitted,  h*width_splitted+width_splitted, v*height_splitted+height_splitted))
                im_list.append(im_splitted_vh)
                if flag:
                    im_splitted_vh.save(f"{v}{h}{self.extension}", quality=95)
            im_matrix.append(im_list)
        return im_matrix
    

    def save(self, dpi = (300, 300)) -> None:
        """画像保存
        """
        if self.current.mode != "RGB":
            self.current.convert("RGB")
        self.current.save(self.name_current+'_changed'+self.extension, quality=95, dpi=dpi)
    
    def change_name(self, new_name: str) -> None:
        """名前変更

        Args:
            new_name (str): 新しい名前
        """
        self.name_current = new_name
    
    def grayscaling(self, flag: bool = False) -> Image:
        """モノクロ化

        Args:
            flag (bool): Trueなら保存
        """
        im_gray: Image = self.current.convert('L')
        new_name: str = self.name_current + '_grayed'
        self.change_name(new_name)
        if flag:
            im_gray.save(new_name + self.extension, quality=95)
        return im_gray
    
    def transparent(self, alpha: int, flag: bool = False) -> Image:
        """透過画像化(png形式)
            alpha == 0(100%透過)~255(0%透過(不透過))

        Args:
            alpha (int): 透過率(0:100%透過, 255:0%透過)
            flag (bool): Trueなら保存
        """
        if self.extension != '.png':
            self.current.save(self.name_current + '_extensioned' + '.png', 'PNG') # PNG形式で保存
            self.current = Image.open(self.name_current + '_extensioned' + '.png') # self.currentをpng画像として改めてopen
        self.current.putalpha(alpha)
        new_name: str = self.name_current + '_tranparent'
        self.change_name(new_name)
        if flag:
            self.current.save(new_name + self.extension, quality=95)
        return self.current

    def eliminate_color(self, color: tuple[int, int, int] = (255, 255, 255), flag: bool = False) -> Image:
        """特定の色を透明化

        Args:
            color (tuple[int, int, int]): (R,G,B)
            flag (bool): Trueなら保存
        """
        if self.extension != '.png':
            self.current.save(self.name_current + '_extensioned' + '.png', 'PNG') # PNG形式で保存
            self.current = Image.open(self.name_current + '_extensioned' + '.png') # self.currentをpng画像として改めてopen
        trans: Image = Image.new('RGBA', self.current.size, (0, 0, 0, 0))
        for x in range(self.current.size[0]):
            for y in range(self.current.size[1]):
                pixel = self.current.getpixel((x,y))
                if pixel[0] == color[0] and pixel[1] == color[1] and pixel[2] == color[2]:
                    continue
                trans.putpixel((x,y), pixel)
        self.current = trans
        new_name: str = self.name_current + '_eliminated'
        self.change_name(new_name)
        if flag:
            self.current.save(new_name + self.extension, quality=95)
        return self.current
    
    def invert(self, flag: bool = False) -> Image:
        """画素反転

        Args:
            flag (bool): Trueなら保存
        """
        im_invert: Image = ImageOps.invert(self.current.convert('RGB'))
        new_name: str = self.name_current + '_invert'
        self.current = im_invert.copy()
        self.change_name(new_name)
        if flag:
            im_invert.save(new_name + self.extension, quality=95)
        return im_invert

    def flip(self, flag: bool = False) -> Image:
        """上下反転

        Args:
            flag (bool): Trueなら保存
        """
        im_flip: Image = ImageOps.flip(self.current)
        new_name: str = self.name_current + '_flip'
        self.current = im_flip.copy()
        self.change_name(new_name)
        if flag:
            im_flip.save(new_name + self.extension, quality=95)
        return im_flip

    def mirror(self, flag: bool = False) -> Image:
        """左右反転

        Args:
            flag (bool): Trueなら保存
        """
        im_mirror: Image = ImageOps.mirror(self.current)
        new_name: str = self.name_current + '_mirror'
        self.current = im_mirror.copy()
        self.change_name(new_name)
        if flag:
            im_mirror.save(new_name + self.extension, quality=95)
        return im_mirror

    def rotate(self, angle: float, flag: bool = False) -> Image:
        """任意角度回転

        Args:
            angle (float): 角度(°)
            flag (bool): Trueなら保存
        """
        im_rotate: Image = self.current.rotate(angle, expand=True)
        new_name: str = self.name_current + '_rotate'
        self.current = im_rotate.copy()
        self.change_name(new_name)
        if flag:
            im_rotate.save(new_name + self.extension, quality=95)
        return im_rotate

    def add_margin(self, left: int, top: int, right: int, bottom: int, color: tuple[int, int, int] | int = (255, 255, 255), flag: bool = False) -> Image:
        """余白(left, top, right, bottom)を外側に追加

        Args:
            left (int): 左側の余白 (pixel)
            top (int): 上側の余白 (pixel)
            right (int): 右側の余白 (pixel)
            bottom (int): 下側の余白 (pixel)
            color (tuple[int, int, int] | int): 余白の色(R,G,B)
            flag (bool): Trueなら保存
        """
        mode: str = self.current.mode
        if mode == 'L':
            color = 255 # これは白
        new_width: int = self.current.width + left+right
        new_height: int = self.current.height + top+bottom
        im_added: Image = Image.new(mode, (new_width, new_height), color)
        im_added.paste(self.current, (left, top))
        new_name: str = self.name_current + '_margined'
        self.current = im_added.copy()
        self.change_name(new_name)
        if flag:
            im_added.save(new_name + self.extension, quality=95)
        return im_added

    def erode_margin(self, left: int, top: int, right: int, bottom: int, color: tuple[int, int, int] | int = (255, 255, 255), flag: bool = False) -> Image:
        """サイズを変えずに元画像に余白を侵食

        Args:
            left (int): 左側の余白 (pixel)
            top (int): 上側の余白 (pixel)
            right (int): 右側の余白 (pixel)
            bottom (int): 下側の余白 (pixel)
            color (tuple[int, int, int] | int): 余白の色(R,G,B)
            flag (bool): Trueなら保存
        """
        mode: str = self.current.mode
        if mode == 'L':
            color = 0
        im_eroded: Image = Image.new(mode, (self.current.width, self.current.height), color)
        cropped: Image = self.current.crop((left, top, self.current.width-right, self.current.height-bottom))
        im_eroded.paste(cropped, (left, top))
        new_name: str = self.name_current + '_eroded'
        self.current = im_eroded.copy()
        self.change_name(new_name)
        if flag:
            im_eroded.save(new_name + self.extension, quality=95)
        return im_eroded
    
    def delete_margin(self, standard_point: tuple[int, int] = (0, 0), offset: int = -50, flag: bool = False) -> Image:
        """余白を削除(基準点と同じ色の画像外縁部に存在する余白を削除)

        Args:
            flag (bool): Trueなら保存
        """
        img: Image = self.current
        bg: Image = Image.new(img.mode, img.size, img.getpixel(standard_point))
        diff: Image = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, offset) # 余白内の色の揺らぎの分を考慮(pixelの値は0以下だと0,255以上だと255になるため、元画像上でstandard_pointの色に近いものも削除対象になる)
        ### ImageChops.add(im1, im2, scale=1.0, offset=0.0) ## 計算式: 出力=((im1+im2)/scale+offset)%255
        croprange: Image = diff.convert("RGB").getbbox()
        demargined_img: Image = img.crop(croprange)
        new_name: str = self.name_current + '_demargined'
        self.current = demargined_img.copy()
        self.change_name(new_name)
        if flag:
            demargined_img.save(new_name + self.extension, quality=95)
        return demargined_img
    
    def resize1(self, magnification: float, flag: bool = False) -> Image:
        """拡大・縮小によるサイズ変更1

        Args:
            magnification (float): 拡大倍率
            flag (bool): Trueなら保存
        """
        im_resize_lanczos: Image = self.current.resize((int(self.current.width/magnification), int(self.current.height/magnification)), Image.LANCZOS)
        new_name: str = self.name_current + '_resized1'
        self.current = im_resize_lanczos.copy()
        self.change_name(new_name)
        if flag:
            im_resize_lanczos.save(new_name + self.extension, quality=95)
        return im_resize_lanczos
    
    def resize2(self, filename: str, flag: bool = False) -> Image:
        """拡大・縮小によるサイズ変更2(大きさを画像nameと同じサイズに変更)

        Args:
            filename (str): 画像ファイルパス
            flag (bool): Trueなら保存
        """
        im_standard: Image = Image.open(filename)
        im_resize_lanczos: Image = self.current.resize((im_standard.width, im_standard.height), Image.LANCZOS)
        new_name: str = self.name_current + '_resized2'
        self.current = im_resize_lanczos.copy()
        self.change_name(new_name)
        if flag:
            im_resize_lanczos.save(new_name + self.extension, quality=95)
        return im_resize_lanczos
    
    def to_A4(self, flag: bool = False) -> Image:
        """A4サイズになるように余白を追加(解像度300dpiの場合)

        Args:
            flag (bool): Trueなら保存
        """
        if self.current.width < self.current.height:
            self.current.rotate(90, expand=True)
        if self.current.height/self.current.width < 3508/2480:
            self.resize1(2480,self.current.width)
            self.add_margin(0, 0, 0, 3508-self.current.height)
        else:
            self.resize1(3508,self.current.height)
            self.add_margin(0, 0, 2480-self.current.width, 0)
        new_name: str = self.name_current + '_A4'
        self.change_name(new_name)
        if flag:
            self.current.save(new_name + self.extension, quality=95, dpi=(300,300))
        return self.current

    def crop(self, left_up: tuple[int, int], right_down: tuple[int, int], flag: bool = False) -> Image:
        """切り出し(left_up,right_downはそれぞれ左上の点の座標と右下の点の座標)
            width = right, height = lowerのピクセルは含まれない

        Args:
            left_up (tuple[int, int]): (width, height)
            right_down (tuple[int, int]): (width, height)
            flag (bool): Trueなら保存
        """
        l, u = left_up
        r, d = right_down
        im_crop: Image = self.current.crop((l, u, r, d))
        new_name: str = self.name_current + '_cropped'
        self.current = im_crop.copy()
        self.change_name(new_name)
        if flag:
            im_crop.save(new_name + self.extension, quality=95)
        return im_crop

    def crop_center(self, crop_width: int, crop_height: int, flag: bool = False) -> Image:
        """切り出し(画像の中心を任意のサイズでトリミング)
            切り出す領域に画像の範囲外を指定した場合、エラーにはならず黒く表示される
        
        Args:
            crop_width (int): width方向の長さ
            crop_height (int): height方向の長さ
            flag (bool): Trueなら保存
        """
        im_crop_center: Image = self.current.crop(((self.current.width - crop_width) // 2,
                             (self.current.height - crop_height) // 2,
                             (self.current.width + crop_width) // 2,
                             (self.current.height + crop_height) // 2))
        new_name: str = self.name_current + '_cropped_center'
        self.current = im_crop_center.copy()
        self.change_name(new_name)
        if flag:
            im_crop_center.save(new_name + self.extension, quality=95)
        return im_crop_center

    def crop_max_square(self, flag: bool = False) -> Image:
        """切り出し3(画像の中心から短辺の長さの正方形(つまり最大の正方形)を切り出す)

        Args:
            flag (bool): Trueなら保存
        """
        im_crop_max_square: Image = self.crop_center(min(self.current.size), min(self.current.size))
        new_name: str = self.name_current + '_cropped_max_square'
        self.current = im_crop_max_square.copy()
        self.change_name(new_name)
        if flag:
            im_crop_max_square.save(new_name + self.extension, quality=95)
        return im_crop_max_square

    def blur(self, flag: bool = False) -> Image:
        """ぼかし

        Args:
            flag (bool): Trueなら保存
        """
        im_blurred: Image = self.current.filter(ImageFilter.GaussianBlur(2))
        new_name: str = self.name_current + '_blurred'
        self.current = im_blurred.copy()
        self.change_name(new_name)
        if flag:
            im_blurred.save(new_name + self.extension, quality=95)
        return im_blurred

    def text(self, txt: str, font_size: int, font_color: tuple[int, int, int] | int, position: tuple[int, int], flag: bool = False) -> Image: 
        """文字の埋め込み

        Args:
            txt (str): テキスト
            font_size (int): フォントサイズ
            font_color (tuple[int, int, int] | int): 文字色
            position (tuple[int, int]): テキストの位置(width, height)
            flag (bool): Trueなら保存
        """
        im_str: Image = self.current.copy()
        draw3: ImageDraw = ImageDraw.Draw(im_str)
        font_str: str = ImageFont.truetype( "/System/Library/Fonts/Courier.dfont", size=font_size)
        draw3.text(position, txt, font_color, fill='red', align='center',font=font_str)
        self.current = im_str.copy()
        new_name: str = self.name_current + '_str'
        self.change_name(new_name)
        if flag:
            im_str.save(new_name + self.extension, quality=95)
        return im_str

    def combine(self, filename: str, direction: str, color: tuple[int, int, int] | int = (255, 255, 255), flag: bool = False) -> Image:
        """連結(2つの画像を連結)

        Note:
            サイズが不一致なら余白はcolorで塗られる

        Args:
            filename (str): ファイル名
            deirection (str): 連結方向 'horizontal' or 'vertical'
            color (tuple[int, int, int] | int): 背景色
            flag (bool): Trueなら保存
        """ 
        mode: str = self.current.mode
        if mode == 'L':
            color = 255
        im: Image = Image.open(filename)
        if direction == 'horizontal':
            dst = Image.new('RGB', (self.current.width + im.width, max(self.current.height, im.height)), color)
            dst.paste(self.current, (0, 0))
            dst.paste(im, (self.current.width, 0))
        else:
            dst = Image.new('RGB', (max(self.current.width,im.width), self.current.height + im.height), color)
            dst.paste(self.current, (0, 0))
            dst.paste(im, (0, self.current.height))
        new_name: str = self.name_current + filename + '_combined'
        self.current = dst.copy()
        self.change_name(new_name)
        if flag:
            dst.save(new_name + self.extension, quality=95)
        return dst

    def paste(self, filename: str, position: tuple[int, int], back: bool = True, flag: bool = False) -> Image:
        """貼り付け

        Args:
            filename (str): ファイル名
            position (tuple[int, int]): 貼り付け位置 (画像左上が(width, height))
            back (bool): Trueならselfが上側にくる
            color (tuple[int, int, int] | int): 背景色
            flag (bool): Trueなら保存
        """
        back_im: Image = self.current.copy()
        main_im: Image = Image.open(filename)
        if not back:
            back_im, main_im = main_im,back_im
        back_im.paste(main_im, position)
        new_name: str = self.name_current + os.path.splitext(os.path.basename(filename))[0] + '_paste'
        self.current = back_im.copy()
        self.change_name(new_name)
        if flag:
            back_im.save(new_name + self.extension, quality=95)
        return back_im

    def paint_rectangle(self, left_up: tuple[int, int], right_down: tuple[int, int], color: tuple[int, int, int] | int = (255, 255, 255), flag: bool = False) -> Image:
        """範囲選択してその範囲をcolorに変更

        Args:
            left_up (tuple[int, int]): (width, height)
            right_down (tuple[int, int]): (width, height)
            color (tuple[int, int, int] | int): 背景色
            flag (bool): Trueなら保存
        """
        l, u = left_up
        r, d = right_down
        mode: str = self.current.mode
        if mode == 'L':
            for h in range(d-u):
                for w in range(r-l):
                    self.current.putpixel((l+w,u+h), color)
        elif mode == 'RGB':
            for h in range(d-u):
                for w in range(r-l):
                    self.current.putpixel((l+w,u+h), color)
        elif mode == 'RGBA':
            for h in range(d-u):
                for w in range(r-l):
                    self.current.putpixel((l+w,u+h), color)
        new_name: str = self.name_current + '_paint_rectangled'
        self.change_name(new_name)
        if flag:
            self.current.save(new_name + self.extension, quality=95)
        return self.current

    def extract_line(self, flag: bool = False) -> Image:
        """線画抽出
        """
        gray: Image = self.current.convert("L")
        gray2: Image = gray.filter(ImageFilter.MaxFilter(5))
        senga_inv: Image = ImageChops.difference(gray, gray2)
        senga: Image = ImageOps.invert(senga_inv)
        self.current = senga.copy()
        new_name: str = self.name_current + '_senga'
        self.change_name(new_name)
        if flag:
            senga.save(new_name + self.extension, quality=95)
        return senga
    
    def blend(self, im: Image, mask: float, flag: bool = False) -> Image:
        """一様な画像合成

        Note:
            im2を強制的にselfと同じサイズに変更
        
        Args:
            im (Image): 合成用の画像
            mask (float): [0.0,1.0]の実数
            flag (bool): Trueなら保存
        """
        im = im.resize(self.current.size, Image.LANCZOS)
        im_blended: Image = Image.blend(self.current, im, mask)
        self.current = im_blended.copy()
        new_name: str = self.name_current + '_blended'
        if flag:
            im_blended.save(new_name + self.extension, quality=95)
        return im_blended
    
    def composite(self, im: Image, mask: Image, flag: bool = False) -> Image:
        """画像合成
        
        Note:
            mask画像のmode=('1','L','RGBA')に依って画像imとのブレンドの仕方が異なる
        
        Args:
            im (Image): 合成用の画像
            mask (Image): マスク画像
            flag (bool): Trueなら保存
        """
        im = im.resize(self.current.size,Image.LANCZOS) # im2を強制的にselfと同じサイズに変更
        im_composited: Image = Image.composite(self.current, im, mask)
        self.current = im_composited.copy()
        new_name: str = self.name_current + '_composited'
        if flag:
            im_composited.save(new_name + self.extension, quality=95)
        return im_composited


def combine_img_list(name_list: list[str], direction: str, color: tuple[int, int, int] | int = (255, 255, 255), flag: bool = False) -> Image:
    """連結(画像名群をリストで渡す)

    Args:
        name_list (list[str]): ファイル名のリスト
        deirection (str): 連結方向 'horizontal' or 'vertical'
        color (tuple[int, int, int] | int): 背景色
        flag (bool): Trueなら保存
    """
    im_list: list = []
    for i, name in enumerate(name_list):
        imi: Image = Image.open(name)
        im_list.append(imi)
    im0: Image = im_list[0]
    w0: int = im0.width
    h0: int = im0.height
    if im0.mode == 'L':
        color = 255
    dst: Image
    if direction == 'horizontal':
        dst = Image.new('RGB', (w0*len(name_list), h0), color)
        for i in range(len(im_list)):
            imi = im_list[i].resize((w0, h0), Image.LANCZOS)
            dst.paste(imi, (w0*i, 0))
    else:
        dst = Image.new('RGB', (w0, h0*len(name_list)), color)
        for i in range(len(im_list)):
            imi = im_list[i].resize((w0, h0), Image.LANCZOS)
            dst.paste(imi, (0, h0*i))
    name, extension = os.path.splitext(im0)
    new_name: str = name + '_multi_combined' + extension
    if flag:
        dst.save(new_name, quality=95)
    return dst

def combine_img_matrix(name_matrix: list[list[str]], color: tuple[int, int, int] | int = (255, 255, 255), flag: bool = False) -> Image:
    """連結(画像名群を行列で渡す)

    Args:
        name_matrix (list[list[str]]): ファイル名の行列
        color (tuple[int, int, int] | int): 背景色
        flag (bool): Trueなら保存
    """
    n_row: int = len(name_matrix)
    n_column: int = len(name_matrix[0])
    im_matrix: list = [[Image.open(name) for name in row] for row in name_matrix]
    im0: Image = im_matrix[0][0]
    w0: int = im0.width
    h0: int = im0.height
    if im0.mode == 'L':
        color = 255
    dst: Image = Image.new('RGB', (w0*n_column, h0*n_row), color)
    for i in range(n_row):
        for j in range(n_column):
            imij: Image = im_matrix[i][j].resize((w0, h0), Image.LANCZOS)
            dst.paste(imij, (w0*j, h0*i))
    name, extension = os.path.splitext(im0)
    new_name: str = name + '_matrix' + extension
    if flag:
        dst.save(new_name, quality=95)
    return dst

def make_gif(im_list: list, save_name: str, duration: float = 40., loop: int = 0) -> Image:
    """gif形式のアニメーションを作成

    Args:
        im_list (list[Image]): 時系列に沿って並んだ画像のリスト
        save_name (str): 保存先のファイル名(拡張子なし)
        duration (float): 画像の変更スピード(単位はミリ秒)
        loop (int): ループ回数(loop=0のとき無限ループ)
    """
    im_list[0].save(save_name + '.gif', save_all=True, append_images=im_list[1:], optimize=False, duration=duration, loop=loop)
    # optimize=Trueのとき予期せぬ動作が起こり得るらしい

def get_all_img() -> list:
    """フォルダ内の全画像(jpgかpng)を取得してlistとして出力
    """
    im_list: list = []
    current_directory_path: str = os.getcwd()
    files: list[str] = glob.glob(current_directory_path + '/*')
    files.sort()
    for f in files:
        if f[-4:] == '.jpg' or f[-4:] == '.jpeg' or f[-4:] == '.png':
            input_im: Image = Image.open(f)
            im_list.append(input_im)
    return im_list

def rename_ordered(prefix: str = '') -> None:
    """フォルダ内の全画像を取得して名前の順序を保存したまま連番に書き換える
    """
    current_directory_path: str = os.getcwd()
    files: list[str] = glob.glob(current_directory_path + '/*')
    files.sort()
    digit: int = len(str(len(files)))
    for i, f in enumerate(files):
        if f[-4:] == '.jpg' or f[-4:] == '.jpeg' or f[-4:] == '.png':
            os.rename(f, prefix + str(i).zfill(digit) + f[-4:])


def main() -> None:
    # 合成
    extension12 = '.jpg'
    name1 = 'def'
    im1 = Image.open(name1+extension12)
    name2 = 'ghi'
    im2 = Image.open(name2+extension12).resize(im1.size) ## ただサイズを調整
    #im2 = crop_max_square(Image.open(name2+extension12)) ## サイズに合うように切り出した

    mask = Image.new("L", im1.size, 0) # 一面真っ黒
    draw = ImageDraw.Draw(mask)
    draw.ellipse((140, 50, 260, 170), fill=255) # 半径120ピクセルの円盤を白でmaskの上に描き込んだ
    im12 = Image.composite(im1, im2, mask) # mask画像の各ピクセルの値に応じて合成

    mask_blur = mask.filter(ImageFilter.GaussianBlur(10)) # ImageFilterでマスク画像をぼかす
    im = Image.composite(im1, im2, mask_blur)

    mask = Image.open('*****.png').convert('L').resize(im1.size) # 画像を白黒にしてマスク画像として採用
    im = Image.composite(im1, im2, mask)
    # im = Image.blend(im1, im2, 0.5) #これは画像全面を一律の割合で合成(比率は0.0~1.0)



    ########## 描画について #############
    # 引数の説明
    # xy = ((左上のx座標, 左上のy座標), (右下のx座標, 右下のy座標))
    # or xy = (左上のx座標, 左上のy座標, 右下のx座標, 右下のy座標)

    # 以下,数字は0~255
    # fill = (R, G, B) # RGBのとき
    # fill = S # Lのとき

    # outline = (R, G, B) # RGBのとき
    # outline = S # Lのとき

    im = Image.new('RGB', (500, 300), (128, 128, 128)) # 今回はベタ画像。普通の画像に上描きしてよい
    draw1 = ImageDraw.Draw(im)
    # 楕円（円）: ellipse(xy, fill, outline)
    draw1.ellipse((100, 100, 150, 200), fill=(255, 0, 0), outline=(0, 0, 0))

    # 四角（長方形、正方形） : rectangle(xy, fill, outline)
    draw1.rectangle((200, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))

    # 直線 : line(xy, fill, width)
    draw1.line((350, 200, 450, 100), fill=(255, 255, 0), width=10)

    # 多角形 : polygon(xy, fill, outline)
    draw1.polygon(((200, 200), (300, 100), (250, 50)), fill=(255, 255, 0), outline=(0, 0, 0))

    # 点 : point(xy, fill)
    draw1.point(((350, 200), (450, 100), (400, 50)), fill=(255, 255, 0))

    # 円弧 : arc(xy, start, end, fill)
    draw1.arc((25, 50, 175, 200), start=30, end=270, fill=(255, 255, 0))

    # 弦（弓） : chord(xy, start, end, fill, outline)
    draw1.chord((225, 50, 375, 200), start=30, end=270, fill=(255, 255, 0), outline=(0, 0, 0))

    # 広義おうぎ形 : pieslice(xy, start, end, fill, outline)
    draw1.pieslice((425, 50, 575, 200), start=30, end=270, fill=(255, 255, 0), outline=(0, 0, 0))




if __name__ == "__main__":
    main()
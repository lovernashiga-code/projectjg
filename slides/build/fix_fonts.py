"""Thai text in PowerPoint uses the theme's complex-script font: point it (and the Thai script slot) at
Leelawadee UI instead of Office's default Angsana New / Cordia New."""
import re, sys, zipfile, shutil, os
src = sys.argv[1]
tmp = src + ".tmp"
FONT = "Leelawadee UI"
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
    for item in zi.infolist():
        data = zi.read(item.filename)
        if re.match(r"ppt/theme/theme\d+\.xml", item.filename):
            s = data.decode("utf-8")
            s = s.replace('<a:cs typeface=""/>', f'<a:cs typeface="{FONT}"/>')
            s = re.sub(r'<a:font script="Thai" typeface="[^"]*"/>', f'<a:font script="Thai" typeface="{FONT}"/>', s)
            data = s.encode("utf-8")
        zo.writestr(item, data)
shutil.move(tmp, src)
print("fonts fixed:", src)

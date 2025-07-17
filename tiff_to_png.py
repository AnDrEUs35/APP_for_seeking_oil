from osgeo import gdal
import os

dir1 = "C:/Users/Admin/Desktop/my_hell"
print(dir1)
for f in os.listdir(dir1):
    in_path = dir1+f
    out_path = 'C:/Users/Admin/Desktop/my_heaven'+os.path.splitext(f)[0]+'.png'
    print(out_path)
    ds = gdal.Translate(out_path, in_path, options="-scale -ot Byte")
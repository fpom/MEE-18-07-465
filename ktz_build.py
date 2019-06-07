import os, platform, stat, os.path, shutil, sys, numpy
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

if os.path.exists("ktz.so") :
    os.unlink("ktz.so")

sourcefiles = ["ktz.pyx", "ktzlib/ktzread.c", "ktzlib/ktzfree.c"]

extensions = [Extension("ktz", sourcefiles,
                        include_dirs = ["ktzlib", numpy.get_include()],
                        libraries = ["z"],
                        library_dirs = [])]

sys.argv[1:] = ["build_ext", "--inplace"]
setup(ext_modules = cythonize(extensions))

if not os.path.exists("ktz.so") :
    for dirpath, dirnames, filenames in os.walk("build") :
        if "ktz.so" in filenames :
            shutil.move(os.path.join(dirpath, "ktz.so"), ".")
            break

if platform.system() in ("Darwin", "Linux") :
    os.chmod("ktz.so",
             stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
             | stat.S_IRGRP | stat.S_IXGRP
             | stat.S_IROTH | stat.S_IXOTH)

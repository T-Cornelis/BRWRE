
# Setup.py for Cython


from distutils.core import setup
from Cython.Build import cythonize


# In the following add the NAME of the file
# which you want to compile.

setup(
    ext_modules = cythonize('brwre_2D.pyx', annotate = True)
)

# annotate = True generates HTML file.

# TO COMPILE: python setup.py build_ext -i --compiler=msvc
# THEN OPEN PYTHON (type "python")
# THEN IMPORT THE FILE (type "import hello") (or whatever name comes before .pyx!)

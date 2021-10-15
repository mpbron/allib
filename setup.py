# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import pathlib
import setuptools # type: ignore

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup( # type: ignore
    name="python-allib",
    version="0.1.5.6",
    description="A typed Active Learning Library",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Michiel Bron",
    author_email="m.p.bron@uu.nl",
    license="GNU LGPL v3",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={"allib": ["py.typed"]},
    packages=setuptools.find_packages(), # type: ignore
    python_requires=">=3.8",
    install_requires=[
        "numpy", 
        "pandas", 
        "h5py", 
        "tables", 
        "scikit-learn",
        "scipy", 
        "openpyxl", 
        "xlrd",
        "instancelib",
        "imblearn",
        "lightgbm",
        "gensim",
        "more-itertools",
        "matplotlib",
        "numba"],
    extras_require={
        "doc2vec": ["gensim"] 
    }
)
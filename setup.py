import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="AdaptToolBox",
    version="0.0.3dev",
    author="SiegfriedWagner",
    author_email="mateus.chojnowski@gmail.com",
    description="Containing methods used in cognitive psychology to modulate experiment stimulus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SiegfriedWagner/AdaptToolBox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    install_requires=['numpy', 'scipy']
)

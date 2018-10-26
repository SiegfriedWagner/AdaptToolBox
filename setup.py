import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="AdaptToolBox",
    version="0.0.3dev",
    author="SiegfriedWagner",
    author_email="zygfrydwagner@gmail.com",
    description="Containing methods used in cognitive psychology to modulate stimulus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SiegfriedWagner/AdaptToolBox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)

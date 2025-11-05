from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "DL_Object_Detection_Faster_RCNN_SSD"
AUTHOR_USER_NAME = "rezjsh"
SRC_REPO = "Object_Detection"
# AUTHOR_EMAIL = "your.email@example.com" # Replace with your actual email


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    # author_email=AUTHOR_EMAIL,
    description="A small package for Object Detection License Plate Recognition using Pytorch and Faster R-CNN and SSD models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": SRC_REPO},
    packages=find_packages(where=SRC_REPO),
    install_requires=[
        "roboflow",
        "python-dotenv",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.23",
        "Pillow>=9.0",
        "opencv-python>=4.5",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "pandas>=1.5",
        "PyYAML>=6.0",
        "tqdm>=4.64",
        "transformers>=4.20",
        "pytesseract>=0.3.9",
        "easyocr>=1.4.1",
    ],
    python_requires=">=3.7",
)

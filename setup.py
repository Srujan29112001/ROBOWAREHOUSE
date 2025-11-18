from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="robo-vla",
    version="1.0.0",
    author="RoboVLA Team",
    author_email="team@robovla.ai",
    description="Multimodal Robotic Vision-Language-Action System for Warehouse Automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/robo-vla",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "cloud": [
            "google-cloud-storage>=2.10.0",
            "boto3>=1.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "robo-vla=robo_vla.cli:main",
            "robo-vla-train=robo_vla.train:main",
            "robo-vla-serve=robo_vla.server.main:main",
        ],
    },
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transferable-dynamics-learning",
    version="1.0.0",
    author="Diego Agudelo",
    author_email="dagudelo@tuebingen.mpg.de",
    description="A benchmark for assessing transferability of "
                "dynamics learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rr-learning/transferable_dynamics_dataset",
    packages=setuptools.find_packages(include=['DL', 'DL.*']),
    entry_points={
        'console_scripts': [
            'compute_errors=DL.evaluation.evaluation:main',
            'plot_errors=DL.plotting.plots:main',
        ],
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
    ],
    python_requires='>=3.6',
)

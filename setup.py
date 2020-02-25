from setuptools import setup


setup(
    name='2passtools',
    version='0.2',
    description=(
        'two pass alignment of long noisy reads'
    ),
    author='Matthew Parker',
    entry_points={
        'console_scripts': [
            '2passtools = lib2pass.main:main',
        ]
    },
    packages=[
        'lib2pass',
    ],
    install_requires=[
        'numpy',
        'click',
        'click-log',
        'pysam',
        'ncls',
        'scikit-learn'
    ],
)
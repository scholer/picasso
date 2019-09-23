from setuptools import setup



setup(
    name='picasso',
    version='0.1.1dev2',
    author='Joerg Schnitzbauer',
    author_email='joschnitzbauer@gmail.com',
    url='https://gitlab.com/jungmannlab/picasso',
    packages=['picasso', 'picasso.gui'],
    entry_points={
        'console_scripts': [
            # console_scripts should all be lower-case, else you may get an error when uninstalling:
            'picasso=picasso.__main__:main',
        ],
        # 'gui_scripts': [
        #     'PicassoGUI=picasso:main',
        # ]
    },
    install_requires=[
        'matplotlib',
        'pyyaml',
        'lmfit',
        #'pyqt',
        'numpy',
        'scipy',
        'scikit-learn',
        'tqdm',
    ],
    classifiers=["Programming Language :: Python",
                 "Programming Language :: Python :: 3.5",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    package_data={'picasso': ['gui/icons/*.ico',
                              'config_template.yaml',
                              'base_sequences.csv',
                              'paint_sequences.csv']}
)

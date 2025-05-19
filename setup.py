"""
    Setup file for fast_mrt.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.3.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup,find_packages

if __name__ == "__main__":
    try:
        setup(use_scm_version={"version_scheme": "no-guess-dev"},
              install_requires=[
                "typer",
                "click",  # Required by Typer
                # Other dependencies...
            ],
            packages=find_packages(where="src"),
            entry_points={
                "console_scripts": [
                    "fit_rfd = scripts.fit_rfd:app", 
                    "fit_rfd_ori = scripts.fit_rfd_ori:app",
                    "bayesian = scripts.bayesian:app", 
                    "simulate = scripts.simulate:app",
                    "visu_bed = scripts.visu_bed:main",
                ],
    },
              )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise

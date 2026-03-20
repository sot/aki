from setuptools import setup

setup(
    name="aki",
    author="Tom Aldcroft",
    description="Fast Chandra star tracking simulator",
    author_email="taldcroft@cfa.harvard.edu",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    zip_safe=False,
    packages=["aki", "aki.tests"],
    tests_require=["pytest"],
)
